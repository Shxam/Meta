from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple

from .data_generator import (
    generate_task1_data,
    generate_task2_data,
    generate_task3_data,
    generate_task4_data,
    generate_task5_data,
    generate_task6_data,
)
from .models import (
    Action,
    GSTR2BEntry,
    Invoice,
    Observation,
    Reward,
    RewardInfo,
    StateResponse,
)
from .graders import grade as grade_action

TASK_GENERATORS = {
    "task1_easy": generate_task1_data,
    "task2_medium": generate_task2_data,
    "task3_hard": generate_task3_data,
    "task4_credit_notes": generate_task4_data,
    "task5_stress": generate_task5_data,
    "task6_mixed_docs": generate_task6_data,
}

INSTRUCTIONS = (
    "Reconcile each purchase invoice against GSTR-2B entries. "
    "Classify each as MATCHED, MISMATCH, MISSING_IN_2B, or EXTRA_IN_2B. "
    "Compute claimable ITC as sum of (cgst+sgst+igst) for MATCHED invoices only."
)


class GSTReconciliationEnv:
    def __init__(self) -> None:
        self._task_id: Optional[str] = None
        self._episode_id: Optional[str] = None
        self._step_number: int = 0
        self._done: bool = False
        self._obs: Optional[Observation] = None
        self._ground_truth: Dict[str, Any] = {}
        self._raw_data: Dict[str, Any] = {}

    @property
    def current_task_id(self) -> Optional[str]:
        return self._task_id

    @property
    def ground_truth(self) -> Dict[str, Any]:
        return self._ground_truth

    def reset(self, task_id: str) -> Observation:
        if task_id not in TASK_GENERATORS:
            raise ValueError(
                f"Unknown task_id {task_id!r}. Valid: {list(TASK_GENERATORS.keys())}"
            )

        self._task_id = task_id
        self._episode_id = str(uuid.uuid4())
        self._step_number = 0
        self._done = False

        data = TASK_GENERATORS[task_id](seed=42)
        self._raw_data = data
        self._ground_truth = data["ground_truth"].copy()
        self._ground_truth["max_itc"] = data["max_itc"]
        if "penalty_days" in data:
            self._ground_truth["penalty_days"] = data["penalty_days"]

        tax_period = "2024-25"

        invoices = [Invoice(**inv) for inv in data["invoices"]]
        gstr2b = [GSTR2BEntry(**e) for e in data["gstr2b_entries"]]

        self._obs = Observation(
            task_id=task_id,
            episode_id=self._episode_id,
            invoices=invoices,
            gstr2b_entries=gstr2b,
            tax_period=tax_period,
            max_itc_possible=Decimal(str(data["max_itc"])),
            step_number=self._step_number,
            instructions=INSTRUCTIONS,
        )
        return self._obs

    def step(
        self, action: Action
    ) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._obs is None or self._task_id is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        self._step_number += 1
        self._done = True

        score = grade_action(self._task_id, action, self._ground_truth)

        total_invoices = len(self._obs.invoices)
        correct_matches = sum(
            1
            for e in action.reconciliation_result
            if self._ground_truth.get(e.invoice_id) == e.status
        )

        true_itc = float(self._ground_truth.get("max_itc", 0.0))
        pred_itc = float(action.claimable_itc)
        itc_error = abs(pred_itc - true_itc) / (true_itc + 1e-9)
        itc_score = max(0.0, 1.0 - itc_error)
        match_score = correct_matches / total_invoices if total_invoices else 0.0

        penalty_days = int(self._ground_truth.get("penalty_days", 0))
        penalty_day_penalty = max(0.0, 1.0 - penalty_days / 30.0)

        reward_info = RewardInfo(
            correct_matches=correct_matches,
            total_invoices=total_invoices,
            episode_id=self._episode_id or "",
            itc_error=round(itc_error, 4),
            task_score=round(score, 4),
        )

        reward = Reward(
            total=round(score, 4),
            match_score=round(match_score, 4),
            itc_score=round(itc_score, 4),
            false_positive_penalty=0.0,
            penalty_day_penalty=round(penalty_day_penalty, 4),
            done=self._done,
            info=reward_info,
        )

        info_dict = reward_info.model_dump()
        return self._obs, reward, self._done, info_dict

    def state(self) -> StateResponse:
        return StateResponse(
            task_id=self._task_id or "",
            episode_id=self._episode_id or "",
            step_number=self._step_number,
            done=self._done,
            has_active_episode=self._obs is not None and not self._done,
        )