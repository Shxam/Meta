from __future__ import annotations

import random
from decimal import Decimal
from typing import Dict, List, Any

from .env import GSTReconciliationEnv
from .models import Action, ReconciliationEntry
from .graders import (
    grade_task1,
    grade_task2,
    grade_task3,
    grade_task4,
    grade_task5,
    grade_task6,
)


def run_baseline() -> Dict[str, Any]:
    base_env = GSTReconciliationEnv()
    scores: Dict[str, float] = {}

    # Task 1 - all MATCHED, full ITC
    obs = base_env.reset("task1_easy")
    entries = [
        ReconciliationEntry(
            invoice_id=i.invoice_id,
            status="MATCHED",
            mismatch_fields=[],
        )
        for i in obs.invoices
    ]
    action = Action(
        reconciliation_result=entries,
        claimable_itc=obs.max_itc_possible,
        confidence=1.0,
    )
    scores["task1_easy"] = grade_task1(action, base_env.ground_truth)

    # Task 2 - random subset flagged as MISMATCH
    obs = base_env.reset("task2_medium")
    all_ids = [i.invoice_id for i in obs.invoices]
    mismatch_ids = set(random.sample(all_ids, min(8, len(all_ids))))
    claim = Decimal("0")
    entries = []
    for inv in obs.invoices:
        if inv.invoice_id in mismatch_ids:
            entries.append(
                ReconciliationEntry(
                    invoice_id=inv.invoice_id,
                    status="MISMATCH",
                    mismatch_fields=["taxable_value"],
                )
            )
        else:
            entries.append(
                ReconciliationEntry(
                    invoice_id=inv.invoice_id,
                    status="MATCHED",
                    mismatch_fields=[],
                )
            )
            claim += inv.cgst + inv.sgst + inv.igst

    action = Action(
        reconciliation_result=entries,
        claimable_itc=claim,
        confidence=0.5,
    )
    scores["task2_medium"] = grade_task2(action, base_env.ground_truth)

    # Task 3 - index-based heuristic classification
    obs = base_env.reset("task3_hard")
    claim = Decimal("0")
    entries = []
    for idx, inv in enumerate(obs.invoices):
        if idx < 20:
            status_val = "MISSING_IN_2B"
        elif idx < 55:
            status_val = "MISMATCH"
        elif idx < 65:
            status_val = "EXTRA_IN_2B"
        else:
            status_val = "MATCHED"
            claim += inv.cgst + inv.sgst + inv.igst

        entries.append(
            ReconciliationEntry(
                invoice_id=inv.invoice_id,
                status=status_val,
                mismatch_fields=[],
            )
        )

    action = Action(
        reconciliation_result=entries,
        claimable_itc=claim,
        confidence=0.3,
    )
    scores["task3_hard"] = grade_task3(action, base_env.ground_truth)

    # Task 4 - credit notes
    obs = base_env.reset("task4_credit_notes")
    claim = Decimal("0")
    entries = []
    for idx, inv in enumerate(obs.invoices):
        if idx < 10:
            status_val = "MISSING_IN_2B"
        elif idx < 17:
            status_val = "MISMATCH"
        else:
            status_val = "MATCHED"
            claim += inv.cgst + inv.sgst + inv.igst

        entries.append(
            ReconciliationEntry(
                invoice_id=inv.invoice_id,
                status=status_val,
                mismatch_fields=["taxable_value"] if status_val == "MISMATCH" else [],
            )
        )

    action = Action(
        reconciliation_result=entries,
        claimable_itc=claim,
        confidence=0.4,
    )
    scores["task4_credit_notes"] = grade_task4(action, base_env.ground_truth)

    # Task 5 - stress (500 invoices)
    obs = base_env.reset("task5_stress")
    claim = Decimal("0")
    entries = []
    for idx, inv in enumerate(obs.invoices):
        if idx < 60:
            status_val = "MISSING_IN_2B"
        elif idx < 130:
            status_val = "MISMATCH"
        elif idx < 155:
            status_val = "EXTRA_IN_2B"
        else:
            status_val = "MATCHED"
            claim += inv.cgst + inv.sgst + inv.igst

        entries.append(
            ReconciliationEntry(
                invoice_id=inv.invoice_id,
                status=status_val,
                mismatch_fields=["taxable_value"] if status_val == "MISMATCH" else [],
            )
        )

    action = Action(
        reconciliation_result=entries,
        claimable_itc=claim,
        confidence=0.3,
    )
    scores["task5_stress"] = grade_task5(action, base_env.ground_truth)

    # Task 6 - mixed document types
    obs = base_env.reset("task6_mixed_docs")
    claim = Decimal("0")
    entries = []
    for idx, inv in enumerate(obs.invoices):
        if idx < 20:
            status_val = "MISSING_IN_2B"
        elif idx < 55:
            status_val = "MISMATCH"
        elif idx < 70:
            status_val = "EXTRA_IN_2B"
        else:
            status_val = "MATCHED"
            claim += inv.cgst + inv.sgst + inv.igst

        entries.append(
            ReconciliationEntry(
                invoice_id=inv.invoice_id,
                status=status_val,
                mismatch_fields=["taxable_value"] if status_val == "MISMATCH" else [],
            )
        )

    action = Action(
        reconciliation_result=entries,
        claimable_itc=claim,
        confidence=0.35,
    )
    scores["task6_mixed_docs"] = grade_task6(action, base_env.ground_truth)

    task_scores = {k: v for k, v in scores.items()}
    avg = round(sum(task_scores.values()) / len(task_scores), 4)
    scores["average"] = avg
    return scores


if __name__ == "__main__":
    results = run_baseline()
    print("\nBaseline Results:")
    print("=" * 40)
    for k, v in results.items():
        print(f"  {k:<25}: {v:.4f}")