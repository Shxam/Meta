from __future__ import annotations

from typing import Dict, Any

from gst_env.models import Action

_WEIGHTS: Dict[str, float] = {
    "MATCHED": 0.2,
    "MISMATCH": 0.3,
    "MISSING_IN_2B": 0.3,
    "EXTRA_IN_2B": 0.2,
}


def grade(action: Action, ground_truth: Dict[str, Any]) -> float:
    entries = action.reconciliation_result
    if not entries:
        return 0.0

    status_counts: Dict[str, int] = {k: 0 for k in _WEIGHTS}
    correct_counts: Dict[str, int] = {k: 0 for k in _WEIGHTS}

    for e in entries:
        expected = ground_truth.get(e.invoice_id)
        if expected is None or expected not in status_counts:
            continue
        status_counts[expected] += 1
        if e.status == expected:
            correct_counts[expected] += 1

    weighted_acc = 0.0
    for k, w in _WEIGHTS.items():
        if status_counts[k] > 0:
            weighted_acc += w * (correct_counts[k] / status_counts[k])
        else:
            weighted_acc += w

    true_itc = float(ground_truth.get("max_itc", 0.0))
    pred_itc = float(action.claimable_itc)
    itc_error = abs(pred_itc - true_itc) / (true_itc + 1e-9)
    itc_score = max(0.0, 1.0 - itc_error)

    penalty_days = ground_truth.get("penalty_days", 0)
    penalty_score = max(0.0, 1.0 - penalty_days / 30.0)

    return round(0.4 * weighted_acc + 0.4 * itc_score + 0.2 * penalty_score, 4)