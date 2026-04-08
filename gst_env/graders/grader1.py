from __future__ import annotations

from typing import Dict, Any

from gst_env.models import Action


def grade(action: Action, ground_truth: Dict[str, Any]) -> float:
    entries = action.reconciliation_result
    if not entries:
        return 0.0

    total = len(entries)
    correct = sum(
        1 for e in entries if ground_truth.get(e.invoice_id) == e.status
    )
    accuracy = correct / total

    true_itc = float(ground_truth.get("max_itc", 0.0))
    pred_itc = float(action.claimable_itc)
    itc_error = abs(pred_itc - true_itc) / (true_itc + 1e-9)
    itc_score = max(0.0, 1.0 - itc_error)

    return round(0.7 * accuracy + 0.3 * itc_score, 4)