from __future__ import annotations

from typing import Dict, Any, Set

from gst_env.models import Action

_WEIGHTS: Dict[str, float] = {
    "MATCHED": 0.20,
    "MISMATCH": 0.30,
    "MISSING_IN_2B": 0.30,
    "EXTRA_IN_2B": 0.20,
}

_VALID_MISMATCH_FIELDS: Set[str] = {
    "taxable_value",
    "invoice_date",
    "supplier_gstin",
    "cgst",
    "sgst",
    "igst",
    "itc_available",
}


def grade(action: Action, ground_truth: Dict[str, Any]) -> float:
    entries = action.reconciliation_result
    if not entries:
        return 0.0

    status_counts: Dict[str, int] = {k: 0 for k in _WEIGHTS}
    correct_counts: Dict[str, int] = {k: 0 for k in _WEIGHTS}
    field_score_total = 0.0
    field_entry_count = 0

    for e in entries:
        expected = ground_truth.get(e.invoice_id)
        if expected is None or expected not in status_counts:
            continue

        status_counts[expected] += 1

        if e.status == expected:
            correct_counts[expected] += 1

            if expected == "MISMATCH" and e.mismatch_fields:
                valid_reported = [
                    f for f in e.mismatch_fields
                    if f in _VALID_MISMATCH_FIELDS
                ]
                field_score_total += min(
                    1.0, len(valid_reported) / max(len(e.mismatch_fields), 1)
                )
                field_entry_count += 1

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

    field_bonus = (
        field_score_total / field_entry_count
        if field_entry_count > 0 else 0.0
    )

    confidence = float(getattr(action, "confidence", 0.5))
    confidence_bonus = confidence * 0.05

    return round(
        0.35 * weighted_acc
        + 0.35 * itc_score
        + 0.20 * field_bonus
        + 0.10 * confidence_bonus,
        4,
    )