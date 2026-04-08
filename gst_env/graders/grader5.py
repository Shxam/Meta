from __future__ import annotations

from typing import Dict, Any, Set

from gst_env.models import Action

_WEIGHTS: Dict[str, float] = {
    "MATCHED": 0.10,
    "MISMATCH": 0.40,
    "MISSING_IN_2B": 0.40,
    "EXTRA_IN_2B": 0.10,
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
    false_positive_count = 0

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

        if e.status == "MATCHED" and expected in ("MISMATCH", "MISSING_IN_2B"):
            false_positive_count += 1

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

    total = len(entries)
    fp_rate = false_positive_count / total if total else 0.0
    fp_penalty = max(0.0, 1.0 - fp_rate * 3)

    penalty_days = int(ground_truth.get("penalty_days", 0))
    penalty_score = max(0.0, 1.0 - penalty_days / 30.0)

    return round(
        0.30 * weighted_acc
        + 0.30 * itc_score
        + 0.15 * field_bonus
        + 0.15 * fp_penalty
        + 0.10 * penalty_score,
        4,
    )