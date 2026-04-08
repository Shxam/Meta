from __future__ import annotations

import copy
import random
import string
from datetime import timedelta, date
from decimal import Decimal
from typing import Dict, List, Any, Optional, Set

from faker import Faker

from .models import Invoice, GSTR2BEntry

fake = Faker("en_IN")

HSN_CODES = ["6403", "8471", "3004", "9401", "8517", "2106", "6204"]

INDIAN_VENDOR_NAMES = [
    "Tata Steel Limited",
    "Reliance Industries Pvt Ltd",
    "Infosys Technologies Ltd",
    "Wipro Consumer Care",
    "Mahindra & Mahindra Ltd",
    "Hindustan Unilever Limited",
    "Bajaj Auto Ltd",
    "Larsen & Toubro Limited",
    "Godrej Industries Ltd",
    "Ashok Leyland Ltd",
    "Bharat Electronics Limited",
    "Sun Pharmaceutical Industries",
    "Dr. Reddy's Laboratories",
    "Cipla Limited",
    "Hero MotoCorp Ltd",
    "Maruti Suzuki India Ltd",
    "Adani Enterprises Limited",
    "JSW Steel Limited",
    "Vedanta Resources Ltd",
    "Grasim Industries Limited",
    "UltraTech Cement Ltd",
    "Dabur India Limited",
    "Marico Limited",
    "Britannia Industries Ltd",
]

STATE_CODES: List[int] = list(range(1, 38))
BUYER_STATE_CODE: int = 29


def generate_gstin(
    state_code: int = 29,
    rng: Optional[random.Random] = None,
) -> str:
    _rng = rng if rng is not None else random
    state_prefix = f"{state_code:02d}"
    pan_letters = "".join(_rng.choices(string.ascii_uppercase, k=5))
    pan_digits = "".join(_rng.choices(string.digits, k=4))
    pan_last = _rng.choice(string.ascii_uppercase)
    entity_num = str(_rng.randint(1, 9))
    checksum = _rng.choice(string.digits + string.ascii_uppercase)
    gstin = (
        state_prefix + pan_letters + pan_digits
        + pan_last + entity_num + "Z" + checksum
    )
    assert len(gstin) == 15, f"Generated GSTIN length {len(gstin)}: {gstin}"
    return gstin


def generate_invoice(invoice_id: str, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    fake.seed_instance(seed)

    vendor = rng.choice(INDIAN_VENDOR_NAMES)
    hsn = rng.choice(HSN_CODES)
    taxable = Decimal(str(round(rng.uniform(1000, 500000), 2)))

    supplier_state = rng.choice(STATE_CODES)
    is_intra_state = supplier_state == BUYER_STATE_CODE
    gstin = generate_gstin(supplier_state, rng)

    if is_intra_state:
        cgst = (taxable * Decimal("0.09")).quantize(Decimal("0.01"))
        sgst = (taxable * Decimal("0.09")).quantize(Decimal("0.01"))
        igst = Decimal("0.00")
    else:
        cgst = Decimal("0.00")
        sgst = Decimal("0.00")
        igst = (taxable * Decimal("0.18")).quantize(Decimal("0.01"))

    inv_date: date = fake.date_between(
        start_date=date(2024, 4, 1),
        end_date=date(2025, 3, 31),
    )

    invoice = Invoice(
        invoice_id=invoice_id,
        vendor_gstin=gstin,
        invoice_number=f"INV-{invoice_id}",
        invoice_date=inv_date,
        taxable_value=taxable,
        cgst=cgst,
        sgst=sgst,
        igst=igst,
        hsn_code=hsn,
        vendor_name=vendor,
    )
    return invoice.model_dump()


def _gstr_from_invoice(inv: Dict[str, Any]) -> Dict[str, Any]:
    entry = GSTR2BEntry(
        supplier_gstin=inv["vendor_gstin"],
        invoice_number=inv["invoice_number"],
        invoice_date=inv["invoice_date"],
        taxable_value=Decimal(str(inv["taxable_value"])),
        cgst=Decimal(str(inv["cgst"])),
        sgst=Decimal(str(inv["sgst"])),
        igst=Decimal(str(inv["igst"])),
        itc_available=True,
        document_type="invoice",
    )
    return entry.model_dump()


def _deep_copy_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(entry)


def _calc_itc(inv: Dict[str, Any]) -> Decimal:
    return (
        Decimal(str(inv["cgst"]))
        + Decimal(str(inv["sgst"]))
        + Decimal(str(inv["igst"]))
    )


def generate_task1_data(seed: int = 42) -> Dict[str, Any]:
    invoices: List[Dict[str, Any]] = []
    gstr: List[Dict[str, Any]] = []
    gt: Dict[str, str] = {}
    max_itc = Decimal("0")

    for i in range(10):
        inv = generate_invoice(f"T1-{i:04d}", seed + i)
        invoices.append(inv)
        gstr.append(_gstr_from_invoice(inv))
        gt[inv["invoice_id"]] = "MATCHED"
        max_itc += _calc_itc(inv)

    return {
        "invoices": invoices,
        "gstr2b_entries": gstr,
        "ground_truth": gt,
        "max_itc": float(max_itc),
    }


def generate_task2_data(seed: int = 42) -> Dict[str, Any]:
    rng = random.Random(seed)
    invoices: List[Dict[str, Any]] = []
    gstr: List[Dict[str, Any]] = []
    gt: Dict[str, str] = {}
    mismatch_indices: Set[int] = set(rng.sample(range(50), 8))

    for i in range(50):
        inv = generate_invoice(f"T2-{i:04d}", seed + i)
        invoices.append(inv)

        if i in mismatch_indices:
            mtype = rng.choice(["amount", "date", "gstin", "missing"])

            if mtype == "missing":
                gt[inv["invoice_id"]] = "MISSING_IN_2B"
                continue

            entry = _gstr_from_invoice(inv)

            if mtype == "amount":
                factor = Decimal(str(round(rng.uniform(0.85, 1.15), 6)))
                new_taxable = (
                    Decimal(str(inv["taxable_value"])) * factor
                ).quantize(Decimal("0.01"))
                entry["taxable_value"] = new_taxable

            elif mtype == "date":
                original_date: date = inv["invoice_date"]
                shift_days = rng.randint(5, 30)
                shifted_date = original_date + timedelta(days=shift_days)
                entry["invoice_date"] = min(shifted_date, date(2025, 3, 31))

            elif mtype == "gstin":
                gstin_chars = list(entry["supplier_gstin"])
                pos = rng.randint(2, 11)
                original_char = gstin_chars[pos]
                pool = [
                    c
                    for c in (string.ascii_uppercase + string.digits)
                    if c != original_char
                ]
                gstin_chars[pos] = rng.choice(pool)
                entry["supplier_gstin"] = "".join(gstin_chars)

            gstr.append(entry)
            gt[inv["invoice_id"]] = "MISMATCH"
        else:
            gstr.append(_gstr_from_invoice(inv))
            gt[inv["invoice_id"]] = "MATCHED"

    max_itc = sum(_calc_itc(inv) for inv in invoices if gt.get(inv["invoice_id"]) == "MATCHED")
    return {
        "invoices": invoices,
        "gstr2b_entries": gstr,
        "ground_truth": gt,
        "max_itc": float(max_itc),
    }


def generate_task3_data(seed: int = 42) -> Dict[str, Any]:
    rng = random.Random(seed)
    invoices: List[Dict[str, Any]] = []
    gstr: List[Dict[str, Any]] = []
    gt: Dict[str, str] = {}

    all_indices = list(range(200))
    rng.shuffle(all_indices)

    missing_set: Set[int] = set(all_indices[:20])
    amount_set: Set[int] = set(all_indices[20:35])
    dup_set: Set[int] = set(all_indices[35:45])
    gstin_set: Set[int] = set(all_indices[45:55])
    reverse_set: Set[int] = set(all_indices[55:60])

    for i in range(200):
        inv = generate_invoice(f"T3-{i:04d}", seed + i)
        invoices.append(inv)

        if i in missing_set:
            gt[inv["invoice_id"]] = "MISSING_IN_2B"
            continue

        entry = _gstr_from_invoice(inv)

        if i in amount_set:
            factor = Decimal(str(round(rng.uniform(1.2, 1.5), 6)))
            new_taxable = (
                Decimal(str(inv["taxable_value"])) * factor
            ).quantize(Decimal("0.01"))
            entry["taxable_value"] = new_taxable
            gt[inv["invoice_id"]] = "MISMATCH"
            gstr.append(entry)

        elif i in dup_set:
            gstr.append(_deep_copy_entry(entry))
            gstr.append(_deep_copy_entry(entry))
            gt[inv["invoice_id"]] = "EXTRA_IN_2B"

        elif i in gstin_set:
            gstin_chars = list(entry["supplier_gstin"])
            pos = rng.randint(2, 11)
            original_char = gstin_chars[pos]
            pool = [
                c
                for c in (string.ascii_uppercase + string.digits)
                if c != original_char
            ]
            gstin_chars[pos] = rng.choice(pool)
            entry["supplier_gstin"] = "".join(gstin_chars)
            gt[inv["invoice_id"]] = "MISMATCH"
            gstr.append(entry)

        elif i in reverse_set:
            entry["itc_available"] = False
            gt[inv["invoice_id"]] = "MISMATCH"
            gstr.append(entry)

        else:
            gt[inv["invoice_id"]] = "MATCHED"
            gstr.append(entry)

    max_itc = sum(_calc_itc(inv) for inv in invoices if gt.get(inv["invoice_id"]) == "MATCHED")
    penalty_days = rng.randint(0, 30)

    return {
        "invoices": invoices,
        "gstr2b_entries": gstr,
        "ground_truth": gt,
        "max_itc": float(max_itc),
        "penalty_days": penalty_days,
    }


def generate_task4_data(seed: int = 42) -> Dict[str, Any]:
    rng = random.Random(seed)
    invoices: List[Dict[str, Any]] = []
    gstr: List[Dict[str, Any]] = []
    gt: Dict[str, str] = {}

    all_indices = list(range(75))
    rng.shuffle(all_indices)

    missing_set: Set[int] = set(all_indices[:10])
    amount_set: Set[int] = set(all_indices[10:20])
    gstin_set: Set[int] = set(all_indices[20:28])
    itc_set: Set[int] = set(all_indices[28:35])

    for i in range(75):
        inv = generate_invoice(f"T4-{i:04d}", seed + i)
        invoices.append(inv)

        if i in missing_set:
            gt[inv["invoice_id"]] = "MISSING_IN_2B"
            continue

        entry = _gstr_from_invoice(inv)

        if i in amount_set:
            factor = Decimal(str(round(rng.uniform(0.80, 1.20), 6)))
            entry["taxable_value"] = (
                Decimal(str(inv["taxable_value"])) * factor
            ).quantize(Decimal("0.01"))
            gt[inv["invoice_id"]] = "MISMATCH"
            gstr.append(entry)

        elif i in gstin_set:
            gstin_chars = list(entry["supplier_gstin"])
            pos = rng.randint(2, 11)
            original_char = gstin_chars[pos]
            pool = [
                c for c in (string.ascii_uppercase + string.digits)
                if c != original_char
            ]
            gstin_chars[pos] = rng.choice(pool)
            entry["supplier_gstin"] = "".join(gstin_chars)
            gt[inv["invoice_id"]] = "MISMATCH"
            gstr.append(entry)

        elif i in itc_set:
            entry["itc_available"] = False
            gt[inv["invoice_id"]] = "MISMATCH"
            gstr.append(entry)

        else:
            gt[inv["invoice_id"]] = "MATCHED"
            gstr.append(entry)

    max_itc = sum(
        _calc_itc(inv) for inv in invoices
        if gt.get(inv["invoice_id"]) == "MATCHED"
    )
    penalty_days = rng.randint(0, 15)

    return {
        "invoices": invoices,
        "gstr2b_entries": gstr,
        "ground_truth": gt,
        "max_itc": float(max_itc),
        "penalty_days": penalty_days,
    }


def generate_task5_data(seed: int = 42) -> Dict[str, Any]:
    rng = random.Random(seed)
    invoices: List[Dict[str, Any]] = []
    gstr: List[Dict[str, Any]] = []
    gt: Dict[str, str] = {}

    all_indices = list(range(500))
    rng.shuffle(all_indices)

    missing_set: Set[int] = set(all_indices[:60])
    amount_set: Set[int] = set(all_indices[60:100])
    dup_set: Set[int] = set(all_indices[100:125])
    gstin_set: Set[int] = set(all_indices[125:155])
    itc_set: Set[int] = set(all_indices[155:170])
    date_set: Set[int] = set(all_indices[170:190])

    for i in range(500):
        inv = generate_invoice(f"T5-{i:04d}", seed + i)
        invoices.append(inv)

        if i in missing_set:
            gt[inv["invoice_id"]] = "MISSING_IN_2B"
            continue

        entry = _gstr_from_invoice(inv)

        if i in amount_set:
            factor = Decimal(str(round(rng.uniform(1.1, 1.5), 6)))
            entry["taxable_value"] = (
                Decimal(str(inv["taxable_value"])) * factor
            ).quantize(Decimal("0.01"))
            gt[inv["invoice_id"]] = "MISMATCH"
            gstr.append(entry)

        elif i in dup_set:
            gstr.append(_deep_copy_entry(entry))
            gstr.append(_deep_copy_entry(entry))
            gt[inv["invoice_id"]] = "EXTRA_IN_2B"

        elif i in gstin_set:
            gstin_chars = list(entry["supplier_gstin"])
            pos = rng.randint(2, 11)
            original_char = gstin_chars[pos]
            pool = [
                c for c in (string.ascii_uppercase + string.digits)
                if c != original_char
            ]
            gstin_chars[pos] = rng.choice(pool)
            entry["supplier_gstin"] = "".join(gstin_chars)
            gt[inv["invoice_id"]] = "MISMATCH"
            gstr.append(entry)

        elif i in itc_set:
            entry["itc_available"] = False
            gt[inv["invoice_id"]] = "MISMATCH"
            gstr.append(entry)

        elif i in date_set:
            original_date: date = inv["invoice_date"]
            shift = rng.randint(10, 45)
            shifted = original_date + timedelta(days=shift)
            entry["invoice_date"] = min(shifted, date(2025, 3, 31))
            gt[inv["invoice_id"]] = "MISMATCH"
            gstr.append(entry)

        else:
            gt[inv["invoice_id"]] = "MATCHED"
            gstr.append(entry)

    max_itc = sum(
        _calc_itc(inv) for inv in invoices
        if gt.get(inv["invoice_id"]) == "MATCHED"
    )
    penalty_days = rng.randint(0, 30)

    return {
        "invoices": invoices,
        "gstr2b_entries": gstr,
        "ground_truth": gt,
        "max_itc": float(max_itc),
        "penalty_days": penalty_days,
    }


def generate_task6_data(seed: int = 42) -> Dict[str, Any]:
    rng = random.Random(seed)
    invoices: List[Dict[str, Any]] = []
    gstr: List[Dict[str, Any]] = []
    gt: Dict[str, str] = {}

    all_indices = list(range(150))
    rng.shuffle(all_indices)

    missing_set: Set[int] = set(all_indices[:20])
    amount_set: Set[int] = set(all_indices[20:40])
    dup_set: Set[int] = set(all_indices[40:55])
    gstin_set: Set[int] = set(all_indices[55:70])
    date_set: Set[int] = set(all_indices[70:85])
    itc_set: Set[int] = set(all_indices[85:95])

    for i in range(150):
        inv = generate_invoice(f"T6-{i:04d}", seed + i)
        invoices.append(inv)

        if i in missing_set:
            gt[inv["invoice_id"]] = "MISSING_IN_2B"
            continue

        entry = _gstr_from_invoice(inv)

        if i in amount_set:
            factor = Decimal(str(round(rng.uniform(0.75, 1.25), 6)))
            entry["taxable_value"] = (
                Decimal(str(inv["taxable_value"])) * factor
            ).quantize(Decimal("0.01"))
            gt[inv["invoice_id"]] = "MISMATCH"
            gstr.append(entry)

        elif i in dup_set:
            gstr.append(_deep_copy_entry(entry))
            gstr.append(_deep_copy_entry(entry))
            gt[inv["invoice_id"]] = "EXTRA_IN_2B"

        elif i in gstin_set:
            gstin_chars = list(entry["supplier_gstin"])
            pos = rng.randint(2, 11)
            original_char = gstin_chars[pos]
            pool = [
                c for c in (string.ascii_uppercase + string.digits)
                if c != original_char
            ]
            gstin_chars[pos] = rng.choice(pool)
            entry["supplier_gstin"] = "".join(gstin_chars)
            gt[inv["invoice_id"]] = "MISMATCH"
            gstr.append(entry)

        elif i in date_set:
            original_date: date = inv["invoice_date"]
            shift = rng.randint(5, 35)
            shifted = original_date + timedelta(days=shift)
            entry["invoice_date"] = min(shifted, date(2025, 3, 31))
            gt[inv["invoice_id"]] = "MISMATCH"
            gstr.append(entry)

        elif i in itc_set:
            entry["itc_available"] = False
            gt[inv["invoice_id"]] = "MISMATCH"
            gstr.append(entry)

        else:
            gt[inv["invoice_id"]] = "MATCHED"
            gstr.append(entry)

    max_itc = sum(
        _calc_itc(inv) for inv in invoices
        if gt.get(inv["invoice_id"]) == "MATCHED"
    )
    penalty_days = rng.randint(0, 20)

    return {
        "invoices": invoices,
        "gstr2b_entries": gstr,
        "ground_truth": gt,
        "max_itc": float(max_itc),
        "penalty_days": penalty_days,
    }