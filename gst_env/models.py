from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import List, Optional, Literal

from pydantic import BaseModel, Field, field_validator


ConfidenceFloat = float


class Invoice(BaseModel):
    model_config = {"extra": "forbid"}

    invoice_id: str = Field(...)
    vendor_gstin: str = Field(...)
    invoice_number: str = Field(...)
    invoice_date: date = Field(...)
    taxable_value: Decimal = Field(...)
    cgst: Decimal = Field(...)
    sgst: Decimal = Field(...)
    igst: Decimal = Field(...)
    hsn_code: str = Field(...)
    vendor_name: str = Field(...)

    @field_validator("vendor_gstin")
    @classmethod
    def validate_gstin(cls, v: str) -> str:
        if len(v) != 15:
            raise ValueError(
                f"GSTIN must be 15 characters, got {len(v)}: {v}"
            )
        return v.upper()


class GSTR2BEntry(BaseModel):
    model_config = {"extra": "forbid"}

    supplier_gstin: str = Field(...)
    invoice_number: str = Field(...)
    invoice_date: date = Field(...)
    taxable_value: Decimal = Field(...)
    cgst: Decimal = Field(...)
    sgst: Decimal = Field(...)
    igst: Decimal = Field(...)
    itc_available: bool = Field(...)
    document_type: str = Field(default="invoice")

    @field_validator("supplier_gstin")
    @classmethod
    def validate_gstin(cls, v: str) -> str:
        if len(v) != 15:
            raise ValueError(
                f"GSTIN must be 15 characters, got {len(v)}: {v}"
            )
        return v.upper()


class ReconciliationEntry(BaseModel):
    model_config = {"extra": "forbid"}

    invoice_id: str = Field(...)
    status: Literal["MATCHED", "MISMATCH", "MISSING_IN_2B", "EXTRA_IN_2B"] = Field(...)
    correction_note: Optional[str] = Field(default=None)
    mismatch_fields: List[str] = Field(default_factory=list)


class Action(BaseModel):
    model_config = {"extra": "forbid"}

    reconciliation_result: List[ReconciliationEntry] = Field(...)
    claimable_itc: Decimal = Field(...)
    confidence: float = Field(ge=0.0, le=1.0)


class Observation(BaseModel):
    model_config = {"extra": "forbid"}

    task_id: str = Field(...)
    episode_id: str = Field(...)
    invoices: List[Invoice] = Field(...)
    gstr2b_entries: List[GSTR2BEntry] = Field(...)
    tax_period: str = Field(...)
    max_itc_possible: Decimal = Field(...)
    step_number: int = Field(...)
    instructions: str = Field(...)


class RewardInfo(BaseModel):
    model_config = {"extra": "forbid"}

    correct_matches: int
    total_invoices: int
    episode_id: str
    itc_error: float = Field(default=0.0)
    task_score: float = Field(default=0.0)


class Reward(BaseModel):
    model_config = {"extra": "forbid"}

    total: float = Field(...)
    match_score: float = Field(...)
    itc_score: float = Field(...)
    false_positive_penalty: float = Field(default=0.0)
    penalty_day_penalty: float = Field(default=0.0)
    done: bool = Field(...)
    info: RewardInfo = Field(...)


class TaskInfo(BaseModel):
    model_config = {"extra": "forbid"}

    task_id: str
    description: str
    difficulty: str
    num_invoices: int
    invoice_range: str


class StateResponse(BaseModel):
    model_config = {"extra": "forbid"}

    task_id: str
    episode_id: str
    step_number: int
    done: bool
    has_active_episode: bool