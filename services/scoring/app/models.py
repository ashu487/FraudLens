from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """Event schema. Anything that doesn't match is rejected by the consumer."""
    txn_id: str = Field(min_length=1)
    type: Literal["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
    amount: float = Field(ge=0)
    account_id: str = Field(min_length=1)
    dest_id: str = Field(min_length=1)
    old_balance_org: float = Field(ge=0)
    new_balance_org: float = Field(ge=0)
    old_balance_dest: float = Field(ge=0)
    new_balance_dest: float = Field(ge=0)
    step: int | None = None
    timestamp: datetime
    # Ground-truth label, present only in simulated data. Used for evaluation,
    # NEVER for scoring.
    is_fraud: bool | None = None


class ScoreResult(BaseModel):
    txn_id: str
    risk_score: float          # 0..1
    decision: str              # approve | review | block
    reasons: list[str] = []
