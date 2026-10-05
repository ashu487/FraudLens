from datetime import datetime
from pydantic import BaseModel


class Transaction(BaseModel):
    txn_id: str
    account_id: str
    merchant_id: str
    amount: float
    currency: str = "INR"
    country: str = "IN"
    device_id: str | None = None
    timestamp: datetime


class ScoreResult(BaseModel):
    txn_id: str
    risk_score: float          # 0..1
    decision: str              # approve | review | block
    reasons: list[str] = []
