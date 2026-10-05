"""Placeholder scorer. Rules engine arrives in Week 3, ML model in Week 4."""
from app.models import Transaction, ScoreResult


def score(txn: Transaction) -> ScoreResult:
    reasons = []
    risk = 0.0
    if txn.amount > 100_000:
        risk += 0.6
        reasons.append("high_amount")
    decision = "block" if risk >= 0.8 else "review" if risk >= 0.5 else "approve"
    return ScoreResult(txn_id=txn.txn_id, risk_score=risk, decision=decision, reasons=reasons)
