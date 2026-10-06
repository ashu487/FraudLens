from app.models import Transaction, ScoreResult
from app import rules

REVIEW_THRESHOLD = 0.5
BLOCK_THRESHOLD = 0.8


def score(txn: Transaction) -> ScoreResult:
    hits = rules.evaluate(txn)
    risk = min(1.0, sum(w for _, w in hits))
    decision = "block" if risk >= BLOCK_THRESHOLD else "review" if risk >= REVIEW_THRESHOLD else "approve"
    return ScoreResult(
        txn_id=txn.txn_id,
        risk_score=round(risk, 3),
        decision=decision,
        reasons=[name for name, _ in hits],
    )
