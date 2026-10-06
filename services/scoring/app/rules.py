"""Rules engine v1. Each rule: (name, weight, predicate).
Weights are summed and capped at 1.0 in scoring.py. Tune after the Week 4 model."""
from app.models import Transaction

RISKY_TYPES = {"TRANSFER", "CASH_OUT"}


def account_drained(t: Transaction) -> bool:
    # 97.8% of PaySim fraud empties the origin account (see notebook 01).
    return t.type in RISKY_TYPES and t.old_balance_org > 0 and t.amount >= t.old_balance_org


def large_risky_amount(t: Transaction) -> bool:
    return t.type in RISKY_TYPES and t.amount >= 200_000


def zero_amount(t: Transaction) -> bool:
    # Zero-amount fraud cluster seen in the amount histogram. Verify in the notebook.
    return t.type in RISKY_TYPES and t.amount == 0


RULES = [
    ("account_drained", 0.85, account_drained),
    ("large_risky_amount", 0.30, large_risky_amount),
    ("zero_amount", 0.30, zero_amount),
]


def evaluate(t: Transaction) -> list[tuple[str, float]]:
    return [(name, weight) for name, weight, fn in RULES if fn(t)]
