import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "services" / "scoring"))
from datetime import datetime
from app.models import Transaction
from app.scoring import score


def make(amount):
    return Transaction(txn_id="t1", account_id="a", merchant_id="m",
                       amount=amount, timestamp=datetime.now())


def test_normal_txn_approved():
    assert score(make(500)).decision == "approve"


def test_high_amount_flagged():
    r = score(make(500_000))
    assert "high_amount" in r.reasons
