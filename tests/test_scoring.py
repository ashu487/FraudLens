from datetime import datetime
from app.models import Transaction
from app.scoring import score


def make(type="TRANSFER", amount=500.0, old_org=10_000.0, **kw):
    return Transaction(txn_id="t1", type=type, amount=amount, account_id="C1", dest_id="C2",
                       old_balance_org=old_org, new_balance_org=max(old_org - amount, 0),
                       old_balance_dest=0, new_balance_dest=0, timestamp=datetime.now(), **kw)


def test_normal_txn_approved():
    assert score(make()).decision == "approve"


def test_drained_account_blocked():
    r = score(make(amount=10_000.0, old_org=10_000.0))
    assert r.decision == "block"
    assert "account_drained" in r.reasons


def test_payment_never_flagged_as_drain():
    assert score(make(type="PAYMENT", amount=10_000.0, old_org=10_000.0)).decision == "approve"


def test_large_amount_alone_is_not_enough():
    r = score(make(amount=300_000.0, old_org=900_000.0))
    assert "large_risky_amount" in r.reasons and r.decision == "approve"
