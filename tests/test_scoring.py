import pytest
from conftest import make_txn
from app.rules import RuleEngine
from app.velocity import EMPTY


def test_normal_txn_approved(scorer):
    assert scorer.score(make_txn()).decision == "approve"


def test_exactly_drained_account_blocked(scorer):
    r = scorer.score(make_txn(amount=10_000.0, old_org=10_000.0))
    assert r.decision == "block" and "account_drained" in r.reasons


def test_amount_above_balance_is_NOT_drain(scorer):
    # Regression for the v1 false-positive bug (amount >= balance flagged legit txns).
    r = scorer.score(make_txn(amount=50_000.0, old_org=10_000.0))
    assert "account_drained" not in r.reasons and r.decision == "approve"


def test_payment_never_flagged_as_drain(scorer):
    assert scorer.score(make_txn(type="PAYMENT", amount=10_000.0, old_org=10_000.0)).decision == "approve"


def test_large_amount_alone_is_not_enough(scorer):
    r = scorer.score(make_txn(amount=300_000.0, old_org=900_000.0))
    assert "large_risky_amount" in r.reasons and r.decision == "approve"


def test_zero_amount_goes_to_review(scorer):
    r = scorer.score(make_txn(amount=0.0, old_org=100.0))
    assert "zero_amount" in r.reasons and r.decision == "review"


def test_velocity_burst_flags_fifth_txn_in_a_minute(scorer):
    decisions = [scorer.score(make_txn(f"b{i}", amount=100.0, at=i)).decision for i in range(6)]
    assert decisions[:4] == ["approve"] * 4
    assert decisions[4] == decisions[5] == "review"


def test_velocity_window_expires(scorer):
    for i in range(4):
        scorer.score(make_txn(f"w{i}", amount=100.0, at=i))
    late = scorer.score(make_txn("w4", amount=100.0, at=300))   # 5 min later
    assert "velocity_burst" not in late.reasons


def test_velocity_is_per_account(scorer):
    for i in range(10):
        r = scorer.score(make_txn(f"a{i}", amount=100.0, account=f"C{i}", at=i))
        assert r.decision == "approve"


def test_velocity_amount_needs_multiple_txns(scorer):
    single = scorer.score(make_txn("big", amount=2_000_000.0, old_org=9_000_000.0))
    assert "velocity_amount" not in single.reasons
    scorer.score(make_txn("m1", amount=600_000.0, old_org=9_000_000.0, account="C9", at=0))
    second = scorer.score(make_txn("m2", amount=600_000.0, old_org=9_000_000.0, account="C9", at=10))
    assert "velocity_amount" in second.reasons


def test_config_can_disable_a_rule():
    cfg = {"thresholds": {"review": 0.5, "block": 0.8},
           "rules": {"account_drained": {"weight": 0.85, "enabled": False}}}
    engine = RuleEngine(cfg)
    assert engine.score(make_txn(amount=10_000.0, old_org=10_000.0), EMPTY).decision == "approve"


def test_config_rejects_unknown_rule_and_bad_thresholds():
    with pytest.raises(ValueError):
        RuleEngine({"rules": {"nope": {"weight": 0.5}}})
    with pytest.raises(ValueError):
        RuleEngine({"thresholds": {"review": 0.9, "block": 0.5}, "rules": {}})
