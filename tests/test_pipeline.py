import json
from conftest import RULES_PATH
from app.adapters.storage import DuckDBStorage
from app.pipeline import Processor
from app.scoring import Scorer


def event(txn_id="t1", **over):
    e = {"txn_id": txn_id, "type": "TRANSFER", "amount": 1000.0, "account_id": "C1", "dest_id": "C2",
         "old_balance_org": 1000.0, "new_balance_org": 0.0, "old_balance_dest": 0.0,
         "new_balance_dest": 0.0, "timestamp": "2026-10-05T12:00:00+00:00", "is_fraud": True}
    e.update(over)
    return e


def proc():
    return Processor(DuckDBStorage(":memory:"), Scorer.from_file(RULES_PATH, load_model=False))


def test_valid_message_scored():
    p = proc()
    assert p.handle(json.dumps(event()).encode()) == "scored"
    assert p.storage.stats()["by_decision"] == {"block": 1}
    assert p.rule_hits["account_drained"] == 1


def test_duplicate_is_scored_once():
    p = proc()
    raw = json.dumps(event()).encode()
    assert p.handle(raw) == "scored"
    assert p.handle(raw) == "duplicate"
    assert sum(p.storage.stats()["by_decision"].values()) == 1


def test_duplicates_do_not_inflate_velocity():
    p = proc()
    raw = json.dumps(event(amount=10.0, old_balance_org=5000.0, new_balance_org=4990.0, is_fraud=False)).encode()
    for _ in range(10):                      # same message redelivered 10 times
        p.handle(raw)
    assert p.counters["scored"] == 1 and p.counters["duplicates"] == 9
    assert p.rule_hits["velocity_burst"] == 0


def test_malformed_messages_rejected_not_raised():
    p = proc()
    bad = [b"not json", json.dumps({"txn_id": "x"}).encode(),
           json.dumps(event("t2", amount=-5)).encode(),
           json.dumps(event("t3", type="UNKNOWN")).encode()]
    assert [p.handle(b) for b in bad] == ["rejected"] * 4
    assert p.counters["rejected"] == 4


def test_stats_precision_recall_fpr():
    p = proc()
    p.handle(json.dumps(event("a")).encode())                                              # fraud, blocked  -> tp
    p.handle(json.dumps(event("b", amount=10, is_fraud=False)).encode())                   # legit, approved -> tn
    p.handle(json.dumps(event("c", amount=10, account_id="C3", is_fraud=True)).encode())   # fraud, approved -> fn
    s = p.storage.stats()
    assert s["confusion"] == {"tp": 1, "fp": 0, "fn": 1, "tn": 1}
    assert s["precision"] == 1.0 and s["recall"] == 0.5 and s["false_positive_rate"] == 0.0
