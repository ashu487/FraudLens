import pathlib
import sys
from datetime import datetime, timedelta, timezone

import pytest

ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "services" / "scoring"))
RULES_PATH = ROOT / "services" / "scoring" / "config" / "rules.yaml"

from app.models import Transaction  # noqa: E402
from app.rules import RuleEngine  # noqa: E402
from app.scoring import Scorer  # noqa: E402

T0 = datetime(2026, 10, 5, 12, 0, 0, tzinfo=timezone.utc)


def make_txn(txn_id="t1", type="TRANSFER", amount=500.0, old_org=10_000.0, account="C1", at=0, **kw):
    return Transaction(txn_id=txn_id, type=type, amount=amount, account_id=account, dest_id="C2",
                       old_balance_org=old_org, new_balance_org=max(old_org - amount, 0),
                       old_balance_dest=0, new_balance_dest=0,
                       timestamp=T0 + timedelta(seconds=at), **kw)


@pytest.fixture
def scorer():
    # rules only: results must not depend on whatever models happen to be in ./models
    return Scorer(RuleEngine.from_file(RULES_PATH))
