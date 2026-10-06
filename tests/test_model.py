import json

import numpy as np
import pytest
import xgboost as xgb
import yaml
from conftest import RULES_PATH, make_txn
from app.features import FEATURE_SETS, compute_features
from app.model import FraudModel
from app.scoring import Scorer


@pytest.fixture
def model_dir(tmp_path):
    """Train a tiny model that learns: fraud <=> amount is >= 90% of the origin balance."""
    rng = np.random.default_rng(0)
    n = 3000
    old = rng.uniform(1_000, 100_000, n)
    ratio = np.where(rng.random(n) < 0.3, rng.uniform(0.9, 1.0, n), rng.uniform(0.0, 0.5, n))
    amount = old * ratio
    c = {"is_transfer": rng.random(n) < 0.5, "amount": amount, "hour": rng.integers(0, 24, n),
         "old_org": old, "new_org": np.maximum(old - amount, 0), "old_dest": np.zeros(n), "new_dest": amount}
    names = FEATURE_SETS["full"]
    clf = xgb.XGBClassifier(n_estimators=40, max_depth=3, random_state=0)
    clf.fit(compute_features(c, names), ratio >= 0.9)
    clf.save_model(str(tmp_path / "model.json"))
    (tmp_path / "meta.json").write_text(json.dumps(
        {"feature_set": "full", "features": names, "review_threshold": 0.5, "block_threshold": 0.9}))
    return tmp_path


def scorer_with(model_dir, tmp_path, **model_cfg):
    cfg = yaml.safe_load(RULES_PATH.read_text())
    cfg["model"] = {"enabled": True, "dir": str(model_dir), **model_cfg}
    p = tmp_path / "rules.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return Scorer.from_file(str(p))


def test_model_flags_near_drain_that_the_rules_miss(model_dir, tmp_path):
    s = scorer_with(model_dir, tmp_path)
    r = s.score(make_txn(amount=9_700.0, old_org=10_000.0))     # 97% of balance, not an exact drain
    assert "account_drained" not in r.reasons
    assert "model_block" in r.reasons and r.decision == "block"
    assert r.model_score > 0.9


def test_model_leaves_normal_txn_alone(model_dir, tmp_path):
    r = scorer_with(model_dir, tmp_path).score(make_txn(amount=100.0, old_org=10_000.0))
    assert r.decision == "approve" and r.model_score < 0.5


def test_out_of_scope_types_skip_the_model(model_dir, tmp_path):
    r = scorer_with(model_dir, tmp_path).score(make_txn(type="PAYMENT", amount=9_700.0, old_org=10_000.0))
    assert r.model_score is None and r.decision == "approve"


def test_thresholds_can_be_overridden_in_yaml(model_dir, tmp_path):
    s = scorer_with(model_dir, tmp_path, review_threshold=0.5, block_threshold=0.9999999)
    r = s.score(make_txn(amount=9_700.0, old_org=10_000.0))
    assert "model_review" in r.reasons and r.decision == "review"


def test_missing_model_folder_falls_back_to_rules_only(tmp_path):
    assert FraudModel.from_config({"enabled": True, "dir": str(tmp_path / "nope")}) is None
    assert FraudModel.from_config({"enabled": False, "dir": "x"}) is None
