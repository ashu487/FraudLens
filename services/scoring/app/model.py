"""Serving wrapper for the trained XGBoost fraud model."""
import json
import logging
import threading
from pathlib import Path

import numpy as np
import xgboost as xgb

from app.features import compute_features
from app.rules import Rule

log = logging.getLogger("model")

IN_SCOPE = {"TRANSFER", "CASH_OUT"}      # the only types that contain fraud in PaySim
_NOOP = lambda *args: True               # noqa: E731  (model hits aren't predicate rules)


class FraudModel:
    def __init__(self, booster, meta: dict, review_threshold: float, block_threshold: float,
                 review_weight: float, block_weight: float):
        self.booster = booster
        self.meta = meta
        self.features = meta["features"]
        self.review_threshold = review_threshold
        self.block_threshold = block_threshold
        self.review_weight = review_weight
        self.block_weight = block_weight
        self._lock = threading.Lock()    # consumer callbacks run on several threads

    @classmethod
    def from_config(cls, cfg: dict | None):
        if not cfg or not cfg.get("enabled", False):
            return None
        folder = Path(cfg["dir"])
        if not (folder / "model.json").exists() or not (folder / "meta.json").exists():
            log.warning("model enabled but %s has no model.json/meta.json; running rules-only. "
                        "Train one with scripts/train_model.py", folder)
            return None
        meta = json.loads((folder / "meta.json").read_text())
        booster = xgb.Booster()
        booster.load_model(str(folder / "model.json"))
        model = cls(booster, meta,
                    review_threshold=cfg.get("review_threshold", meta["review_threshold"]),
                    block_threshold=cfg.get("block_threshold", meta["block_threshold"]),
                    review_weight=cfg.get("review_weight", 0.5),
                    block_weight=cfg.get("block_weight", 0.85))
        log.info("loaded model %s (features=%s, review>=%.4f, block>=%.4f)", meta.get("feature_set"),
                 len(model.features), model.review_threshold, model.block_threshold)
        return model

    def predict_proba(self, txn) -> float | None:
        """Fraud probability, or None when the transaction type is outside the model's scope."""
        if txn.type not in IN_SCOPE:
            return None
        hour = txn.step if txn.step is not None else txn.timestamp.hour
        cols = {
            "is_transfer": np.array([txn.type == "TRANSFER"]), "amount": np.array([txn.amount]),
            "hour": np.array([hour]), "old_org": np.array([txn.old_balance_org]),
            "new_org": np.array([txn.new_balance_org]), "old_dest": np.array([txn.old_balance_dest]),
            "new_dest": np.array([txn.new_balance_dest]),
        }
        X = compute_features(cols, self.features)
        with self._lock:
            return float(self.booster.inplace_predict(X)[0])

    def hits(self, prob: float | None) -> list[Rule]:
        if prob is None:
            return []
        if prob >= self.block_threshold:
            return [Rule("model_block", self.block_weight, {}, _NOOP)]
        if prob >= self.review_threshold:
            return [Rule("model_review", self.review_weight, {}, _NOOP)]
        return []

    def describe(self) -> dict:
        return {"feature_set": self.meta.get("feature_set"), "features": self.features,
                "review_threshold": self.review_threshold, "block_threshold": self.block_threshold,
                "review_weight": self.review_weight, "block_weight": self.block_weight,
                "trained_at": self.meta.get("trained_at")}
