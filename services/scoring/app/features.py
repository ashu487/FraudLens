"""Feature engineering shared by training AND serving.

One implementation, used by scripts/train_model.py (on whole columns) and by the
live scorer (on one-element arrays), so the model can never see different
features in production than it saw in training (train/serve skew).

Inputs (dict of equal-length arrays):
  is_transfer, amount, hour, old_org, new_org, old_dest, new_dest
"""
import numpy as np

BASE = ["is_transfer", "amount", "log_amount", "hour_of_day"]
DEST = ["old_balance_dest", "new_balance_dest", "dest_balance_error"]
ORIGIN = ["old_balance_org", "new_balance_org", "org_balance_error", "amount_to_balance"]

# base      : no balances at all (the "honest" model)
# no_origin : adds destination balances, still no origin-account balance
# full      : everything (PaySim leaks its fraud pattern through origin balances)
FEATURE_SETS = {"base": BASE, "no_origin": BASE + DEST, "full": BASE + DEST + ORIGIN}


def compute_features(c: dict, names: list[str]) -> np.ndarray:
    amount = np.asarray(c["amount"], dtype=np.float64)
    old_org = np.asarray(c["old_org"], dtype=np.float64)
    new_org = np.asarray(c["new_org"], dtype=np.float64)
    old_dest = np.asarray(c["old_dest"], dtype=np.float64)
    new_dest = np.asarray(c["new_dest"], dtype=np.float64)
    feats = {
        "is_transfer": np.asarray(c["is_transfer"], dtype=np.float64),
        "amount": amount,
        "log_amount": np.log1p(amount),
        "hour_of_day": np.asarray(c["hour"], dtype=np.float64) % 24,
        "old_balance_dest": old_dest,
        "new_balance_dest": new_dest,
        "dest_balance_error": old_dest + amount - new_dest,
        "old_balance_org": old_org,
        "new_balance_org": new_org,
        "org_balance_error": old_org - amount - new_org,
        "amount_to_balance": np.divide(amount, old_org, out=np.zeros_like(amount), where=old_org > 0),
    }
    return np.column_stack([feats[n] for n in names]).astype(np.float32)
