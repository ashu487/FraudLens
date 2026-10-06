import numpy as np
from app.features import FEATURE_SETS, compute_features


def cols(**over):
    c = {"is_transfer": [True], "amount": [100.0], "hour": [30], "old_org": [1000.0],
         "new_org": [900.0], "old_dest": [0.0], "new_dest": [100.0]}
    c.update({k: [v] for k, v in over.items()})
    return c


def test_shapes_and_dtype():
    for name, feats in FEATURE_SETS.items():
        X = compute_features(cols(), feats)
        assert X.shape == (1, len(feats)) and X.dtype == np.float32


def test_hour_wraps_and_ratio_values():
    X = compute_features(cols(), FEATURE_SETS["full"])
    f = dict(zip(FEATURE_SETS["full"], X[0]))
    assert f["hour_of_day"] == 6                      # 30 % 24
    assert abs(f["amount_to_balance"] - 0.1) < 1e-6
    assert f["org_balance_error"] == 0                # 1000 - 100 - 900
    assert f["dest_balance_error"] == 0               # 0 + 100 - 100


def test_zero_origin_balance_does_not_divide_by_zero():
    X = compute_features(cols(old_org=0.0, new_org=0.0), FEATURE_SETS["full"])
    assert np.isfinite(X).all()
    assert dict(zip(FEATURE_SETS["full"], X[0]))["amount_to_balance"] == 0


def test_vectorised_matches_single_row():
    c = {"is_transfer": [True, False], "amount": [100.0, 5000.0], "hour": [3, 15], "old_org": [1000.0, 5000.0],
         "new_org": [900.0, 0.0], "old_dest": [0.0, 20.0], "new_dest": [100.0, 5020.0]}
    batch = compute_features(c, FEATURE_SETS["full"])
    for i in range(2):
        one = compute_features({k: [v[i]] for k, v in c.items()}, FEATURE_SETS["full"])
        assert np.array_equal(batch[i], one[0])
