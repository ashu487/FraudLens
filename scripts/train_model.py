"""Train and evaluate fraud models on PaySim, honestly.

    pip install -r requirements-train.txt
    python scripts/train_model.py data/PS_20174392719_1491204439457_log.csv
    python scripts/train_model.py data/PS_...csv --max-rows 1000000     # quick dry run

What it does
  * keeps only TRANSFER and CASH_OUT (the only types that contain fraud)
  * splits by TIME (steps): train 60% / validation 20% / test 20% -- never randomly
  * trains XGBoost (supervised) and Isolation Forest (unsupervised) on three feature sets:
        base       no balances at all
        no_origin  + destination balances
        full       + origin balances (PaySim leaks its fraud pattern through these)
  * picks alert thresholds on VALIDATION at fixed false-positive rates, reports on TEST
  * compares the rules alone vs rules + model, and counts the fraud the rules miss
    that the model recovers
  * writes models/<set>/{model.json,meta.json}, docs/model_results.md, docs/pr_curves.png
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "scoring"))
from app.features import FEATURE_SETS, compute_features  # noqa: E402
from app.rules import RuleEngine  # noqa: E402
from app.velocity import EMPTY  # noqa: E402

RISKY = ["TRANSFER", "CASH_OUT"]


def load(path: str, max_rows: int) -> pd.DataFrame:
    cols = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest", "isFraud"]
    df = pd.read_csv(path, usecols=cols, nrows=max_rows or None)
    df = df[df["type"].isin(RISKY)].reset_index(drop=True)
    return df


def feature_inputs(df: pd.DataFrame) -> dict:
    return {"is_transfer": (df["type"] == "TRANSFER").to_numpy(), "amount": df["amount"].to_numpy(),
            "hour": (df["step"] % 24).to_numpy(), "old_org": df["oldbalanceOrg"].to_numpy(),
            "new_org": df["newbalanceOrig"].to_numpy(), "old_dest": df["oldbalanceDest"].to_numpy(),
            "new_dest": df["newbalanceDest"].to_numpy()}


def confusion(flag: np.ndarray, y: np.ndarray) -> dict:
    tp, fp = int((flag & y).sum()), int((flag & ~y).sum())
    fn, tn = int((~flag & y).sum()), int((~flag & ~y).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "fpr": fp / (fp + tn) if fp + tn else None}


def recall_at_fpr(y, scores, fpr) -> float:
    f, t, _ = roc_curve(y, scores)
    return float(t[f <= fpr].max())


def threshold_at_fpr(val_neg_scores: np.ndarray, fpr: float) -> float:
    return float(np.quantile(val_neg_scores, 1 - fpr))


def fmt(x, pct=True, nd=2):
    if x is None:
        return "n/a"
    return f"{100 * x:.{nd}f}%" if pct else f"{x:.4f}"


def rules_flags(df: pd.DataFrame, rules_path: str) -> np.ndarray:
    """Run the real rules engine (rules.yaml) over the slice. Velocity rules are skipped:
    PaySim timestamps are whole hours, so a 60-second window means nothing here."""
    engine = RuleEngine.from_file(rules_path)
    engine.rules = [r for r in engine.rules if not r.needs_velocity]
    out = np.zeros(len(df), dtype=bool)
    for i, (t, a, o) in enumerate(zip(df["type"].to_numpy(), df["amount"].to_numpy(),
                                      df["oldbalanceOrg"].to_numpy())):
        txn = SimpleNamespace(type=t, amount=a, old_balance_org=o)
        _, decision = engine.decide(engine.evaluate(txn, EMPTY))
        out[i] = decision != "approve"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--rules", default=str(ROOT / "services/scoring/config/rules.yaml"))
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--docs", default=str(ROOT / "docs"))
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--review-fpr", type=float, default=0.001, help="validation FPR for the review threshold")
    ap.add_argument("--block-fpr", type=float, default=0.0001, help="validation FPR for the block threshold")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t_start = time.time()

    df = load(args.csv_path, args.max_rows)
    y_all = df["isFraud"].to_numpy().astype(bool)
    c1, c2 = np.quantile(df["step"].to_numpy(), [0.6, 0.8])
    tr = (df["step"] <= c1).to_numpy()
    va = ((df["step"] > c1) & (df["step"] <= c2)).to_numpy()
    te = (df["step"] > c2).to_numpy()
    print(f"\nTRANSFER+CASH_OUT rows: {len(df):,}   fraud: {y_all.sum():,} ({y_all.mean():.3%})")
    for name, m in [("train", tr), ("val", va), ("test", te)]:
        print(f"  {name:<5} {m.sum():>9,} rows  {y_all[m].sum():>6,} fraud   steps "
              f"{df['step'][m].min()}-{df['step'][m].max()}")
    if min(y_all[tr].sum(), y_all[va].sum(), y_all[te].sum()) < 20:
        sys.exit("Too few fraud rows in a split; use more data (drop --max-rows).")

    inputs = feature_inputs(df)
    y_tr, y_va, y_te = y_all[tr], y_all[va], y_all[te]

    # ---- the rules baseline on the test slice ----
    print("\nRunning the rules engine on the test slice...")
    test_df = df[te].reset_index(drop=True)
    rf = rules_flags(test_df, args.rules)
    rules_cm = confusion(rf, y_te)
    missed_by_rules = y_te & ~rf
    print(f"  rules alone: precision {fmt(rules_cm['precision'])}  recall {fmt(rules_cm['recall'])}  "
          f"FPR {fmt(rules_cm['fpr'], nd=3)}  missed fraud: {int(missed_by_rules.sum())}")

    results, curves = {}, {}
    out_dir = Path(args.out)
    for fs_name, names in FEATURE_SETS.items():
        print(f"\n=== feature set: {fs_name} ({len(names)} features) ===")
        X = compute_features(inputs, names)
        X_tr, X_va, X_te = X[tr], X[va], X[te]

        # --- XGBoost ---
        pos, neg = int(y_tr.sum()), int((~y_tr).sum())
        clf = xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=float(np.sqrt(neg / pos)),   # softened class weighting
            tree_method="hist", eval_metric="aucpr", early_stopping_rounds=30,
            random_state=args.seed, n_jobs=-1)
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        p_va, p_te = clf.predict_proba(X_va)[:, 1], clf.predict_proba(X_te)[:, 1]
        review_thr = threshold_at_fpr(p_va[~y_va], args.review_fpr)
        block_thr = threshold_at_fpr(p_va[~y_va], args.block_fpr)

        xgb_res = {
            "pr_auc": float(average_precision_score(y_te, p_te)),
            "recall_at_fpr_0.1": recall_at_fpr(y_te, p_te, 0.001),
            "recall_at_fpr_1": recall_at_fpr(y_te, p_te, 0.01),
            "review": confusion(p_te >= review_thr, y_te),
            "block": confusion(p_te >= block_thr, y_te),
            "best_iteration": int(clf.best_iteration),
        }
        mf = p_te >= review_thr
        hybrid = confusion(rf | mf, y_te)
        xgb_res["hybrid"] = hybrid
        xgb_res["recovered"] = int((missed_by_rules & mf).sum())
        xgb_res["new_false_positives"] = int((mf & ~rf & ~y_te).sum())
        gain = clf.get_booster().get_score(importance_type="gain")
        total = sum(gain.values()) or 1.0
        xgb_res["top_features"] = sorted(
            ((names[int(k[1:])], v / total) for k, v in gain.items()), key=lambda kv: -kv[1])[:6]

        # --- Isolation Forest (unsupervised: never sees labels) ---
        iso = IsolationForest(n_estimators=200, max_samples=256, random_state=args.seed, n_jobs=-1).fit(X_tr)
        s_te = -iso.score_samples(X_te)
        iso_res = {"pr_auc": float(average_precision_score(y_te, s_te)),
                   "recall_at_fpr_0.1": recall_at_fpr(y_te, s_te, 0.001),
                   "recall_at_fpr_1": recall_at_fpr(y_te, s_te, 0.01)}

        print(f"  XGBoost  PR-AUC {xgb_res['pr_auc']:.4f}   recall@FPR0.1% {fmt(xgb_res['recall_at_fpr_0.1'])}"
              f"   recall@FPR1% {fmt(xgb_res['recall_at_fpr_1'])}")
        print(f"  IsoForest PR-AUC {iso_res['pr_auc']:.4f}  recall@FPR0.1% {fmt(iso_res['recall_at_fpr_0.1'])}"
              f"   recall@FPR1% {fmt(iso_res['recall_at_fpr_1'])}")
        print(f"  rules + model(review): precision {fmt(hybrid['precision'])} recall {fmt(hybrid['recall'])} "
              f"FPR {fmt(hybrid['fpr'], nd=3)} | recovered {xgb_res['recovered']}/{int(missed_by_rules.sum())} "
              f"missed fraud, +{xgb_res['new_false_positives']} new false positives")

        # --- save the model ---
        folder = out_dir / fs_name
        folder.mkdir(parents=True, exist_ok=True)
        clf.save_model(str(folder / "model.json"))
        meta = {"feature_set": fs_name, "features": names, "review_threshold": review_thr,
                "block_threshold": block_thr, "review_fpr_target": args.review_fpr,
                "block_fpr_target": args.block_fpr, "xgboost_version": xgb.__version__,
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "split_steps": {"train_max": int(c1), "val_max": int(c2)},
                "test_pr_auc": xgb_res["pr_auc"], "test_review": xgb_res["review"]}
        (folder / "meta.json").write_text(json.dumps(meta, indent=2))
        results[fs_name] = {"xgb": xgb_res, "iso": iso_res}
        curves[fs_name] = (precision_recall_curve(y_te, p_te), precision_recall_curve(y_te, s_te))

    write_report(args, results, rules_cm, int(missed_by_rules.sum()), len(df), int(y_all.sum()),
                 {"train": (int(tr.sum()), int(y_tr.sum())), "val": (int(va.sum()), int(y_va.sum())),
                  "test": (int(te.sum()), int(y_te.sum()))}, (int(c1), int(c2)))
    plot(args, curves, rules_cm)
    print(f"\nDone in {time.time() - t_start:.0f}s. Models in {out_dir}, report in {args.docs}/model_results.md")


def write_report(args, results, rules_cm, n_missed, n_rows, n_fraud, splits, cuts):
    L = []
    L.append("# Model results\n")
    L.append(f"_Generated by `scripts/train_model.py` on {datetime.now().strftime('%Y-%m-%d')}. "
             "Scope: TRANSFER and CASH_OUT rows of PaySim (the only types with fraud)._\n")
    L.append("## Setup\n")
    L.append(f"- Rows: {n_rows:,}, fraud: {n_fraud:,} ({n_fraud / n_rows:.3%})")
    L.append(f"- **Time-based split** on `step`: train <= {cuts[0]}, validation <= {cuts[1]}, test after that")
    for k, (n, f) in splits.items():
        L.append(f"  - {k}: {n:,} rows, {f:,} fraud")
    L.append("- Alert thresholds are chosen on **validation** at a fixed false-positive rate "
             f"(review: {args.review_fpr:.2%}, block: {args.block_fpr:.2%}); every number below is on the **test** slice.")
    L.append("- FPR here is measured against legit TRANSFER/CASH_OUT rows only, so it is stricter than an all-traffic FPR.\n")
    L.append("## Rules alone (test slice)\n")
    L.append(f"Precision {fmt(rules_cm['precision'])}, recall {fmt(rules_cm['recall'])}, "
             f"FPR {fmt(rules_cm['fpr'], nd=3)}; **{n_missed} fraud cases missed**.\n")
    L.append("## Model comparison (test slice)\n")
    L.append("| Features | Model | PR-AUC | Recall @ FPR 0.1% | Recall @ FPR 1% |")
    L.append("|---|---|---|---|---|")
    for fs, r in results.items():
        L.append(f"| {fs} | XGBoost | {r['xgb']['pr_auc']:.4f} | {fmt(r['xgb']['recall_at_fpr_0.1'])} | {fmt(r['xgb']['recall_at_fpr_1'])} |")
        L.append(f"| {fs} | Isolation Forest | {r['iso']['pr_auc']:.4f} | {fmt(r['iso']['recall_at_fpr_0.1'])} | {fmt(r['iso']['recall_at_fpr_1'])} |")
    L.append("\nFeature sets: `base` = type, amount, hour (no balances); `no_origin` = base + destination balances; "
             "`full` = + origin balances.\n")
    L.append("## Rules vs rules + model (model at the review threshold)\n")
    L.append("| Features | Precision | Recall | FPR | Missed fraud recovered | New false positives |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| rules only | {fmt(rules_cm['precision'])} | {fmt(rules_cm['recall'])} | {fmt(rules_cm['fpr'], nd=3)} | - | - |")
    for fs, r in results.items():
        h = r["xgb"]["hybrid"]
        L.append(f"| rules + {fs} | {fmt(h['precision'])} | {fmt(h['recall'])} | {fmt(h['fpr'], nd=3)} | "
                 f"{r['xgb']['recovered']} of {n_missed} | {r['xgb']['new_false_positives']} |")
    L.append("\n## Top features (XGBoost, share of total gain)\n")
    for fs, r in results.items():
        L.append(f"- **{fs}**: " + ", ".join(f"{n} ({s:.0%})" for n, s in r["xgb"]["top_features"]))
    L.append("\n## Caveats\n")
    L.append("- PaySim is synthetic and leaks its fraud pattern through account balances, so `full` scores are optimistic. "
             "`base` is the honest estimate of what amount/type/time alone can do.")
    L.append("- Time-of-day (`step % 24`) is a real signal in PaySim mainly because its generator makes legit volume "
             "cyclical while fraud is not. Treat it with suspicion.")
    L.append("- Velocity rules are excluded from the offline comparison (PaySim timestamps are whole hours).")
    Path(args.docs).mkdir(parents=True, exist_ok=True)
    (Path(args.docs) / "model_results.md").write_text("\n".join(L) + "\n")


def plot(args, curves, rules_cm):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"base": "tab:blue", "no_origin": "tab:orange", "full": "tab:green"}
    for fs, ((px, rx, _), (pi, ri, _)) in curves.items():
        ax.plot(rx, px, color=colors[fs], label=f"XGBoost / {fs}")
        ax.plot(ri, pi, color=colors[fs], linestyle=":", label=f"IsoForest / {fs}")
    if rules_cm["recall"] is not None and rules_cm["precision"] is not None:
        ax.scatter([rules_cm["recall"]], [rules_cm["precision"]], color="red", marker="*", s=140, zorder=5, label="rules")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall on the time-held-out test slice"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    Path(args.docs).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(args.docs) / "pr_curves.png", dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()
