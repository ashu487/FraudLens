"""Offline rule evaluation on the full PaySim file (no Docker needed).

    pip install pydantic pyyaml
    python scripts/evaluate_rules.py data/PS_20174392719_1491204439457_log.csv
    python scripts/evaluate_rules.py data/PS_...csv --limit 500000

Prints overall precision/recall/FPR, per-rule precision and coverage, and what the
rules miss. Edit services/scoring/config/rules.yaml and re-run to see the effect.

Velocity rules are skipped by default: PaySim timestamps are whole hours and accounts
rarely repeat, so a 60-second window is meaningless there. Use --with-velocity to include them.
"""
import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "scoring"))
from app.rules import RuleEngine  # noqa: E402
from app.velocity import VelocityTracker, EMPTY  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--rules", default=str(ROOT / "services/scoring/config/rules.yaml"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--with-velocity", action="store_true")
    args = ap.parse_args()

    engine = RuleEngine.from_file(args.rules)
    skipped = []
    if not args.with_velocity:
        skipped = [r.name for r in engine.rules if r.needs_velocity]
        engine.rules = [r for r in engine.rules if not r.needs_velocity]
    windows = engine.windows
    tracker = VelocityTracker(engine.retention_seconds) if windows else None

    tp = fp = fn = tn = 0
    hits, fraud_hits = Counter(), Counter()
    missed_by_type, missed_zero_balance = Counter(), 0
    n = total_fraud = 0
    t0 = time.time()

    with open(args.csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if args.limit and n >= args.limit:
                break
            n += 1
            fraud = row["isFraud"] == "1"
            step = int(row["step"])
            txn = SimpleNamespace(
                type=row["type"], amount=float(row["amount"]), account_id=row["nameOrig"],
                old_balance_org=float(row["oldbalanceOrg"]))
            velocity = tracker.observe(txn.account_id, step * 3600.0, txn.amount, windows) if tracker else EMPTY
            fired = engine.evaluate(txn, velocity)
            _, decision = engine.decide(fired)
            flagged = decision != "approve"
            for r in fired:
                hits[r.name] += 1
                fraud_hits[r.name] += fraud
            total_fraud += fraud
            if fraud and flagged: tp += 1
            elif fraud: 
                fn += 1
                missed_by_type[txn.type] += 1
                missed_zero_balance += txn.old_balance_org == 0
            elif flagged: fp += 1
            else: tn += 1

    def pct(a, b): return f"{100 * a / b:.2f}%" if b else "n/a"
    print(f"\nRows: {n:,}   fraud: {total_fraud:,} ({pct(total_fraud, n)})   [{time.time() - t0:.0f}s]")
    if skipped:
        print(f"Skipped velocity rules: {', '.join(skipped)}  (use --with-velocity to include)")
    print("\nOVERALL (flagged = review or block)")
    print(f"  tp={tp:,} fp={fp:,} fn={fn:,} tn={tn:,}")
    print(f"  precision={pct(tp, tp + fp)}  recall={pct(tp, tp + fn)}  false-positive-rate={pct(fp, fp + tn)}")
    print("\nPER RULE")
    print(f"  {'rule':<22}{'hits':>10}{'fraud':>10}{'precision':>12}{'share of all fraud':>22}")
    for r in engine.rules:
        print(f"  {r.name:<22}{hits[r.name]:>10,}{fraud_hits[r.name]:>10,}"
              f"{pct(fraud_hits[r.name], hits[r.name]):>12}{pct(fraud_hits[r.name], total_fraud):>22}")
    print(f"\nMISSED FRAUD: {fn:,} by type {dict(missed_by_type)}; {missed_zero_balance:,} had origin balance 0")


if __name__ == "__main__":
    main()
