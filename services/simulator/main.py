"""Replays PaySim rows into Pub/Sub, with fraud boosting, duplicates and malformed events.
Falls back to synthetic rows if the PaySim CSV isn't mounted."""
import csv, json, os, random, time, uuid
from datetime import datetime, timezone
from google.api_core.exceptions import AlreadyExists
from google.cloud import pubsub_v1

project = os.getenv("GCP_PROJECT", "fraud-local")
topic_name = os.getenv("PUBSUB_TOPIC", "transactions")
rate = float(os.getenv("SIM_RATE_PER_SEC", "5"))
fraud_ratio = float(os.getenv("SIM_FRAUD_RATIO", "0.05"))
legit_sample = float(os.getenv("SIM_LEGIT_SAMPLE", "0.01"))
dup_rate = float(os.getenv("SIM_DUPLICATE_RATE", "0.02"))
bad_rate = float(os.getenv("SIM_BAD_RATE", "0.01"))
csv_path = os.getenv("PAYSIM_CSV", "/data/PS_20174392719_1491204439457_log.csv")


def load_pools():
    fraud, legit = [], []
    print(f"Loading {csv_path} (keeping all fraud, {legit_sample:.1%} of legit)...", flush=True)
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["isFraud"] == "1":
                fraud.append(row)
            elif random.random() < legit_sample:
                legit.append(row)
    print(f"Loaded {len(fraud)} fraud and {len(legit)} legit rows", flush=True)
    return fraud, legit


def synthetic_row(is_fraud: bool) -> dict:
    bal = round(random.lognormvariate(10, 1.2), 2)
    amt = bal if is_fraud else round(bal * random.uniform(0.01, 0.6), 2)
    return {"type": random.choice(["TRANSFER", "CASH_OUT"] if is_fraud else ["PAYMENT", "CASH_IN", "TRANSFER", "CASH_OUT"]),
            "amount": amt, "nameOrig": f"C{random.randint(1, 500)}", "nameDest": f"M{random.randint(1, 100)}",
            "oldbalanceOrg": bal, "newbalanceOrig": bal - amt, "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
            "step": 1, "isFraud": "1" if is_fraud else "0"}


def to_event(row: dict) -> dict:
    return {
        "txn_id": str(uuid.uuid4()),
        "type": row["type"],
        "amount": float(row["amount"]),
        "account_id": row["nameOrig"],
        "dest_id": row["nameDest"],
        "old_balance_org": float(row["oldbalanceOrg"]),
        "new_balance_org": float(row["newbalanceOrig"]),
        "old_balance_dest": float(row["oldbalanceDest"]),
        "new_balance_dest": float(row["newbalanceDest"]),
        "step": int(row["step"]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "is_fraud": row["isFraud"] == "1",
    }


def corrupt(event: dict) -> dict:
    bad = dict(event)
    choice = random.choice(["missing_amount", "negative_amount", "bad_type"])
    if choice == "missing_amount":
        bad.pop("amount")
    elif choice == "negative_amount":
        bad["amount"] = -abs(bad["amount"]) - 1
    else:
        bad["type"] = "UNKNOWN"
    return bad


use_csv = os.path.exists(csv_path)
fraud_pool, legit_pool = load_pools() if use_csv else ([], [])
if not use_csv:
    print(f"{csv_path} not found, using synthetic data", flush=True)

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project, topic_name)
for _ in range(60):  # wait for emulator
    try:
        publisher.create_topic(request={"name": topic_path})
        break
    except AlreadyExists:
        break
    except Exception:
        time.sleep(1)

print(f"Publishing to {topic_path} at {rate}/sec (fraud {fraud_ratio:.0%}, dup {dup_rate:.0%}, bad {bad_rate:.0%})", flush=True)
sent = 0
while True:
    want_fraud = random.random() < fraud_ratio
    if use_csv:
        row = random.choice(fraud_pool if want_fraud else legit_pool)
    else:
        row = synthetic_row(want_fraud)
    event = to_event(row)
    if random.random() < bad_rate:
        event = corrupt(event)
    data = json.dumps(event).encode()
    publisher.publish(topic_path, data)
    if random.random() < dup_rate:      # at-least-once delivery: same event again
        publisher.publish(topic_path, data)
    sent += 1
    if sent % 100 == 0:
        print(f"sent {sent} events", flush=True)
    time.sleep(1 / rate)
