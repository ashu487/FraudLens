"""Generates fake transactions and publishes them to Pub/Sub.
Week 2: replay PaySim data and inject fraud patterns."""
import json, os, random, time, uuid
from datetime import datetime, timezone
from google.cloud import pubsub_v1
from google.api_core.exceptions import AlreadyExists

project = os.getenv("GCP_PROJECT", "fraud-local")
topic_name = os.getenv("PUBSUB_TOPIC", "transactions")
rate = float(os.getenv("SIM_RATE_PER_SEC", "5"))

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project, topic_name)

for _ in range(30):  # wait for emulator
    try:
        publisher.create_topic(request={"name": topic_path})
        break
    except AlreadyExists:
        break
    except Exception:
        time.sleep(1)

print(f"Publishing to {topic_path} at {rate}/sec")
while True:
    txn = {
        "txn_id": str(uuid.uuid4()),
        "account_id": f"acc_{random.randint(1, 500)}",
        "merchant_id": f"m_{random.randint(1, 100)}",
        "amount": round(random.lognormvariate(7, 1.2), 2),
        "currency": "INR",
        "country": "IN",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    publisher.publish(topic_path, json.dumps(txn).encode())
    time.sleep(1 / rate)
