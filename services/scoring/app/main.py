import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from app.config import settings
from app.models import Transaction
from app.pipeline import Processor
from app.adapters.storage import get_storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage = get_storage(settings.storage_backend, path=settings.duckdb_path)
    processor = Processor(storage)
    app.state.storage, app.state.processor = storage, processor
    consumer = None
    if settings.queue_backend == "pubsub":
        from app.consumer import PubSubConsumer
        consumer = PubSubConsumer(settings.gcp_project, settings.pubsub_topic,
                                  settings.pubsub_subscription, processor.handle)
        consumer.start()
    yield
    if consumer:
        consumer.stop()


app = FastAPI(title="Fraud Scoring Service", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "env": settings.app_env, "counters": dict(request.app.state.processor.counters)}


@app.post("/score")
def score_txn(txn: Transaction, request: Request):
    status, result = request.app.state.processor.process(txn)
    if result is None:
        return {"status": status, "txn_id": txn.txn_id}
    return {"status": status, **result.model_dump()}


@app.get("/stats")
def stats(request: Request):
    return {"counters": dict(request.app.state.processor.counters), **request.app.state.storage.stats()}


@app.get("/recent")
def recent(request: Request, n: int = 20):
    return request.app.state.storage.recent(min(n, 200))
