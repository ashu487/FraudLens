import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from app.config import settings
from app.models import Transaction
from app.pipeline import Processor
from app.scoring import Scorer
from app.adapters.storage import get_storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage = get_storage(settings.storage_backend, path=settings.duckdb_path)
    scorer = Scorer.from_file(settings.rules_config)
    log.info("loaded rules: %s", [r.name for r in scorer.engine.rules])
    processor = Processor(storage, scorer)
    app.state.storage, app.state.processor, app.state.scorer = storage, processor, scorer
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


@app.get("/rules")
def rules(request: Request):
    return request.app.state.scorer.engine.describe()


@app.post("/score")
def score_txn(txn: Transaction, request: Request):
    status, result = request.app.state.processor.process(txn)
    if result is None:
        return {"status": status, "txn_id": txn.txn_id}
    return {"status": status, **result.model_dump()}


@app.get("/stats")
def stats(request: Request):
    p = request.app.state.processor
    return {"counters": dict(p.counters), "rule_hits": dict(p.rule_hits), **request.app.state.storage.stats()}


@app.get("/recent")
def recent(request: Request, n: int = 20):
    return request.app.state.storage.recent(min(n, 200))
