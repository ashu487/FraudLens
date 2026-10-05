from fastapi import FastAPI
from app.config import settings
from app.models import Transaction, ScoreResult
from app.scoring import score
from app.adapters.storage import get_storage

app = FastAPI(title="Fraud Scoring Service")
storage = get_storage(settings.storage_backend, path=settings.duckdb_path)


@app.get("/health")
def health():
    return {"status": "ok", "env": settings.app_env}


@app.post("/score", response_model=ScoreResult)
def score_txn(txn: Transaction):
    if storage.seen(txn.txn_id):          # idempotency: duplicates are no-ops
        return ScoreResult(txn_id=txn.txn_id, risk_score=0.0, decision="duplicate")
    result = score(txn)
    storage.save_result(result)
    return result
