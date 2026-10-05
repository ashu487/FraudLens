"""Storage interface. DuckDB locally, BigQuery on GCP (added in Week 7)."""
from abc import ABC, abstractmethod
import duckdb
from app.models import ScoreResult


class Storage(ABC):
    @abstractmethod
    def save_result(self, result: ScoreResult) -> None: ...

    @abstractmethod
    def seen(self, txn_id: str) -> bool:
        """Idempotency check."""


class DuckDBStorage(Storage):
    def __init__(self, path: str):
        self.con = duckdb.connect(path)
        self.con.execute(
            """CREATE TABLE IF NOT EXISTS results (
                txn_id VARCHAR PRIMARY KEY,
                risk_score DOUBLE,
                decision VARCHAR,
                reasons VARCHAR
            )"""
        )

    def save_result(self, result: ScoreResult) -> None:
        self.con.execute(
            "INSERT OR IGNORE INTO results VALUES (?, ?, ?, ?)",
            [result.txn_id, result.risk_score, result.decision, ";".join(result.reasons)],
        )

    def seen(self, txn_id: str) -> bool:
        return self.con.execute(
            "SELECT 1 FROM results WHERE txn_id = ?", [txn_id]
        ).fetchone() is not None


def get_storage(backend: str, **kw) -> Storage:
    if backend == "duckdb":
        return DuckDBStorage(kw["path"])
    raise ValueError(f"Unknown storage backend: {backend}")
