"""Storage interface. DuckDB locally, BigQuery on GCP (Week 7)."""
import threading
from abc import ABC, abstractmethod
import duckdb
from app.models import Transaction, ScoreResult


class Storage(ABC):
    @abstractmethod
    def seen(self, txn_id: str) -> bool: ...

    @abstractmethod
    def save_if_new(self, txn: Transaction, result: ScoreResult) -> bool:
        """Atomically store the result. Returns False if txn_id was already stored (idempotency)."""

    @abstractmethod
    def stats(self) -> dict: ...

    @abstractmethod
    def recent(self, n: int = 20) -> list[dict]: ...


class DuckDBStorage(Storage):
    def __init__(self, path: str):
        self.con = duckdb.connect(path)
        self.lock = threading.Lock()   # consumer callbacks run on multiple threads
        self.con.execute(
            """CREATE TABLE IF NOT EXISTS scored_transactions (
                txn_id VARCHAR PRIMARY KEY,
                type VARCHAR,
                amount DOUBLE,
                account_id VARCHAR,
                risk_score DOUBLE,
                decision VARCHAR,
                reasons VARCHAR,
                is_fraud BOOLEAN,
                scored_at TIMESTAMP DEFAULT current_timestamp
            )"""
        )

    def seen(self, txn_id: str) -> bool:
        with self.lock:
            return self.con.execute(
                "SELECT 1 FROM scored_transactions WHERE txn_id = ?", [txn_id]).fetchone() is not None

    def save_if_new(self, txn: Transaction, result: ScoreResult) -> bool:
        with self.lock:
            if self.con.execute(
                "SELECT 1 FROM scored_transactions WHERE txn_id = ?", [txn.txn_id]
            ).fetchone():
                return False
            self.con.execute(
                """INSERT INTO scored_transactions
                   (txn_id, type, amount, account_id, risk_score, decision, reasons, is_fraud)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [txn.txn_id, txn.type, txn.amount, txn.account_id, result.risk_score,
                 result.decision, ";".join(result.reasons), txn.is_fraud],
            )
            return True

    def stats(self) -> dict:
        with self.lock:
            by_decision = dict(self.con.execute(
                "SELECT decision, count(*) FROM scored_transactions GROUP BY decision").fetchall())
            tp, fp, fn, tn = self.con.execute(
                """SELECT
                    coalesce(sum(CASE WHEN decision <> 'approve' AND is_fraud THEN 1 ELSE 0 END), 0),
                    coalesce(sum(CASE WHEN decision <> 'approve' AND NOT is_fraud THEN 1 ELSE 0 END), 0),
                    coalesce(sum(CASE WHEN decision = 'approve' AND is_fraud THEN 1 ELSE 0 END), 0),
                    coalesce(sum(CASE WHEN decision = 'approve' AND NOT is_fraud THEN 1 ELSE 0 END), 0)
                   FROM scored_transactions WHERE is_fraud IS NOT NULL"""
            ).fetchone()
        tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
        return {
            "by_decision": by_decision,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else None,
        }

    def recent(self, n: int = 20) -> list[dict]:
        with self.lock:
            cur = self.con.execute(
                "SELECT txn_id, type, amount, account_id, risk_score, decision, reasons, is_fraud "
                "FROM scored_transactions ORDER BY scored_at DESC LIMIT ?", [n])
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_storage(backend: str, **kw) -> Storage:
    if backend == "duckdb":
        return DuckDBStorage(kw["path"])
    raise ValueError(f"Unknown storage backend: {backend}")
