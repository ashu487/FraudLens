"""Message handling: validate -> dedupe -> score -> store."""
import logging
import threading
from collections import Counter
from pydantic import ValidationError
from app.models import Transaction, ScoreResult
from app.scoring import Scorer
from app.adapters.storage import Storage

log = logging.getLogger("pipeline")


class Processor:
    def __init__(self, storage: Storage, scorer: Scorer):
        self.storage = storage
        self.scorer = scorer
        self.counters: Counter = Counter()
        self.rule_hits: Counter = Counter()
        self._lock = threading.Lock()

    def _inc(self, key: str) -> None:
        with self._lock:
            self.counters[key] += 1

    def handle(self, raw: bytes) -> str:
        """Bad data is rejected (and acked), never raised. Infra errors propagate so the message is retried."""
        self._inc("received")
        try:
            txn = Transaction.model_validate_json(raw)
        except ValidationError as exc:
            self._inc("rejected")
            log.warning("rejected malformed message: %s", exc.errors()[:1])
            return "rejected"
        status, _ = self.process(txn)
        return status

    def process(self, txn: Transaction) -> tuple[str, ScoreResult | None]:
        # Duplicate check comes BEFORE scoring so a redelivered event can't inflate velocity counts.
        if self.storage.seen(txn.txn_id):
            self._inc("duplicates")
            return "duplicate", None
        result = self.scorer.score(txn)
        if self.storage.save_if_new(txn, result):
            self._inc("scored")
            with self._lock:
                for reason in result.reasons:
                    self.rule_hits[reason] += 1
            return "scored", result
        self._inc("duplicates")   # lost a race with a concurrent duplicate
        return "duplicate", None
