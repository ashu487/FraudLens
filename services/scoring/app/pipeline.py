"""Message handling: validate -> score -> store (idempotent)."""
import logging
import threading
from collections import Counter
from pydantic import ValidationError
from app.models import Transaction, ScoreResult
from app.scoring import score
from app.adapters.storage import Storage

log = logging.getLogger("pipeline")


class Processor:
    def __init__(self, storage: Storage):
        self.storage = storage
        self.counters: Counter = Counter()
        self._lock = threading.Lock()

    def _inc(self, key: str) -> None:
        with self._lock:
            self.counters[key] += 1

    def handle(self, raw: bytes) -> str:
        """Process one raw message. Bad data is rejected (and acked), never raised.
        Infrastructure errors (e.g. storage) propagate so the message is retried."""
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
        result = score(txn)
        if self.storage.save_if_new(txn, result):
            self._inc("scored")
            return "scored", result
        self._inc("duplicates")
        return "duplicate", None
