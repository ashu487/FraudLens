"""Per-account sliding-window velocity features, kept in memory.

Uses event time (the transaction's own timestamp), not wall-clock time, so replays
and tests are deterministic. Slightly out-of-order events are tolerated.
On GCP with several Cloud Run instances this state must move to a shared store
(Redis/Memorystore or Firestore); the interface can stay the same.
"""
import threading
from collections import deque


class VelocitySnapshot:
    """Window stats for one account, as of one transaction (current txn included)."""

    def __init__(self, stats: dict[int, tuple[int, float]]):
        self._stats = stats

    def count(self, window_seconds: int) -> int:
        return self._stats.get(window_seconds, (0, 0.0))[0]

    def total(self, window_seconds: int) -> float:
        return self._stats.get(window_seconds, (0, 0.0))[1]


EMPTY = VelocitySnapshot({})


class VelocityTracker:
    def __init__(self, retention_seconds: int = 3600):
        self.retention = retention_seconds
        self._events: dict[str, deque] = {}
        self._lock = threading.Lock()
        self._ops = 0

    def observe(self, account_id: str, ts: float, amount: float, windows) -> VelocitySnapshot:
        with self._lock:
            dq = self._events.setdefault(account_id, deque())
            dq.append((ts, amount))
            horizon = ts - self.retention
            while dq and dq[0][0] < horizon:
                dq.popleft()
            stats = {}
            for w in windows:
                cutoff, count, total = ts - w, 0, 0.0
                for t, a in reversed(dq):
                    if t < cutoff:
                        break
                    count += 1
                    total += a
                stats[w] = (count, total)
            self._ops += 1
            if self._ops % 10_000 == 0:
                self._evict(horizon)
            return VelocitySnapshot(stats)

    def _evict(self, horizon: float) -> None:
        for acct in [a for a, dq in self._events.items() if not dq or dq[-1][0] < horizon]:
            del self._events[acct]

    def tracked_accounts(self) -> int:
        return len(self._events)
