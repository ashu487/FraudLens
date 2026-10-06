from app.models import Transaction, ScoreResult
from app.rules import RuleEngine
from app.velocity import VelocityTracker, EMPTY


class Scorer:
    """Combines the rules engine with velocity state. The ML model joins in Week 4."""

    def __init__(self, engine: RuleEngine, tracker: VelocityTracker | None = None):
        self.engine = engine
        self.tracker = tracker or VelocityTracker(engine.retention_seconds)

    @classmethod
    def from_file(cls, path: str) -> "Scorer":
        return cls(RuleEngine.from_file(path))

    def score(self, txn: Transaction) -> ScoreResult:
        windows = self.engine.windows
        velocity = (self.tracker.observe(txn.account_id, txn.timestamp.timestamp(), txn.amount, windows)
                    if windows else EMPTY)
        return self.engine.score(txn, velocity)
