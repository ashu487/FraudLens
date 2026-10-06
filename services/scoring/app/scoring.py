import yaml
from app.models import Transaction, ScoreResult
from app.model import FraudModel
from app.rules import RuleEngine
from app.velocity import VelocityTracker, EMPTY


class Scorer:
    """Rules engine + velocity state + (optional) ML model. The model adds to the rule score."""

    def __init__(self, engine: RuleEngine, tracker: VelocityTracker | None = None,
                 model: FraudModel | None = None):
        self.engine = engine
        self.tracker = tracker or VelocityTracker(engine.retention_seconds)
        self.model = model

    @classmethod
    def from_file(cls, path: str, load_model: bool = True) -> "Scorer":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        model = FraudModel.from_config(cfg.get("model")) if load_model else None
        return cls(RuleEngine(cfg), model=model)

    def score(self, txn: Transaction) -> ScoreResult:
        windows = self.engine.windows
        velocity = (self.tracker.observe(txn.account_id, txn.timestamp.timestamp(), txn.amount, windows)
                    if windows else EMPTY)
        fired = self.engine.evaluate(txn, velocity)
        prob = None
        if self.model:
            prob = self.model.predict_proba(txn)
            fired = fired + self.model.hits(prob)
        risk, decision = self.engine.decide(fired)
        return ScoreResult(txn_id=txn.txn_id, risk_score=round(risk, 3), decision=decision,
                           reasons=[r.name for r in fired],
                           model_score=None if prob is None else round(prob, 5))

    def describe(self) -> dict:
        return {**self.engine.describe(), "model": self.model.describe() if self.model else None}
