"""Rules engine v2: rule logic lives here, rule settings live in config/rules.yaml."""
from dataclasses import dataclass
from typing import Callable
import yaml
from app.models import ScoreResult

RISKY_TYPES = {"TRANSFER", "CASH_OUT"}


# ---- rule predicates: fn(txn, params, velocity) -> bool ----
def account_drained(t, p, v) -> bool:
    # Exact-match only. `amount >= balance` was the v1 bug: PaySim has many legit
    # transactions whose amount exceeds the balance, which caused ~80% false positives.
    return (t.type in RISKY_TYPES and t.old_balance_org > 0
            and abs(t.amount - t.old_balance_org) <= p.get("tolerance", 0.01))


def large_risky_amount(t, p, v) -> bool:
    return t.type in RISKY_TYPES and t.amount >= p["min_amount"]


def zero_amount(t, p, v) -> bool:
    return t.type in RISKY_TYPES and t.amount == 0


def velocity_burst(t, p, v) -> bool:
    return v.count(p["window_seconds"]) >= p["max_txns"]


def velocity_amount(t, p, v) -> bool:
    w = p["window_seconds"]
    return v.count(w) >= p.get("min_txns", 2) and v.total(w) >= p["max_total"]


RULE_FUNCS: dict[str, Callable] = {
    "account_drained": account_drained,
    "large_risky_amount": large_risky_amount,
    "zero_amount": zero_amount,
    "velocity_burst": velocity_burst,
    "velocity_amount": velocity_amount,
}


@dataclass(frozen=True)
class Rule:
    name: str
    weight: float
    params: dict
    fn: Callable

    @property
    def needs_velocity(self) -> bool:
        return "window_seconds" in self.params


class RuleEngine:
    def __init__(self, config: dict):
        th = config.get("thresholds", {})
        self.review, self.block = th.get("review", 0.5), th.get("block", 0.8)
        if not 0 < self.review < self.block <= 1:
            raise ValueError("thresholds must satisfy 0 < review < block <= 1")
        self.rules: list[Rule] = []
        for name, spec in (config.get("rules") or {}).items():
            spec = spec or {}
            if name not in RULE_FUNCS:
                raise ValueError(f"unknown rule in config: {name}")
            if not spec.get("enabled", True):
                continue
            weight = float(spec["weight"])
            if not 0 < weight <= 1:
                raise ValueError(f"rule {name}: weight must be in (0, 1]")
            self.rules.append(Rule(name, weight, spec.get("params") or {}, RULE_FUNCS[name]))

    @classmethod
    def from_file(cls, path: str) -> "RuleEngine":
        with open(path) as f:
            return cls(yaml.safe_load(f))

    @property
    def windows(self) -> list[int]:
        return sorted({r.params["window_seconds"] for r in self.rules if r.needs_velocity})

    @property
    def retention_seconds(self) -> int:
        return max(self.windows, default=0)

    def evaluate(self, txn, velocity) -> list[Rule]:
        return [r for r in self.rules if r.fn(txn, r.params, velocity)]

    def decide(self, fired: list[Rule]) -> tuple[float, str]:
        risk = min(1.0, sum(r.weight for r in fired))
        decision = "block" if risk >= self.block else "review" if risk >= self.review else "approve"
        return risk, decision

    def score(self, txn, velocity) -> ScoreResult:
        fired = self.evaluate(txn, velocity)
        risk, decision = self.decide(fired)
        return ScoreResult(txn_id=txn.txn_id, risk_score=round(risk, 3),
                           decision=decision, reasons=[r.name for r in fired])

    def describe(self) -> dict:
        return {"thresholds": {"review": self.review, "block": self.block},
                "rules": [{"name": r.name, "weight": r.weight, "params": r.params} for r in self.rules]}
