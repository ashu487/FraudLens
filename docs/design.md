# Design Doc (draft)

## Goal
Real-time fraud detection platform: stream transactions, score with rules + ML,
explain decisions, give analysts a dashboard and a feedback loop.

## Success metrics
- Recall >= __% at false positive rate <= __%
- p95 scoring latency <= __ ms
- Sustained throughput >= __ txn/min

## Architecture
Simulator -> Pub/Sub -> Scoring service (rules + model) -> Storage (BigQuery / DuckDB)
-> Dashboard. Cloud pieces sit behind adapters so local and GCP share one codebase.

## Decisions & tradeoffs
(Add as you go. Interviewers love this section.)
Rules v2 on full PaySim: precision 100%, recall 97.8%, false positive rate 0%; 179 fraud cases missed (154 TRANSFER, 25 CASH_OUT, 25 with origin balance 0)

week 3 result: {"counters":{"received":1287,"scored":1247,"duplicates":29,"rejected":11},"rule_hits":{"account_drained":54,"large_risky_amount":258,"velocity_burst":24},"by_decision":{"review":24,"approve":1169,"block":54},"confusion":{"tp":78,"fp":0,"fn":24,"tn":1145},"precision":1.0,"recall":0.7647,"false_positive_rate":0.0}
