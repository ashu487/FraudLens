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
