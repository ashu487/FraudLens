# Real-Time Fraud Detection Platform

Streaming fraud detection with a rules engine, ML scoring, explainability and an
analyst feedback loop. Built local-first (Docker Compose + Pub/Sub emulator),
deployable to GCP (Cloud Run, Pub/Sub, BigQuery).

## Quick start
```bash
cp .env.example .env
make up
curl localhost:8080/health
```

## Structure
- `services/scoring` - FastAPI scoring service
- `services/simulator` - transaction generator
- `infra/terraform` - GCP infrastructure (Week 7)
- `docs/` - design doc, architecture diagram
- `tests/` - unit and integration tests

## Roadmap
- [x] Week 1: scaffold, design doc
- [ ] Week 2: simulator + ingestion
- [ ] Week 3: rules engine
- [ ] Week 4: ML model
- [ ] Week 5: scoring pipeline + SHAP
- [ ] Week 6: dashboard + feedback loop
- [ ] Week 7: Terraform, CI/CD, GCP deploy, load test
- [ ] Week 8: polish, demo, blog
