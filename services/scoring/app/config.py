from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_env: str = "local"
    gcp_project: str = "fraud-local"
    pubsub_topic: str = "transactions"
    pubsub_subscription: str = "transactions-scoring"
    queue_backend: str = "pubsub"
    storage_backend: str = "duckdb"
    duckdb_path: str = "/data/fraud.duckdb"
    rules_config: str = "config/rules.yaml"


settings = Settings()
