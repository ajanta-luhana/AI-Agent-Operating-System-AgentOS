from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://agentos:agentos@localhost:5432/agentos"
    redis_url: str = "redis://localhost:6379/0"
    ollama_url: str = "http://localhost:11434"
    chat_model: str = "qwen2.5:14b-instruct"
    code_model: str = "qwen2.5-coder:14b"
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768

    runs_dir: str = "runs"
    host_runs_dir: str = ""          # absolute host path (for docker -v); defaults to abspath(runs_dir)
    sandbox_mode: str = "docker"     # docker | local (local = UNSAFE, dev/tests only)
    sandbox_image: str = "agentos-sandbox:latest"
    sandbox_timeout: int = 60

    max_retries: int = 2
    require_plan_approval: bool = True
    searxng_url: str = ""
    no_db: bool = False              # tests: skip Postgres/Redis event logging


settings = Settings()
