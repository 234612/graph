from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"

_ENV_SETTINGS = {
    "env_file": ENV_FILE,
    "case_sensitive": False,
    "extra": "ignore",
}


class MySQLSettings(BaseSettings):
    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = "app-db"

    model_config = SettingsConfigDict(**_ENV_SETTINGS, env_prefix="MYSQL_")


class Neo4jSettings(BaseSettings):
    uri: str = "neo4j://localhost:7687"
    user: str = "neo4j"
    password: str = ""
    database: str = "neo4j"

    model_config = SettingsConfigDict(**_ENV_SETTINGS, env_prefix="NEO4J_")


class LLMSettings(BaseSettings):
    provider: str = "openai"
    api_key: SecretStr = ""
    model: str = "gpt-4o-mini"
    base_url: str | None = None

    model_config = SettingsConfigDict(**_ENV_SETTINGS, env_prefix="LLM_")


class EmbeddingSettings(BaseSettings):
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    base_url: str | None = None

    model_config = SettingsConfigDict(**_ENV_SETTINGS, env_prefix="EMBEDDING_")


class Paths(BaseSettings):
    data_dir: Path = DATA_DIR
    checkpoints_dir: Path = CHECKPOINTS_DIR
    logs_dir: Path = LOGS_DIR

    model_config = SettingsConfigDict(**_ENV_SETTINGS, env_prefix="PATHS_")