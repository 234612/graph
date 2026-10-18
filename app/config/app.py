from pydantic_settings import BaseSettings, SettingsConfigDict

from .infra import (
    ENV_FILE,
    EmbeddingSettings,
    LLMSettings,
    MySQLSettings,
    Neo4jSettings,
    Paths,
)


class AppSettings(BaseSettings):
    app_name: str = "app"
    environment: str = "dev"

    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    mysql: MySQLSettings = MySQLSettings()
    neo4j: Neo4jSettings = Neo4jSettings()
    paths: Paths = Paths()

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        case_sensitive=False,
        extra="ignore",
    )