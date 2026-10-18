from functools import lru_cache

from neo4j import GraphDatabase

from app.config import get_settings


@lru_cache(maxsize=1)
def get_neo4j_driver():
    """返回进程内唯一的同步 Neo4j driver。"""
    settings = get_settings().neo4j
    return GraphDatabase.driver(
        settings.uri,
        auth=(settings.user, settings.password),
    )

