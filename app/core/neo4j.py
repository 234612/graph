from neo4j import GraphDatabase

from app.config import get_settings


def get_neo4j_driver():
    """返回一个同步 Neo4j driver。调用方负责在使用后 close()。

    注意：不缓存为单例——每个 Neo4jWriter 持有并关闭自己的 driver，
    避免多个同步模块（sync_table / sync_text）共享同一实例时，其中一个
    先 close 导致后续模块拿到已关闭的 driver。
    """
    settings = get_settings().neo4j
    return GraphDatabase.driver(
        settings.uri,
        auth=(settings.user, settings.password),
    )
