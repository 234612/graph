from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.neo4j import get_neo4j_driver

_neo4j_driver = None


async def init_clients():
    global _neo4j_driver
    _neo4j_driver = get_neo4j_driver()


async def close_clients():
    global _neo4j_driver
    if _neo4j_driver is not None:
        await _neo4j_driver.close()
        _neo4j_driver = None


def get_db():
    if _neo4j_driver is None:
        raise RuntimeError("客户端未初始化，请检查 lifespan 是否注册")
    return _neo4j_driver


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_clients()
    yield
    await close_clients()