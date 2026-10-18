from app.core.neo4j import get_neo4j_driver


class Neo4jWriter:
    def __init__(self):
        self._driver = get_neo4j_driver()

    async def run(self, cypher, **params):
        async with self._driver.session() as session:
            return await session.run(cypher, **params)

    async def close(self):
        await self._driver.close()