from app.core.neo4j import get_neo4j_driver


class Neo4jWriter:
    def __init__(self):
        self._driver = get_neo4j_driver()

    def run(self, cypher: str, **params):
        with self._driver.session() as session:
            return session.run(cypher, **params)

    def ensure_constraints(self):
        with self._driver.session() as session:
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (s:SPU) REQUIRE s.spu_id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (t:Tag) REQUIRE t.name IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (c:Category3) REQUIRE c.category3_id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (b:Trademark) REQUIRE b.tm_id IS UNIQUE"
            )

    def close(self):
        self._driver.close()
