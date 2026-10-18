import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neo4j import GraphDatabase

from app.config import get_settings


if __name__ == "__main__":
    settings = get_settings().neo4j

    with GraphDatabase.driver(
        settings.uri, auth=(settings.user, settings.password)
    ) as driver:
        records, summary, keys = driver.execute_query(
            """
            MATCH (p:Person{name:$name})-[r:DIRECTED]->(m:Movie)
            WHERE m.year > $year
            RETURN p.name AS director, m.year AS year, m.title AS movie
            """,
            parameters_={"name": "张艺谋", "year": 1990},
            database_=settings.database,
        )

        print(f"查询返回了 {len(records)} 条记录")
        for record in records:
            print(f"{record['director']}在{record['year']}年拍摄了《{record['movie']}》")