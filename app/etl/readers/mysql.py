from sqlalchemy import text

from app.core.mysql import get_mysql_engine


class MysqlReader:
    def __init__(self):
        self._engine = get_mysql_engine()

    def query(self, sql, **params):
        with self._engine.connect() as conn:
            result = conn.execute(text(sql), params)
            return result.mappings().all()

    def close(self):
        self._engine.dispose()