from app.config import get_settings


def get_mysql_engine():
    from sqlalchemy import create_engine

    settings = get_settings().mysql
    url = (
        f"mysql+pymysql://{settings.user}:{settings.password}"
        f"@{settings.host}:{settings.port}/{settings.database}?charset=utf8mb4"
    )
    return create_engine(url, pool_pre_ping=True)