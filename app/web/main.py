import uvicorn
from fastapi import FastAPI

from app.web.deps import get_db, lifespan

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def read_root():
    driver = get_db()
    await driver.verify_connectivity()
    return {"msg": "Hello", "neo4j": "connected"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)