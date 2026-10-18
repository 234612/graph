from app.web.deps import get_db


class ChatService:
    def __init__(self):
        self._driver = get_db()