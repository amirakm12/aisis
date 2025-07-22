import sqlite3
from typing import Dict, Any
from .base_agent import BaseAgent
from loguru import logger

class PreferenceFeedbackLoopAgent(BaseAgent):
    def __init__(self):
        super().__init__("PreferenceFeedbackLoopAgent")
        self.db_path = "storage/user_preferences.sqlite"
        self.db = None

    async def _initialize(self) -> None:
        try:
            self.db = sqlite3.connect(self.db_path)
            self.db.execute("CREATE TABLE IF NOT EXISTS preferences (style TEXT PRIMARY KEY, count INTEGER DEFAULT 0)")
            logger.info("Preference database initialized")
        except Exception as e:
            logger.error(f"Failed to initialize preference database: {e}")
            raise

    async def _process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        style = task.get("style")
        if not style:
            return {"status": "error", "message": "No style provided"}
        try:
            self.db.execute(
                "INSERT OR REPLACE INTO preferences (style, count) VALUES (?, COALESCE((SELECT count FROM preferences WHERE style=?) + 1, 1))",
                (style, style)
            )
            self.db.commit()
            return {"status": "success", "updated_style": style}
        except Exception as e:
            logger.error(f"Failed to update preference: {e}")
            return {"status": "error", "message": str(e)}

    def get_preferences(self) -> list:
        if not self.db:
            return []
        return self.db.execute("SELECT * FROM preferences ORDER BY count DESC").fetchall()

    async def _cleanup(self) -> None:
        if self.db:
            self.db.close()
