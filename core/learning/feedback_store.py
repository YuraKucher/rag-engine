import json
import os
from datetime import datetime
from typing import Dict, Optional


class FeedbackStore:
    """
    Persistent user feedback storage (👍 / 👎).

    НЕ:
    - не застосовує learning
    - не знає структуру evaluation
    """

    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    # --------------------------------------------------
    # CREATE
    # --------------------------------------------------

    def create(self, evaluation_id: str) -> str:
        """
        Створює feedback shell для evaluation.
        feedback_id == evaluation_id
        """

        record = {
            "feedback_id": evaluation_id,
            "evaluation_id": evaluation_id,
            "rating": None,
            "comment": "",
            "applied": False,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": None,
        }

        path = os.path.join(self.base_path, f"{evaluation_id}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        return evaluation_id

    # --------------------------------------------------
    # UPDATE
    # --------------------------------------------------

    def update(
        self,
        feedback_id: str,
        rating: int,
        comment: str = ""
    ) -> None:
        """
        Оновлює user feedback (👍 / 👎).
        rating: -1 | 0 | 1
        """

        path = os.path.join(self.base_path, f"{feedback_id}.json")

        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)

        record["rating"] = rating
        record["comment"] = comment
        record["updated_at"] = datetime.utcnow().isoformat()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    # --------------------------------------------------
    # READ
    # --------------------------------------------------

    def load(self, feedback_id: str) -> Optional[Dict]:
        """
        Завантажує feedback за id.
        """
        path = os.path.join(self.base_path, f"{feedback_id}.json")
        if not os.path.exists(path):
            return None

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def mark_applied(self, feedback_id: str) -> None:
        """
        Позначає feedback як застосований до state.
        """
        path = os.path.join(self.base_path, f"{feedback_id}.json")

        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)

        record["applied"] = True
        record["updated_at"] = datetime.utcnow().isoformat()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
