import json
import os
from datetime import datetime
from typing import Dict


class FeedbackStore:
    """
    Сховище user feedback (👍 / 👎).
    """

    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def save(self, evaluation: Dict) -> str:
        """
        Створює feedback shell для evaluation.
        Повертає feedback_id (== evaluation_id).
        """

        feedback_id = evaluation["evaluation_id"]

        record = {
            "evaluation_id": feedback_id,
            "rating": None,
            "comment": "",
            "created_at": datetime.utcnow().isoformat()
        }

        path = os.path.join(self.base_path, f"{feedback_id}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        return feedback_id

    def update_feedback(
        self,
        feedback_id: str,
        rating: int,
        comment: str = ""
    ) -> None:
        """
        Оновлює user feedback (👍 / 👎).
        """

        path = os.path.join(self.base_path, f"{feedback_id}.json")

        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)

        record["rating"] = rating
        record["comment"] = comment
        record["updated_at"] = datetime.utcnow().isoformat()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
