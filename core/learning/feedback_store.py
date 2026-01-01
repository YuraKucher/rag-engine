import json
import os
from typing import Dict
from datetime import datetime
import uuid


class FeedbackStore:
    """
    Сховище evaluation + user feedback.
    """

    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def save(self, evaluation: Dict) -> str:
        """
        Зберігає evaluation (без user feedback).
        """

        feedback_id = str(uuid.uuid4())

        record = {
            "feedback_id": feedback_id,
            "evaluation_id": evaluation["evaluation_id"],
            "question": evaluation["question"],
            "answer": evaluation["answer"],
            "metrics": evaluation["metrics"],
            "user_feedback": None,
            "created_at": datetime.utcnow().isoformat()
        }

        path = os.path.join(self.base_path, f"{feedback_id}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        return feedback_id

    def update_feedback(self, feedback_id: str, rating: int, comment: str = "") -> None:
        """
        Додає 👍 / 👎 після факту.
        rating: 1 (like), -1 (dislike)
        """

        path = os.path.join(self.base_path, f"{feedback_id}.json")

        if not os.path.exists(path):
            raise FileNotFoundError("Feedback record not found")

        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)

        record["user_feedback"] = {
            "rating": rating,
            "comment": comment,
            "timestamp": datetime.utcnow().isoformat()
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
