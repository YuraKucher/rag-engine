import json
import os
import uuid
from datetime import datetime
from typing import Dict


class EvaluationStore:
    """
    Сховище evaluation результатів (автоматична оцінка).
    """

    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def save(self, evaluation: Dict) -> str:
        """
        Зберігає evaluation і повертає evaluation_id.
        """

        evaluation_id = evaluation.get("evaluation_id") or str(uuid.uuid4())

        record = {
            "evaluation_id": evaluation_id,
            "question": evaluation["question"],
            "answer": evaluation["answer"],
            "metrics": evaluation["metrics"],
            "created_at": datetime.utcnow().isoformat()
        }

        path = os.path.join(self.base_path, f"{evaluation_id}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        return evaluation_id
