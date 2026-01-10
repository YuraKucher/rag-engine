import json
import os
import uuid
from typing import Dict


class EvaluationStore:
    """
    Persistent storage for automatic evaluations.

    Зберігає evaluation ПОВНІСТЮ, без втрати сигналу.
    """

    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def save(self, evaluation: Dict) -> str:
        """
        Зберігає evaluation як є і повертає evaluation_id.
        """

        evaluation_id = evaluation.get("evaluation_id") or str(uuid.uuid4())
        evaluation["evaluation_id"] = evaluation_id

        path = os.path.join(self.base_path, f"{evaluation_id}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, ensure_ascii=False, indent=2)

        return evaluation_id
