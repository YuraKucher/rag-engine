import os
import json
from typing import List, Dict


class Trainer:
    """
    Аналізує evaluation + feedback
    і формує навчальні сигнали.
    """

    def __init__(self, feedback_path: str):
        self.feedback_path = feedback_path

    def load_records(self) -> List[Dict]:
        records = []

        for file in os.listdir(self.feedback_path):
            if file.endswith(".json"):
                with open(os.path.join(self.feedback_path, file), "r", encoding="utf-8") as f:
                    records.append(json.load(f))

        return records

    def analyze(self) -> Dict:
        """
        Агрегує сигнали.
        Повертає статистику — НЕ рішення.
        """

        records = self.load_records()
        if not records:
            return {}

        relevance = [r["metrics"]["relevance"] for r in records]
        groundedness = [r["metrics"]["groundedness"] for r in records]
        answerability = [r["metrics"]["answerability"] for r in records]

        return {
            "count": len(records),
            "avg_relevance": sum(relevance) / len(relevance),
            "avg_groundedness": sum(groundedness) / len(groundedness),
            "avg_answerability": sum(answerability) / len(answerability)
        }
