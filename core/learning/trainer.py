"""
Trainer
=======

Offline analytics helper.

Агрегує evaluation та feedback дані
для аналізу якості роботи системи.

НЕ:
- не оновлює state
- не застосовує learning
- не впливає на retrieval / rerank
"""

import os
import json
from typing import Dict, List


class Trainer:
    """
    Offline evaluation analyzer.
    """

    def __init__(
        self,
        evaluation_path: str,
        feedback_path: str
    ):
        self.evaluation_path = evaluation_path
        self.feedback_path = feedback_path

    # --------------------------------------------------
    # DATA LOADING
    # --------------------------------------------------

    def _load_evaluations(self) -> List[Dict]:
        if not os.path.exists(self.evaluation_path):
            return []

        records = []
        for fname in os.listdir(self.evaluation_path):
            if fname.endswith(".json"):
                with open(
                    os.path.join(self.evaluation_path, fname),
                    "r",
                    encoding="utf-8"
                ) as f:
                    records.append(json.load(f))
        return records

    def _load_feedback(self) -> List[Dict]:
        if not os.path.exists(self.feedback_path):
            return []

        records = []
        for fname in os.listdir(self.feedback_path):
            if fname.endswith(".json"):
                with open(
                    os.path.join(self.feedback_path, fname),
                    "r",
                    encoding="utf-8"
                ) as f:
                    records.append(json.load(f))
        return records

    # --------------------------------------------------
    # AGGREGATION
    # --------------------------------------------------

    def _aggregate_index_stats(self, evaluations: List[Dict]) -> Dict:
        stats: Dict[str, Dict] = {}

        for e in evaluations:
            for idx in e.get("indexes", []):
                index_id = idx["index_id"]
                s = stats.setdefault(
                    index_id,
                    {
                        "count": 0,
                        "avg_relevance": 0.0,
                        "avg_groundedness": 0.0,
                    }
                )

                s["count"] += 1
                s["avg_relevance"] += idx["avg_relevance"]
                s["avg_groundedness"] += idx["avg_groundedness"]

        for s in stats.values():
            if s["count"] > 0:
                s["avg_relevance"] /= s["count"]
                s["avg_groundedness"] /= s["count"]

        return stats

    def _aggregate_document_stats(self, evaluations: List[Dict]) -> Dict:
        stats: Dict[str, Dict] = {}

        for e in evaluations:
            for doc in e.get("documents", []):
                doc_id = doc["document_id"]
                s = stats.setdefault(
                    doc_id,
                    {
                        "count": 0,
                        "avg_relevance": 0.0,
                        "avg_answerability": 0.0,
                    }
                )

                s["count"] += 1
                s["avg_relevance"] += doc["relevance"]
                s["avg_answerability"] += doc["answerability"]

        for s in stats.values():
            if s["count"] > 0:
                s["avg_relevance"] /= s["count"]
                s["avg_answerability"] /= s["count"]

        return stats

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def run_analysis(self) -> Dict:
        """
        Повертає агреговану статистику для дебагу / UI.
        """

        evaluations = self._load_evaluations()
        feedback = self._load_feedback()

        return {
            "evaluations_count": len(evaluations),
            "feedback_count": len(feedback),
            "index_stats": self._aggregate_index_stats(evaluations),
            "document_stats": self._aggregate_document_stats(evaluations),
        }
