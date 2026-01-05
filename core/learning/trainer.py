"""
Trainer
=======

Агрегує evaluation та feedback дані
і запускає learning policies.
"""

import os
import json
from typing import Dict, List

from core.learning.policies_update import LearningPolicyEngine


class Trainer:
    """
    Learning orchestrator (v1).
    """

    def __init__(
        self,
        evaluation_path: str,
        feedback_path: str
    ):
        self.evaluation_path = evaluation_path
        self.feedback_path = feedback_path
        self.policy_engine = LearningPolicyEngine()

    # --------------------------------------------------
    # DATA LOADING
    # --------------------------------------------------

    def _load_evaluations(self) -> List[Dict]:
        records = []
        for fname in os.listdir(self.evaluation_path):
            if fname.endswith(".json"):
                with open(os.path.join(self.evaluation_path, fname), "r", encoding="utf-8") as f:
                    records.append(json.load(f))
        return records

    def _load_feedback(self) -> List[Dict]:
        if not os.path.exists(self.feedback_path):
            return []

        records = []
        for fname in os.listdir(self.feedback_path):
            if fname.endswith(".json"):
                with open(os.path.join(self.feedback_path, fname), "r", encoding="utf-8") as f:
                    records.append(json.load(f))
        return records

    # --------------------------------------------------
    # STATISTICS
    # --------------------------------------------------

    def _aggregate_evaluation_stats(self, evaluations: List[Dict]) -> Dict:
        if not evaluations:
            return {}

        relevance = []
        groundedness = []
        answerability = []

        for e in evaluations:
            metrics = e.get("metrics", {})
            if "relevance" in metrics:
                relevance.append(metrics["relevance"])
            if "groundedness" in metrics:
                groundedness.append(metrics["groundedness"])
            if "answerability" in metrics:
                answerability.append(metrics["answerability"])

        return {
            "avg_relevance": sum(relevance) / len(relevance) if relevance else 1.0,
            "avg_groundedness": sum(groundedness) / len(groundedness) if groundedness else 1.0,
            "avg_answerability": sum(answerability) / len(answerability) if answerability else 1.0
        }

    def _aggregate_feedback_stats(self, feedback: List[Dict]) -> Dict:
        if not feedback:
            return {}

        conflicts = 0
        total = len(feedback)

        for f in feedback:
            if f.get("rating") == -1:
                conflicts += 1

        return {
            "conflict_rate": conflicts / total if total else 0.0
        }

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def run_learning(self) -> List[Dict]:
        """
        Запускає learning cycle і повертає policy proposals.
        """

        evaluations = self._load_evaluations()
        feedback = self._load_feedback()

        evaluation_stats = self._aggregate_evaluation_stats(evaluations)
        feedback_stats = self._aggregate_feedback_stats(feedback)

        proposals = self.policy_engine.evaluate_policies(
            evaluation_stats=evaluation_stats,
            feedback_stats=feedback_stats
        )

        return proposals
