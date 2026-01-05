"""
Learning Policies v1
====================

Rule-based learning policies.

Відповідальність:
- аналіз evaluation + feedback сигналів
- формування рекомендацій щодо оновлення системи

НЕ:
- не застосовує зміни напряму
- не викликає retrieval / generation
"""

from typing import Dict, List, Optional


class LearningPolicyEngine:
    """
    Rule-based engine для learning v1.
    """

    def __init__(self, thresholds: Optional[Dict] = None):
        self.thresholds = thresholds or {
            "relevance": 0.5,
            "groundedness": 0.5,
            "answerability": 0.4
        }

    def evaluate_policies(
        self,
        evaluation_stats: Dict,
        feedback_stats: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Аналізує статистику і повертає список policy proposals.
        """

        proposals: List[Dict] = []

        # --------------------------------------------------
        # Rule 1: Low groundedness → increase top_k
        # --------------------------------------------------
        if evaluation_stats.get("avg_groundedness", 1.0) < self.thresholds["groundedness"]:
            proposals.append({
                "policy": "retrieval.top_k",
                "action": "increase",
                "delta": 2,
                "reason": "Low groundedness → insufficient context"
            })

        # --------------------------------------------------
        # Rule 2: Low relevance → enable reranker
        # --------------------------------------------------
        if evaluation_stats.get("avg_relevance", 1.0) < self.thresholds["relevance"]:
            proposals.append({
                "policy": "retrieval.reranker",
                "action": "enable",
                "reason": "Low relevance → poor chunk selection"
            })

        # --------------------------------------------------
        # Rule 3: Low answerability → query rewriting
        # --------------------------------------------------
        if evaluation_stats.get("avg_answerability", 1.0) < self.thresholds["answerability"]:
            proposals.append({
                "policy": "retrieval.query_rewrite",
                "action": "enable",
                "reason": "Low answerability → complex or underspecified queries"
            })

        # --------------------------------------------------
        # Rule 4: Conflict detection (evaluation vs feedback)
        # --------------------------------------------------
        if feedback_stats:
            if feedback_stats.get("conflict_rate", 0.0) > 0.3:
                proposals.append({
                    "policy": "analysis",
                    "action": "flag",
                    "reason": "High conflict between evaluation and user feedback"
                })

        return proposals
