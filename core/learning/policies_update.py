from typing import Dict


class PolicyUpdater:
    """
    Перетворює training signals
    у рекомендації для системи.
    """

    def propose(self, stats: Dict) -> Dict:
        """
        Повертає пропозиції,
        а не застосовує їх.
        """

        proposals = {}

        if not stats:
            return proposals

        if stats["avg_groundedness"] < 0.5:
            proposals["retrieval"] = {
                "action": "increase_top_k",
                "reason": "Low groundedness"
            }

        if stats["avg_relevance"] < 0.5:
            proposals["reranking"] = {
                "action": "enable_cross_encoder",
                "reason": "Low relevance"
            }

        return proposals
