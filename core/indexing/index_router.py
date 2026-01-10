"""
IndexRouter
===========

Semantic router for index selection.

Відповідає за:
- аналіз типу питання
- вибір semantic roles індексів

НЕ:
- не знає реальних index_id
- не лоадить індекси
- не приймає рішень про retrieval
"""

import re
from typing import Dict, List


class SemanticIndexRouter:
    """
    Lightweight semantic router.
    """

    def __init__(self):
        self._definition_patterns = re.compile(
            r"^(what is|who is|define)\b",
            re.IGNORECASE
        )

        self._procedure_patterns = re.compile(
            r"\b(how to|steps?|install|configure|setup)\b",
            re.IGNORECASE
        )

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def route(self, question: str) -> List[Dict]:
        """
        Повертає список semantic roles індексів
        у порядку пріоритету.
        """

        roles: List[Dict] = []

        q = question.strip()

        # --------- definition ---------
        if self._definition_patterns.search(q):
            roles.append(self._role("definition", score=1.0))

        # --------- procedure ---------
        if self._procedure_patterns.search(q):
            roles.append(self._role("procedure", score=0.9))

        # --------- fallback ---------
        roles.append(self._role("general", score=0.5))

        return roles

    # --------------------------------------------------
    # INTERNALS
    # --------------------------------------------------

    @staticmethod
    def _role(role: str, score: float) -> Dict:
        return {
            "index_role": role,
            "router_score": score
        }
