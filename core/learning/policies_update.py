"""
StatePolicyUpdater
==================

Єдиний policy engine для online learning (LTR).

Відповідає за:
- застосування evaluation (+ feedback) до state

НЕ:
- не змінює retrieval policy
- не змінює конфіги
- не приймає продуктових рішень
"""

from typing import Dict, Optional
from core.learning.state_maneger import StateManager


class StatePolicyUpdater:
    """
    Apply learning signals to state.
    """

    def __init__(self, state_manager: StateManager):
        self.state = state_manager

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def apply(self, evaluation: Dict, feedback: Optional[Dict] = None) -> None:
        """
        ЄДИНА точка застосування learning.
        """

        # ---------------- Documents ----------------
        for doc in evaluation.get("documents", []):
            self.state.update_document(
                doc_id=doc["document_id"],
                relevance=doc["relevance"],
                answerability=doc["answerability"],
            )

        # ---------------- Chunks ----------------
        for chunk in evaluation.get("chunks", []):
            self.state.update_chunk(
                chunk_id=chunk["chunk_id"],
                relevance=chunk["relevance"],
                groundedness=chunk["groundedness"],
            )

        # ---------------- Indexes (MULTI-INDEX) ----------------
        for index in evaluation.get("indexes", []):
            self.state.update_index(
                index_id=index["index_id"],
                relevance=index["avg_relevance"],
                groundedness=index["avg_groundedness"],
                used_chunks=len(index.get("used_chunks", []))
            )

        # ---------------- Optional human feedback ----------------
        if feedback:
            self._apply_human_signal(feedback)

        # 🔒 ЄДИНЕ місце save
        self.state.save_all()

    # --------------------------------------------------
    # INTERNALS
    # --------------------------------------------------

    def _apply_human_signal(self, feedback: Dict) -> None:
        """
        Людський сигнал — слабкий, глобальний, стабілізуючий.
        """

        rating = feedback.get("rating")
        if rating is None:
            return

        # ❗ людський фідбек НЕ повинен ламати структуру state
        # лише мʼяка корекція ваг

        if rating < 0:
            decay = 0.98
        elif rating > 0:
            decay = 1.02
        else:
            return

        for doc in self.state.document_state.values():
            doc["weight"] *= decay

        for chunk in self.state.chunk_state.values():
            chunk["weight"] *= decay

        for index in self.state.index_state.values():
            index["prior"] *= decay
