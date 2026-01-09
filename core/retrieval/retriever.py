from typing import List, Optional

from core.indexing.index_manager import IndexManager
from .policies import RetrievalPolicy
from .query_rewriter import QueryRewriter


class Retriever:
    """
    Відповідає за semantic retrieval.

    Відповідальність:
    - підготувати query (rewrite / normalize)
    - викликати index_manager
    - повернути candidate chunk_ids (recall stage)

    НЕ:
    - не ранжує
    - не фільтрує
    - не знає про чанки як обʼєкти
    """

    def __init__(
        self,
        index_manager: IndexManager,
        policy: RetrievalPolicy,
        query_rewriter: Optional[QueryRewriter] = None
    ):
        self.index_manager = index_manager
        self.policy = policy
        self.query_rewriter = query_rewriter or QueryRewriter()

    # ======================================================
    # PUBLIC API
    # ======================================================

    def retrieve(self, query: str) -> List[str]:
        """
        Повертає список chunk_ids (recall stage).
        """

        effective_query = self._prepare_query(query)

        return self.index_manager.query(
            query=effective_query,
            k=self.policy.top_k
        )

    # ======================================================
    # INTERNALS
    # ======================================================

    def _prepare_query(self, query: str) -> str:
        """
        Готує запит до retrieval:
        - rewrite (опційно)
        - normalization (на майбутнє)
        """

        if self.policy.use_query_rewrite:
            return self.query_rewriter.rewrite(query)

        return query
