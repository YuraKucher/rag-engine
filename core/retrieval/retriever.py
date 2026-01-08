from typing import List, Optional

from core.indexing.index_manager import IndexManager
from .policies import RetrievalPolicy
from .query_rewriter import QueryRewriter


class Retriever:
    """
    Відповідає за semantic retrieval.

    Повертає ідентифікатори чанків,
    але також знає, ЯК саме був виконаний retrieval.
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

    def retrieve(self, query: str) -> List[str]:
        """
        Повертає список chunk_ids.
        """

        effective_query = query

        if self.policy.use_query_rewrite:
            effective_query = self.query_rewriter.rewrite(query)

        return self.index_manager.query(
            query=effective_query,
            k=self.policy.top_k
        )
