from typing import List, Optional, Dict

from core.indexing.index_manager import IndexManager
from .policies import RetrievalPolicy
from .query_rewriter import QueryRewriter


class Retriever:
    """
    Semantic retriever (recall stage).

    Відповідальність:
    - підготувати запит
    - вибрати релевантні індекси (через IndexManager)
    - виконати recall (chunk_ids)

    НЕ:
    - не ранжує
    - не працює зі state
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

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def retrieve(
        self,
        query: str,
        index_roles: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        Повертає список candidate chunk_ids.

        index_roles:
        [
          {"index_role": "definition", "router_score": 1.0},
          {"index_role": "general", "router_score": 0.5}
        ]
        """
        effective_query = self._prepare_query(query)

        # 1️⃣ Визначаємо index_ids для пошуку
        index_ids = self._resolve_index_ids(index_roles)

        # 2️⃣ Recall по кожному індексу
        all_chunk_ids: List[str] = []

        for index_id in index_ids:
            chunk_ids = self.index_manager.query(
                query=effective_query,
                k=self.policy.top_k,
                index_id=index_id
            )
            all_chunk_ids.extend(chunk_ids)

        return all_chunk_ids

    # --------------------------------------------------
    # INTERNALS
    # --------------------------------------------------

    def _resolve_index_ids(
        self,
        index_roles: Optional[List[Dict]]
    ) -> List[str]:
        """
        Перетворює roles → concrete index_ids.
        """

        # fallback: всі індекси
        if not index_roles:
            return self.index_manager.list_indexes()

        resolved: List[str] = []

        for role in index_roles:
            role_name = role.get("index_role")
            if not role_name:
                continue

            ids = self.index_manager.get_indexes_by_role(role_name)
            resolved.extend(ids)

        # дедуплікація
        return list(dict.fromkeys(resolved))

    def _prepare_query(self, query: str) -> str:
        """
        Готує запит до retrieval.
        """
        if self.policy.use_query_rewrite:
            return self.query_rewriter.rewrite(query)
        return query.strip()
