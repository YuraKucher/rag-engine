"""
Indexing Service
================

Сервіс індексації для RAG-системи.

Відповідальність:
- створення індексів (chunks, evaluation)
- формування Index-метаданих (index.schema.v2.json)
- підготовка даних для retrieval / learning
"""

from typing import Dict, List
from datetime import datetime
import uuid

from core.indexing.index_manager import IndexManager
from core.indexing.faiss_index import FaissIndex
from core.indexing.embedder import Embedder


class IndexingService:
    """
    Сервіс індексації.
    """

    def __init__(
        self,
        index_manager: IndexManager,
        evaluation_index_path: str
    ):
        self.index_manager = index_manager
        self.embedder: Embedder = index_manager.embedder

        # Окремий FAISS-індекс для evaluation
        self.evaluation_index = FaissIndex(
            embedding_dim=self.embedder.embedding_dim,
            index_path=evaluation_index_path
        )

    # ==================================================
    # CHUNK INDEXING
    # ==================================================

    def index_chunks(self, chunks: List[Dict]) -> Dict:
        """
        Індексує чанки документа.
        Повертає Index metadata (schema v2).
        """

        # 1. Побудова індексу через IndexManager
        self.index_manager.build_index(chunks)

        index_id = str(uuid.uuid4())
        chunk_ids = [c["chunk_id"] for c in chunks]

        # 2. Формуємо Index-обʼєкт (schema v2)
        return {
            "index_id": index_id,
            "index_type": "faiss",
            "index_scope": "chunk",
            "embedding_model": self.embedder.model_name,
            "target_ids": chunk_ids,
            "created_at": datetime.utcnow().isoformat()
        }

    # ==================================================
    # EVALUATION INDEXING
    # ==================================================

    def index_evaluation(self, evaluation: Dict) -> Dict:
        """
        Індексує evaluation результат.
        Повертає Index metadata (schema v2).
        """

        index_id = str(uuid.uuid4())
        evaluation_id = evaluation["evaluation_id"]

        # 1. Embed питання (основа для learning)
        embedding = self.embedder.embed(evaluation["question"])

        # 2. Додаємо у FAISS
        self.evaluation_index.add(
            vectors=[embedding],
            metadata=[{
                "evaluation_id": evaluation_id,
                "metrics": evaluation["metrics"]
            }]
        )

        self.evaluation_index.save()

        # 3. Формуємо Index-обʼєкт
        return {
            "index_id": index_id,
            "index_type": "faiss",
            "index_scope": "evaluation",
            "embedding_model": self.embedder.model_name,
            "target_ids": [evaluation_id],
            "created_at": datetime.utcnow().isoformat()
        }
