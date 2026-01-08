"""
RAG Service
===========

Єдина точка входу в RAG-систему.

ВАЖЛИВО:
- тут НІЯКОЇ бізнес-логіки
- тут НІЯКИХ евристик
- лише оркестрація шарів
"""

# ======================================================
# IMPORTS
# ======================================================
from typing import Dict

from core.cache.manager import CacheManager
# Cache
from core.cache.semantic_cache import SemanticCache
# Chunking
from core.chunking.chunker import Chunker
# Evaluation
from core.evaluation.evaluator import Evaluator
# Generation
from core.generation.llm_client import LLMClient
from core.generation.prompts import PromptFactory
# Indexing
from core.indexing.index_manager import IndexManager
# Ingestion
from core.ingestion.registry import LoaderRegistry
# Knowledge
from core.knowledge.chunk_store import ChunkStore
from core.knowledge.document_store import DocumentStore
# Learning / Feedback
from core.learning.feedback_store import FeedbackStore
# Reasoning
from core.reasoning.agent import ReasoningAgent
from core.reasoning.strategies import ReasoningStrategy
# Retrieval
from core.retrieval.policies import RetrievalPolicy
from core.retrieval.reranker import Reranker
from core.retrieval.retriever import Retriever


# ======================================================
# RAG SERVICE
# ======================================================

class RAGService:
    """
    Оркестратор всієї RAG-системи.
    """

    def __init__(
        self,
        documents_path: str,
        chunks_path: str,
        feedback_path: str,
        embedding_model: str,
        llm_model: str
    ):
        # ---------------- Ingestion / Chunking ----------------
        self.chunker = Chunker()

        # ---------------- Knowledge ----------------
        self.document_store = DocumentStore(documents_path)
        self.chunk_store = ChunkStore(chunks_path)

        # ---------------- Indexing ----------------
        indexes_path = chunks_path.replace("chunks", "indexes")
        self.index_manager = IndexManager(
            embedding_model=embedding_model,
            indexes_path=indexes_path
        )

        # ---------------- Retrieval ----------------
        self.retrieval_policy = RetrievalPolicy(
            top_k=5,
            rerank_k=3,
            use_query_rewrite=False
        )
        self.retriever = Retriever(
            index_manager=self.index_manager,
            policy=self.retrieval_policy
        )
        self.reranker = Reranker()

        # ---------------- Reasoning ----------------
        self.reasoning_agent = ReasoningAgent(
            strategy=ReasoningStrategy.QA
        )

        # ---------------- Generation ----------------
        self.llm_client = LLMClient(
            model_name=llm_model
        )

        # ---------------- Evaluation ----------------
        self.evaluator = Evaluator(
            embedder=self.index_manager.embedder
        )

        # ---------------- Learning / Feedback ----------------
        self.feedback_store = FeedbackStore(
            base_path=feedback_path
        )

        # ---------------- Cache ----------------
        self.semantic_cache = SemanticCache(
            embedder=self.index_manager.embedder,
            similarity_threshold=0.9
        )
        self.cache_manager = CacheManager(
            semantic_cache=self.semantic_cache,
            ttl=60 * 60
        )

        # ---------------- Index restore ----------------
        self._restore_index()

    # ======================================================
    # INDEX RESTORE
    # ======================================================

    def _restore_index(self) -> None:
        """
        Відновлює FAISS-індекс з диску через IndexManager metadata.
        НІЯКОЇ індексації, тільки load.
        """

        # знаходимо всі metadata-файли індексів
        metadata_files = list(
            self.index_manager.indexes_path.glob("*.index.json")
        )

        if not metadata_files:
            return

        # просте правило: беремо останній створений індекс
        metadata_files.sort(
            key=lambda p: p.stat().st_mtime
        )
        latest_metadata = metadata_files[-1]

        index_id = latest_metadata.stem.replace(".index", "")

        # delegate everything to IndexManager
        self.index_manager.load_index(index_id)

    # ======================================================
    # INGESTION PIPELINE
    # ======================================================

    def ingest_document(self, source: str, file_type: str) -> Dict:
        """
        Повний pipeline індексації документа.
        """

        # 1. Load document
        loader = LoaderRegistry.get_loader(file_type)
        document = loader.load(source)

        # 2. Save document
        self.document_store.save(document)

        # 3. Chunking
        chunks = self.chunker.split(document)

        # 4. Build index
        index_metadata = self.index_manager.build_index(chunks)

        # 5. Save chunks WITH index binding
        for chunk in chunks:
            chunk.setdefault("metadata", {})
            chunk["metadata"]["index_id"] = index_metadata["index_id"]
            chunk["metadata"]["embedding_model"] = index_metadata["embedding_model"]
            self.chunk_store.save(chunk)

        return {
            "document_id": document["document_id"],
            "chunks_count": len(chunks),
            "index": index_metadata
        }

    # ======================================================
    # QUESTION ANSWERING PIPELINE
    # ======================================================

    def ask(self, question: str) -> Dict:
        """
        Повний RAG pipeline + evaluation.
        """

        # 0. Semantic cache lookup
        cached = self.semantic_cache.lookup(question)
        if cached is not None:
            return cached

        # 1. Retrieval
        chunk_ids = self.retriever.retrieve(question)
        chunks = [
            self.chunk_store.load(chunk_id)
            for chunk_id in chunk_ids
            if self.chunk_store.load(chunk_id) is not None
        ]

        # 2. Reranking
        ranked_chunks = self.reranker.rerank(
            query=question,
            chunks=chunks,
            top_k=self.retrieval_policy.rerank_k
        )

        # 3. Reasoning
        reasoning_payload = self.reasoning_agent.prepare(
            question=question,
            chunks=ranked_chunks
        )

        # 4. Prompt
        prompt = PromptFactory.qa_prompt(reasoning_payload)

        # 5. Generation
        answer = self.llm_client.generate(prompt)

        # 6. Evaluation
        evaluation = self.evaluator.evaluate(
            question=question,
            answer=answer,
            chunks=ranked_chunks,
            index_id=self.index_manager.index_id
        )

        # 7. Save evaluation shell
        feedback_id = self.feedback_store.save(evaluation)

        result = {
            "question": question,
            "answer": answer,
            "evaluation": evaluation,
            "sources": reasoning_payload["sources"],
            "feedback_id": feedback_id
        }

        # 8. Store in semantic cache
        self.semantic_cache.store(question, result)

        return result

    # ======================================================
    # FEEDBACK LOOP
    # ======================================================

    def submit_feedback(
        self,
        feedback_id: str,
        rating: int,
        comment: str = ""
    ) -> None:
        """
        rating:
          1  -> 👍
         -1  -> 👎
        """

        self.feedback_store.update_feedback(
            feedback_id=feedback_id,
            rating=rating,
            comment=comment
        )
