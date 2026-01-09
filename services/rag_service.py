"""
RAG Service
"""

# ======================================================
# IMPORTS
# ======================================================
from typing import Dict, List

# Cache
from core.cache.semantic_cache import SemanticCache
from core.cache.manager import CacheManager

# Chunking
from core.chunking.chunker import Chunker

# Evaluation
from core.evaluation.evaluator import Evaluator

# Generation
from core.generation.llm_client import LLMClient
from core.generation.prompts import PromptFactory

# Indexing
from core.indexing.index_manager import IndexManager
from core.indexing.index_registry import IndexRegistry
from core.indexing.index_router import SemanticIndexRouter

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
from core.retrieval.retriever import Retriever
from core.retrieval.reranker import Reranker


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

        # ---------------- Indexing paths ----------------
        indexes_path = chunks_path.replace("chunks", "indexes")

        # ---------------- Index Manager ----------------
        self.index_manager = IndexManager(
            embedding_model=embedding_model,
            indexes_path=indexes_path
        )
        embedder = self.index_manager.embedder

        # ---------------- Index Registry ----------------
        self.index_registry = IndexRegistry(indexes_path)

        # ---------------- Semantic Router ----------------
        self.index_router = SemanticIndexRouter(
            embedder=self.index_manager.embedder,
            index_registry=self.index_registry,
            similarity_threshold=0.35,
            top_k=3
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

        self.reranker = Reranker(
            embedder=self.index_manager.embedder
        )

        # ---------------- Reasoning ----------------
        self.reasoning_agent = ReasoningAgent(
            strategy=ReasoningStrategy.QA
        )

        # ---------------- Generation ----------------
        self.llm_client = LLMClient(
            model_name=llm_model,
            backend="hf"
        )

        # ---------------- Evaluation ----------------
        self.evaluator = Evaluator(embedder=embedder)

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

        # ---------------- Restore registry ----------------
        self._restore_indexes()

    # ======================================================
    # INDEX RESTORE
    # ======================================================

    def _restore_indexes(self) -> None:
        """
        Відновлює registry з диску.
        FAISS індекси вантажаться тільки при ask().
        """
        self.index_registry.reload()

    # ======================================================
    # INGESTION PIPELINE
    # ======================================================

    def ingest_document(self, source: str, file_type: str) -> Dict:
        loader = LoaderRegistry.get_loader(file_type)
        document = loader.load(source)

        self.document_store.save(document)

        chunks = self.chunker.split(document)

        index_metadata = self.index_manager.build_index(chunks)
        self.index_registry.register_index(index_metadata)

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
        cached = self.semantic_cache.lookup(question)
        if cached is not None:
            return cached

        # --------- Semantic routing ---------
        candidate_indexes = self.index_router.route(question)

        all_chunks: List[Dict] = []

        # --------- Retrieval per index ---------
        for meta in candidate_indexes:
            index_id = meta["index_id"]

            self.index_manager.load_index(index_id)

            chunk_ids = self.retriever.retrieve(question)
            for cid in chunk_ids:
                chunk = self.chunk_store.load(cid)
                if chunk:
                    all_chunks.append(chunk)

        if not all_chunks:
            return {
                "question": question,
                "answer": "I do not know.",
                "evaluation": None,
                "sources": [],
                "feedback_id": None
            }

        # --------- Cross-index reranking ---------
        ranked_chunks = self.reranker.rerank(
            query=question,
            chunks=all_chunks,
            top_k=self.retrieval_policy.rerank_k
        )

        # --------- Reasoning ---------
        reasoning_payload = self.reasoning_agent.prepare(
            question=question,
            chunks=ranked_chunks
        )

        # --------- Prompt ---------
        prompt = PromptFactory.qa_prompt(reasoning_payload)

        # --------- Generation ---------
        answer = self.llm_client.generate(prompt)

        # --------- Evaluation ---------
        evaluation = self.evaluator.evaluate(
            question=question,
            answer=answer,
            chunks=ranked_chunks,
            index_ids=[m["index_id"] for m in candidate_indexes]
        )

        feedback_id = self.feedback_store.save(evaluation)

        result = {
            "question": question,
            "answer": answer,
            "evaluation": evaluation,
            "sources": reasoning_payload["sources"],
            "feedback_id": feedback_id
        }

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
        self.feedback_store.update_feedback(
            feedback_id=feedback_id,
            rating=rating,
            comment=comment
        )
