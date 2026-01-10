"""
RAG Service
===========

ЄДИНА роль:
- оркестрація RAG pipeline

НЕ:
- не bootstrap систему
- не знає, що існує на диску
- не керує state ініціалізацією
"""

from typing import Dict
import os
import json
# Chunking
from core.chunking.chunker import Chunker

# Ingestion
from core.ingestion.loader import DocumentLoader

# Knowledge
from core.knowledge.document_store import DocumentStore
from core.knowledge.chunk_store import ChunkStore

# Indexing
from core.indexing.index_manager import IndexManager
from core.indexing.index_router import SemanticIndexRouter

# Retrieval
from core.retrieval.retriever import Retriever
from core.retrieval.reranker import Reranker
from core.retrieval.policies import RetrievalPolicy

# Reasoning
from core.reasoning.agent import ReasoningAgent
from core.reasoning.strategies import ReasoningStrategy

# Generation
from core.generation.llm_client import LLMClient
from core.generation.prompts import PromptFactory

# Evaluation
from core.evaluation.evaluator import Evaluator

# Learning
from core.learning.state_maneger import StateManager
from core.learning.policies_update import StatePolicyUpdater

# System registry
from core.system.system_registry import SystemRegistry
# Feedback
from core.learning.feedback_store import FeedbackStore
# Cache
from core.cache.semantic_cache import SemanticCache
from core.cache.manager import CacheManager


class RAGService:
    """
    Thin orchestration layer.
    """

    def __init__(
        self,
        documents_path: str,
        chunks_path: str,
        indexes_path: str,
        state_path: str,
    ):
        # ---------------- System registry ----------------
        self.system_registry = SystemRegistry(
            documents_path=documents_path,
            chunks_path=chunks_path,
            indexes_path=indexes_path,
            state_path=state_path,
        )

        # ---------------- Core infrastructure ----------------
        self.document_store = DocumentStore(documents_path)
        self.chunk_store = ChunkStore(chunks_path)

        self.loader = DocumentLoader()
        self.chunker = Chunker()

        # ---------------- Indexing ----------------
        self.index_manager = IndexManager(
            indexes_path=indexes_path,
        )

        self.index_router = SemanticIndexRouter()

        # ---------------- State ----------------
        self.state_manager = StateManager(
            base_path=state_path
        )

        # 🔑 ONLINE LEARNING POLICY
        self.state_policy = StatePolicyUpdater(
            state_manager=self.state_manager
        )

        # ---------------- Retrieval ----------------
        self.retrieval_policy = RetrievalPolicy(
            top_k=5,
            rerank_k=3,
            use_query_rewrite=False,
        )

        self.retriever = Retriever(
            index_manager=self.index_manager,
            policy=self.retrieval_policy,
        )

        self.reranker = Reranker(
            embedder=self.index_manager.embedder,
            state_manager=self.state_manager,
        )

        # ---------------- Reasoning ----------------
        self.reasoning_agent = ReasoningAgent(
            strategy=ReasoningStrategy.QA
        )

        # ---------------- Generation ----------------
        self.llm_client = LLMClient(mode="colab")  # або "local"

        # ---------------- Evaluation ----------------
        self.evaluator = Evaluator(
            embedder=self.index_manager.embedder
        )

        # ---------------- Feedback ----------------
        self.feedback_store = FeedbackStore(
            base_path=os.path.join(state_path, "feedback")
        )

        # ---------------- Cache ----------------
        self.semantic_cache = SemanticCache(
            embedder=self.index_manager.embedder,
            similarity_threshold=0.9,
        )

        self.cache_manager = CacheManager(
            semantic_cache=self.semantic_cache,
            ttl=60 * 60,
        )

    # ======================================================
    # INGESTION
    # ======================================================

    def ingest_pdf(self, source: str) -> Dict:
        """
        PDF → document → chunks → index
        """
        document = self.loader.load_pdf(source)
        self.document_store.save(document)

        chunks = self.chunker.split(document)

        index_metadata = self.index_manager.build_index(chunks)

        for chunk in chunks:
            chunk.setdefault("metadata", {})
            chunk["metadata"].setdefault("index_ids", [])
            chunk["metadata"]["index_ids"].append(index_metadata["index_id"])
            chunk["metadata"]["embedding_model"] = index_metadata["embedding_model"]
            self.chunk_store.save(chunk)

        return {
            "document_id": document["document_id"],
            "chunks_count": len(chunks),
            "index": index_metadata,
        }

    # ======================================================
    # QUESTION ANSWERING + ONLINE LEARNING
    # ======================================================

    def ask(self, question: str) -> Dict:
        cached = self.semantic_cache.lookup(question)
        if cached is not None:
            return cached

        # --------- semantic routing ---------
        candidate_roles = self.index_router.route(question)
        index_roles = [
            {"index_role": r["index_role"], "router_score": r["router_score"]}
            for r in candidate_roles
        ]

        # --------- retrieval ---------
        chunk_ids = self.retriever.retrieve(question, index_roles)

        chunks = []
        for cid in chunk_ids:
            chunk = self.chunk_store.load(cid)
            if chunk:
                chunks.append(chunk)

        if not chunks:
            return {
                "question": question,
                "answer": "I do not know.",
                "evaluation": None,
                "sources": [],
                "feedback_id": None,
            }

        # --------- rerank ---------
        ranked_chunks = self.reranker.rerank(
            query=question,
            chunks=chunks,
            top_k=self.retrieval_policy.rerank_k,
        )

        # --------- reasoning ---------
        reasoning_payload = self.reasoning_agent.prepare(
            question=question,
            chunks=ranked_chunks,
        )

        prompt = PromptFactory.qa_prompt(reasoning_payload)
        answer = self.llm_client.generate(prompt)

        # --------- evaluation ---------
        evaluation = self.evaluator.evaluate(
            question=question,
            answer=answer,
            chunks=ranked_chunks,
            index_ids=[r["index_role"] for r in index_roles],
        )

        # 🔁 ONLINE LEARNING (AUTOMATIC)
        self.state_policy.apply(
            evaluation=evaluation,
            feedback=None
        )

        # 📝 CREATE FEEDBACK SHELL
        feedback_id = self.feedback_store.create(
            evaluation_id=evaluation["evaluation_id"]
        )

        result = {
            "question": question,
            "answer": answer,
            "evaluation": evaluation,
            "sources": reasoning_payload["sources"],
            "feedback_id": feedback_id,
        }

        self.semantic_cache.store(question, result)
        return result

    def submit_feedback(
            self,
            feedback_id: str,
            rating: int,
            comment: str = ""
    ) -> None:
        """
        Приймає user feedback і застосовує його ОДИН раз.
        """

        # 1. update feedback record
        self.feedback_store.update(
            feedback_id=feedback_id,
            rating=rating,
            comment=comment
        )

        feedback = self.feedback_store.load(feedback_id)
        if not feedback or feedback.get("applied"):
            return

        # 2. load evaluation
        evaluation_id = feedback["evaluation_id"]
        evaluation_path = os.path.join(
            self.state_manager.base_path,
            "evaluations",
            f"{evaluation_id}.json"
        )

        if not os.path.exists(evaluation_path):
            return

        with open(evaluation_path, "r", encoding="utf-8") as f:
            evaluation = json.load(f)

        # 3. apply human signal
        self.state_policy.apply(
            evaluation=evaluation,
            feedback=feedback
        )

        # 4. mark feedback as applied
        self.feedback_store.mark_applied(feedback_id)