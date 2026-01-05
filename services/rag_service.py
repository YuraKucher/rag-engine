"""
RAG Service
===========

Єдина точка входу в RAG-систему.

ВАЖЛИВО:
- тут НІЯКОЇ бізнес-логіки
- тут НІЯКИХ евристик
- лише оркестрація шарів
"""

from typing import Dict

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
from core.knowledge.chunk_store import ChunkStore
# Knowledge
from core.knowledge.document_store import DocumentStore
# Learning / Feedback
from core.learning.feedback_store import FeedbackStore
# Reasoning
from core.reasoning.agent import ReasoningAgent
from core.reasoning.strategies import ReasoningStrategy
from core.retrieval.policies import RetrievalPolicy
from core.retrieval.reranker import Reranker
# Retrieval
from core.retrieval.retriever import Retriever


# ======================================================
# IMPORTS
# ======================================================


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
        self.index_manager = IndexManager(
            embedding_model=embedding_model
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

        # 3. Chunking (ОКРЕМИЙ ШАР)
        chunks = self.chunker.split(document)

        # 4. Save chunks
        for chunk in chunks:
            self.chunk_store.save(chunk)

        # 5. Build index
        index_metadata = self.index_manager.build_index(chunks)

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
        Повний RAG pipeline + evaluation + feedback hook.
        """

        # 1. Retrieval
        chunk_ids = self.retriever.retrieve(question)

        chunks = [
            self.chunk_store.load(chunk_id)
            for chunk_id in chunk_ids
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
            chunks=ranked_chunks
        )

        # 7. Save feedback shell (без 👍/👎)
        feedback_id = self.feedback_store.save(evaluation)

        return {
            "question": question,
            "answer": answer,
            "evaluation": evaluation,
            "sources": reasoning_payload["sources"],
            "feedback_id": feedback_id
        }

    # ======================================================
    # FEEDBACK LOOP (UI calls this)
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
