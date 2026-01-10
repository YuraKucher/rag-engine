"""
StateManager
============

Єдиний компонент, який відповідає за:
- створення state-файлів
- завантаження state
- збереження state
- читання state для rerank

НЕ:
- не знає, що існує в системі
- не приймає рішень
- не навчає
- не знає про retrieval / rerank логіку
"""

import os
import json
from datetime import datetime
from typing import Dict, List


# --------------------------------------------------
# FILE NAMES
# --------------------------------------------------

DOCUMENT_STATE_FILE = "document_state.json"
CHUNK_STATE_FILE = "chunk_state.json"
INDEX_STATE_FILE = "index_state.json"


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _load_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# --------------------------------------------------
# STATE MANAGER
# --------------------------------------------------

class StateManager:
    """
    Persistent, lazy-initialized learning state.
    """

    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

        self.document_state_path = os.path.join(base_path, DOCUMENT_STATE_FILE)
        self.chunk_state_path = os.path.join(base_path, CHUNK_STATE_FILE)
        self.index_state_path = os.path.join(base_path, INDEX_STATE_FILE)

        self.document_state: Dict = {}
        self.chunk_state: Dict = {}
        self.index_state: Dict = {}

        self.load_all()

    # --------------------------------------------------
    # LOAD / SAVE
    # --------------------------------------------------

    def load_all(self) -> None:
        self.document_state = _load_json(self.document_state_path)
        self.chunk_state = _load_json(self.chunk_state_path)
        self.index_state = _load_json(self.index_state_path)

    def save_all(self) -> None:
        _save_json(self.document_state_path, self.document_state)
        _save_json(self.chunk_state_path, self.chunk_state)
        _save_json(self.index_state_path, self.index_state)

    # --------------------------------------------------
    # READ API (reranker)
    # --------------------------------------------------

    def get_document_weight(self, doc_id: str) -> float:
        doc = self.document_state.setdefault(doc_id, self._init_document_state())
        return doc["weight"]

    def get_chunk_weight(self, chunk_id: str) -> float:
        chunk = self.chunk_state.setdefault(chunk_id, self._init_chunk_state())
        return chunk["weight"]

    def get_index_prior(self, index_id: str) -> float:
        index = self.index_state.setdefault(index_id, self._init_index_state())
        return index["prior"]

    # --------------------------------------------------
    # MUTATION API (feedback updater ONLY)
    # --------------------------------------------------

    def update_document(self, doc_id: str, relevance: float, answerability: float) -> None:
        doc = self.document_state.setdefault(doc_id, self._init_document_state())

        doc["relevance_score"] = self._running_avg(
            doc["relevance_score"], doc["relevance_count"], relevance
        )
        doc["relevance_count"] += 1

        doc["answerability_score"] = self._running_avg(
            doc["answerability_score"], doc["answerability_count"], answerability
        )
        doc["answerability_count"] += 1

        doc["usage_count"] += 1
        doc["weight"] = self._compute_document_weight(doc)
        doc["last_updated"] = _now()

    def update_chunk(self, chunk_id: str, relevance: float, groundedness: float) -> None:
        chunk = self.chunk_state.setdefault(chunk_id, self._init_chunk_state())

        chunk["relevance_score"] = self._running_avg(
            chunk["relevance_score"], chunk["relevance_count"], relevance
        )
        chunk["relevance_count"] += 1

        chunk["groundedness_score"] = self._running_avg(
            chunk["groundedness_score"], chunk["groundedness_count"], groundedness
        )
        chunk["groundedness_count"] += 1

        chunk["usage_count"] += 1
        chunk["weight"] = self._compute_chunk_weight(chunk)
        chunk["last_updated"] = _now()

    def update_index(self, index_ids: List[str], relevance: float, groundedness: float) -> None:
        """
        Застосовує зміни для мультиіндексів.
        Зберігає avg_relevance та avg_groundedness для кожного індексу.
        """
        for index_id in index_ids:
            index = self.index_state.setdefault(index_id, self._init_index_state())

            index["avg_relevance"] = self._running_avg(
                index["avg_relevance"], index["usage_count"], relevance
            )
            index["avg_groundedness"] = self._running_avg(
                index["avg_groundedness"], index["usage_count"], groundedness
            )

            index["usage_count"] += 1
            index["prior"] = self._compute_index_prior(index)
            index["last_updated"] = _now()

    # --------------------------------------------------
    # INTERNALS
    # --------------------------------------------------

    @staticmethod
    def _running_avg(old: float, count: int, new: float) -> float:
        return (old * count + new) / (count + 1)

    @staticmethod
    def _compute_document_weight(doc: Dict) -> float:
        return 0.5 * doc["relevance_score"] + 0.5 * doc["answerability_score"]

    @staticmethod
    def _compute_chunk_weight(chunk: Dict) -> float:
        return 0.5 * chunk["relevance_score"] + 0.5 * chunk["groundedness_score"]

    @staticmethod
    def _compute_index_prior(index: Dict) -> float:
        return 0.5 * index["avg_relevance"] + 0.5 * index["avg_groundedness"]

    @staticmethod
    def _init_document_state() -> Dict:
        return {
            "weight": 1.0,
            "relevance_score": 0.0,
            "answerability_score": 0.0,
            "relevance_count": 0,
            "answerability_count": 0,
            "usage_count": 0,
            "last_updated": _now(),
        }

    @staticmethod
    def _init_chunk_state() -> Dict:
        return {
            "weight": 1.0,
            "relevance_score": 0.0,
            "groundedness_score": 0.0,
            "relevance_count": 0,
            "groundedness_count": 0,
            "usage_count": 0,
            "last_updated": _now(),
        }

    @staticmethod
    def _init_index_state() -> Dict:
        return {
            "prior": 1.0,
            "avg_relevance": 0.0,
            "avg_groundedness": 0.0,
            "usage_count": 0,
            "last_updated": _now(),
        }
