"""
SystemRegistry
==============

Єдине джерело правди про те,
ЯКІ сутності існують у системі зараз.

Відповідає ТІЛЬКИ за discover / existence.
"""

import os
import json
from typing import Dict, List


class SystemRegistry:
    """
    Registry всіх knowledge- та infra-сутностей системи.
    """

    def __init__(
        self,
        documents_path: str,
        chunks_path: str,
        indexes_path: str,
        state_path: str,
    ):
        self.documents_path = documents_path
        self.chunks_path = chunks_path
        self.indexes_path = indexes_path
        self.state_path = state_path

    # --------------------------------------------------
    # DOCUMENTS
    # --------------------------------------------------

    def list_documents(self) -> List[str]:
        return self._list_ids(
            base_path=self.documents_path,
            id_key="document_id"
        )

    # --------------------------------------------------
    # CHUNKS
    # --------------------------------------------------

    def list_chunks(self) -> List[str]:
        return self._list_ids(
            base_path=self.chunks_path,
            id_key="chunk_id"
        )

    # --------------------------------------------------
    # INDEXES
    # --------------------------------------------------

    def list_indexes(self) -> List[str]:
        return self._list_ids(
            base_path=self.indexes_path,
            id_key="index_id"
        )

    # --------------------------------------------------
    # STATE
    # --------------------------------------------------

    def list_state_files(self) -> Dict[str, bool]:
        return {
            "document_state": os.path.exists(os.path.join(self.state_path, "document_state.json")),
            "chunk_state": os.path.exists(os.path.join(self.state_path, "chunk_state.json")),
            "index_state": os.path.exists(os.path.join(self.state_path, "index_state.json")),
        }

    # --------------------------------------------------
    # INTERNAL
    # --------------------------------------------------

    @staticmethod
    def _list_ids(base_path: str, id_key: str) -> List[str]:
        ids = []

        if not os.path.exists(base_path):
            return ids

        for fname in os.listdir(base_path):
            if not fname.endswith(".json"):
                continue

            try:
                with open(os.path.join(base_path, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                if id_key in data:
                    ids.append(data[id_key])
            except Exception:
                # registry НІКОЛИ не падає через биті файли
                continue

        return ids
