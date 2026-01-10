"""
Chunker
=======

Єдиний компонент для чанкінгу документів.

Відповідає за:
- розбиття document -> chunks
- формування стабільної структури чанка
- ініціалізацію metadata

НЕ:
- не знає про індекси
- не знає про state
- не знає про retrieval
"""

import uuid
from datetime import datetime
from typing import Dict, List


class Chunker:
    """
    Simple text chunker.
    """

    def __init__(
        self,
        max_length: int = 500,
        overlap: int = 50
    ):
        if overlap >= max_length:
            raise ValueError("overlap must be smaller than max_length")

        self.max_length = max_length
        self.overlap = overlap

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def split(self, document: Dict) -> List[Dict]:
        """
        Розбиває документ на чанки.
        """
        text = document.get("content", "")
        if not text:
            return []

        chunks: List[Dict] = []
        start = 0
        position = 0

        while start < len(text):
            end = min(start + self.max_length, len(text))
            chunk_text = text[start:end]

            chunk = self._create_chunk(
                document_id=document["document_id"],
                content=chunk_text,
                position=position
            )
            chunks.append(chunk)

            start = end - self.overlap
            position += 1

        return chunks

    # --------------------------------------------------
    # INTERNALS
    # --------------------------------------------------

    def _create_chunk(
        self,
        document_id: str,
        content: str,
        position: int
    ) -> Dict:
        """
        Формує структуру чанка (schema-compatible).
        """
        return {
            "chunk_id": str(uuid.uuid4()),
            "document_id": document_id,
            "content": content,
            "metadata": {
                # 🔑 для мультиіндексації
                "index_ids": [],

                # позиція чанка в документі
                "position": position
            },
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
