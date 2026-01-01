from typing import Dict, List
from .base_chunker import BaseChunker


class TextChunker(BaseChunker):
    """
    Базовий текстовий chunker.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Dict) -> List[Dict]:
        text = document["content"]
        document_id = document["document_id"]

        chunks = []
        start = 0
        index = 0

        while start < len(text):
            end = start + self.chunk_size

            chunk = {
                "chunk_id": f"{document_id}_{index}",
                "document_id": document_id,
                "content": text[start:end],
                "position": {
                    "start": start,
                    "end": min(end, len(text))
                },
                "metadata": {
                    "chunk_index": index
                }
            }

            chunks.append(chunk)

            index += 1
            start = end - self.overlap

        return chunks
