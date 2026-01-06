from typing import List, Dict


class Chunker:
    """
    Відповідає за розбиття Document на Chunk-и.
    Працює зі структурами document.schema.json та chunk.schema.json
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document: Dict, progress=None) -> List[Dict]:
        text = document["content"]
        document_id = document["document_id"]

        chunks = []
        start = 0
        index = 0

        approx_total = max(1, len(text) // self.chunk_size)

        while start < len(text):
            end = start + self.chunk_size
            content = text[start:end]

            chunk = {
                "chunk_id": f"{document_id}_{index}",
                "document_id": document_id,
                "content": content,
                "position": {
                    "start": start,
                    "end": end
                },
                "metadata": {
                    "chunk_index": index
                }
            }

            chunks.append(chunk)

            index += 1
            start = end - self.overlap

            if progress:
                progress.step()

        return chunks


# ✅ ОЦЕ ДОДАНО
def chunk_document(document: Dict, *, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """
    Функціональний фасад для chunking.
    Потрібен для сервісів, щоб не створювати Chunker вручну.
    """
    chunker = Chunker(chunk_size=chunk_size, overlap=overlap)
    return chunker.split(document)
