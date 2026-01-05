from typing import Dict

from core.ingestion.registry import IngestionRegistry
from core.ingestion.base_loader import compute_document_hash
from core.chunking.chunker import chunk_document
from services.indexing_service import IndexingService


class IngestionService:
    def __init__(
        self,
        indexing_service: IndexingService,
        chunk_config: Dict
    ):
        self.indexing_service = indexing_service
        self.chunk_config = chunk_config
        self.registry = IngestionRegistry()

    def ingest(
        self,
        filename: str,
        content: bytes,
        mime: str
    ) -> Dict:
        doc_hash = compute_document_hash(content)

        if self.registry.exists(doc_hash):
            return {
                "status": "cached",
                "document_hash": doc_hash,
                "index": self.registry.get(doc_hash)["index"]
            }

        document = {
            "document_id": doc_hash,
            "content": content.decode("utf-8")
        }

        chunks = chunk_document(
            document,
            **self.chunk_config
        )

        index_meta = self.indexing_service.index_chunks(chunks)

        self.registry.register({
            "hash": doc_hash,
            "filename": filename,
            "mime": mime,
            "chunks": [c["chunk_id"] for c in chunks],
            "index": index_meta
        })

        return {
            "status": "processed",
            "document_hash": doc_hash,
            "index": index_meta
        }
