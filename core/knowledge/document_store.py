"""
Document Store — source of truth for documents.
"""


class DocumentStore:
    def save(self, document: dict) -> None:
        raise NotImplementedError

    def load(self, source_id: str) -> dict:
        raise NotImplementedError
