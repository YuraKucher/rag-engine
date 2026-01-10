"""
DocumentLoader
==============

Єдиний loader для документів.

Відповідає за:
- завантаження документа з джерела
- нормалізацію у внутрішній document-формат

НЕ:
- не реєструє документ
- не думає про індекси
- не думає про state
"""

from typing import Dict
from datetime import datetime
import uuid
import hashlib

from langchain_community.document_loaders import PyPDFLoader


class DocumentLoader:
    def load_pdf(self, source: str) -> Dict:
        loader = PyPDFLoader(source)
        pages = loader.load()

        full_text = "\n".join(page.page_content for page in pages)

        return {
            "document_id": str(uuid.uuid4()),
            "source": source,
            "hash": self._hash(full_text.encode("utf-8")),
            "metadata": {
                "title": source.split("/")[-1],
                "pages": len(pages),
                "loader": "pdf",
            },
            "content": full_text,
            "created_at": datetime.utcnow().isoformat(),
        }

    @staticmethod
    def _hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()
