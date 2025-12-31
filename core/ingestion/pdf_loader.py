from typing import Dict
from datetime import datetime
import uuid

from langchain.document_loaders import PyPDFLoader

from .base_loader import BaseLoader


class PDFLoader(BaseLoader):
    """
    Loader для PDF-документів.
    """

    def load(self, source: str) -> Dict:
        loader = PyPDFLoader(source)
        pages = loader.load()

        full_text = "\n".join(page.page_content for page in pages)

        document = {
            "document_id": str(uuid.uuid4()),
            "source": source,
            "metadata": {
                "title": source.split("/")[-1],
                "pages": len(pages),
                "loader": "pdf"
            },
            "content": full_text,
            "created_at": datetime.utcnow().isoformat()
        }

        return document
