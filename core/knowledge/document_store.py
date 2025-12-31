import json
import os
from typing import Dict, Optional


class DocumentStore:
    """
    Сховище документів.
    Працює з обʼєктами document.schema.json
    """

    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def save(self, document: Dict) -> None:
        document_id = document["document_id"]
        path = os.path.join(self.base_path, f"{document_id}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(document, f, ensure_ascii=False, indent=2)

    def load(self, document_id: str) -> Optional[Dict]:
        path = os.path.join(self.base_path, f"{document_id}.json")
        if not os.path.exists(path):
            return None

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_documents(self) -> list[str]:
        return [
            filename.replace(".json", "")
            for filename in os.listdir(self.base_path)
            if filename.endswith(".json")
        ]

