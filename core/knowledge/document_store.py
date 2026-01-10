import json
import os
from typing import Dict, Optional, List


class DocumentStore:
    """
    Сховище документів.
    Source of truth для document-level knowledge.
    Працює з document.schema.json
    """

    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    # --------------------------------------------------
    # BASIC IO
    # --------------------------------------------------

    def save(self, document: Dict) -> None:
        """
        Зберігає документ як окремий JSON-файл.
        """

        document_id = document["document_id"]

        # 🔒 гарантуємо стабільну metadata (мультиіндекс-safe)
        document.setdefault("metadata", {})
        document["metadata"].setdefault("index_ids", [])

        path = os.path.join(self.base_path, f"{document_id}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(document, f, ensure_ascii=False, indent=2)

    def load(self, document_id: str) -> Optional[Dict]:
        """
        Завантажує документ за ID.
        """

        path = os.path.join(self.base_path, f"{document_id}.json")
        if not os.path.exists(path):
            return None

        with open(path, "r", encoding="utf-8") as f:
            document = json.load(f)

        # 🔒 backward compatibility
        document.setdefault("metadata", {})
        document["metadata"].setdefault("index_ids", [])

        return document

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------

    def list_documents(self) -> List[str]:
        """
        Повертає список всіх document_id.
        """
        return [
            filename.replace(".json", "")
            for filename in os.listdir(self.base_path)
            if filename.endswith(".json")
        ]
