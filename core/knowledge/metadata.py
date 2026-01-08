from typing import Dict, Any


class MetadataManager:
    """
    Утиліти для роботи з metadata документів та чанків.

    Metadata розглядається як:
    - розширюваний
    - необовʼязковий
    - некритичний для core-логіки
    """

    @staticmethod
    def ensure_metadata(entity: Dict) -> Dict:
        """
        Гарантує наявність поля metadata.
        """
        entity.setdefault("metadata", {})
        return entity

    @staticmethod
    def add_metadata(entity: Dict, metadata: Dict[str, Any]) -> Dict:
        """
        Додає або оновлює metadata.
        """
        MetadataManager.ensure_metadata(entity)
        entity["metadata"].update(metadata)
        return entity

    @staticmethod
    def get_metadata(entity: Dict) -> Dict[str, Any]:
        """
        Повертає metadata або порожній dict.
        """
        return entity.get("metadata", {})

    @staticmethod
    def get_metadata_value(
        entity: Dict,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Безпечне отримання значення з metadata.
        """
        return entity.get("metadata", {}).get(key, default)

    @staticmethod
    def remove_metadata_key(entity: Dict, key: str) -> Dict:
        """
        Видаляє ключ з metadata, якщо він існує.
        """
        if "metadata" in entity:
            entity["metadata"].pop(key, None)
        return entity
