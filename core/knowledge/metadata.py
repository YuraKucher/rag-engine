from typing import Dict


class MetadataManager:
    """
    Утиліти для роботи з metadata документів та чанків.
    """

    @staticmethod
    def add_metadata(entity: Dict, metadata: Dict) -> Dict:
        entity_metadata = entity.get("metadata", {})
        entity_metadata.update(metadata)
        entity["metadata"] = entity_metadata
        return entity

    @staticmethod
    def get_metadata(entity: Dict) -> Dict:
        return entity.get("metadata", {})

    @staticmethod
    def remove_metadata_key(entity: Dict, key: str) -> Dict:
        if "metadata" in entity and key in entity["metadata"]:
            del entity["metadata"][key]
        return entity
