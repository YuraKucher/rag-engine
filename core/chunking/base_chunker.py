from abc import ABC, abstractmethod
from typing import Dict, List


class BaseChunker(ABC):
    """
    Абстрактний chunker.

    Відповідає ТІЛЬКИ за розбиття документа на чанки.
    """

    @abstractmethod
    def chunk(self, document: Dict) -> List[Dict]:
        """
        Приймає document (document.schema.json)
        Повертає список chunk (chunk.schema.json)
        """
        raise NotImplementedError
