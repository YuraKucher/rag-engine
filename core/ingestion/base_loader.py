from typing import Dict
from abc import ABC, abstractmethod
import hashlib


class BaseLoader(ABC):
    """
    Абстрактний базовий loader.
    Усі loader-и повинні реалізувати load().
    """

    @abstractmethod
    def load(self, source: str) -> Dict:
        pass


def compute_document_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
