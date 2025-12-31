from abc import ABC, abstractmethod
from typing import Dict


class BaseLoader(ABC):
    """
    Базовий клас для всіх завантажувачів документів.
    Повертає обʼєкт, узгоджений з document.schema.json
    """

    @abstractmethod
    def load(self, source: str) -> Dict:
        """
        source: шлях до файлу або інше джерело
        return: Document (dict)
        """
        pass
