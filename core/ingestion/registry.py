from typing import Dict, Type
from .base_loader import BaseLoader
from .pdf_loader import PDFLoader


class LoaderRegistry:
    """
    Реєстр доступних loader-ів.
    """

    _loaders: Dict[str, Type[BaseLoader]] = {
        "pdf": PDFLoader
    }

    @classmethod
    def get_loader(cls, file_type: str) -> BaseLoader:
        file_type = file_type.lower()
        if file_type not in cls._loaders:
            raise ValueError(f"No loader registered for type: {file_type}")
        return cls._loaders[file_type]()

    @classmethod
    def register_loader(cls, file_type: str, loader: Type[BaseLoader]) -> None:
        cls._loaders[file_type.lower()] = loader
