from typing import List, Tuple
from pathlib import Path
import faiss
import numpy as np


class FaissIndex:
    """
    Обгортка над FAISS-індексом.
    Не знає нічого про чанки, лише вектори.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)

    def add(self, vectors: List[List[float]]) -> None:
        np_vectors = np.array(vectors).astype("float32")
        self.index.add(np_vectors)

    def search(
        self, query_vector: List[float], k: int
    ) -> Tuple[List[int], List[float]]:
        query = np.array([query_vector]).astype("float32")
        distances, indices = self.index.search(query, k)
        return indices[0].tolist(), distances[0].tolist()

    def size(self) -> int:
        return self.index.ntotal

    # --- NEW ---
    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: str) -> "FaissIndex":
        index = faiss.read_index(str(path))
        obj = cls(index.d)
        obj.index = index
        return obj
