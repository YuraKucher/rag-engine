from typing import List, Tuple
from pathlib import Path
import faiss
import numpy as np


class FaissIndex:
    """
    FAISS index wrapper (cosine similarity via L2 normalization).
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index: faiss.Index = faiss.IndexFlatL2(dimension)

    # --------------------------------------------------
    # ADD
    # --------------------------------------------------

    def add(self, vectors: List[List[float]]) -> None:
        if not vectors:
            return

        x = np.asarray(vectors, dtype="float32")

        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected (*, {self.dimension}), got {x.shape}"
            )

        # cosine similarity via normalization
        faiss.normalize_L2(x)

        # ✅ high-level API (correct at runtime)
        self.index.add(x)  # type: ignore[arg-type]

    # --------------------------------------------------
    # SEARCH
    # --------------------------------------------------

    def search(
        self,
        query_vector: List[float],
        k: int,
    ) -> Tuple[List[int], List[float]]:
        if self.index.ntotal == 0:
            return [], []

        q = np.asarray([query_vector], dtype="float32")

        if q.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dimension}, got {q.shape[1]}"
            )

        faiss.normalize_L2(q)

        # ✅ high-level API
        distances, indices = self.index.search(q, k)  # type: ignore[arg-type]

        return indices[0].tolist(), distances[0].tolist()

    # --------------------------------------------------
    # META
    # --------------------------------------------------

    def size(self) -> int:
        return int(self.index.ntotal)

    # --------------------------------------------------
    # PERSISTENCE
    # --------------------------------------------------

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
