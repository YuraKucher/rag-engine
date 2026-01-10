# core/indexing/embedder.py

from typing import List
from sentence_transformers import SentenceTransformer

from config.settings import settings


class Embedder:
    """
    Config-driven embedding engine.

    НЕ:
    - не приймає model_name напряму
    - не знає, де він запускається

    ВСЕ береться з models.yaml
    """

    def __init__(self):
        cfg = settings.models["embeddings"]

        backend = cfg.get("backend")
        if backend != "sentence_transformers":
            raise ValueError(f"Unsupported embedding backend: {backend}")

        model_name = cfg["model"]
        device = cfg.get("device", "cpu")
        self.normalize = cfg.get("normalize", True)

        self.model = SentenceTransformer(
            model_name,
            device=device
        )
    # --------------------------------------------------
    # API
    # --------------------------------------------------

    def embed(self, text: str) -> List[float]:
        vec = self.model.encode(
            text,
            normalize_embeddings=self.normalize
        )
        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=self.normalize
        )
        return [v.tolist() for v in vectors]
