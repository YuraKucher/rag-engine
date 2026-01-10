# core/generation/llm_client.py

from transformers import pipeline
from config.settings import settings


class LLMClient:
    """
    Config-driven LLM client.

    НЕ:
    - не приймає model_name напряму
    - не має hardcoded моделей
    """

    def __init__(self, mode: str):
        """
        mode: "local" | "colab"
        """
        cfg = settings.models["llm"].get(mode)
        if not cfg:
            raise ValueError(f"LLM config for mode '{mode}' not found")

        self.backend = cfg["backend"]
        self.temperature = cfg.get("temperature", 0.0)
        self.max_tokens = cfg.get("max_tokens", 256)

        if self.backend == "hf":
            self._init_hf(cfg)
        elif self.backend == "ollama":
            self._init_ollama(cfg)
        else:
            raise ValueError(f"Unsupported LLM backend: {self.backend}")

    # --------------------------------------------------
    # BACKENDS
    # --------------------------------------------------

    def _init_hf(self, cfg: dict):
        model_name = cfg["model"]

        self.pipeline = pipeline(
            task="text2text-generation",
            model=model_name,
            max_new_tokens=self.max_tokens,
            do_sample=self.temperature > 0.0,
            temperature=self.temperature if self.temperature > 0 else None,
        )

    def _init_ollama(self, cfg: dict):
        # для локального dev (залишаємо як є, мінімально)
        from langchain.llms import Ollama

        self.llm = Ollama(
            model=cfg["model"],
            temperature=self.temperature
        )

    # --------------------------------------------------
    # API
    # --------------------------------------------------

    def generate(self, prompt: str) -> str:
        if self.backend == "hf":
            out = self.pipeline(prompt)[0]["generated_text"]
            return out.strip()

        if self.backend == "ollama":
            return self.llm(prompt)

        raise RuntimeError("LLM backend not initialized")
