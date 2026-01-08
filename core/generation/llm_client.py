from typing import Optional

from langchain_core.language_models.llms import LLM
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline


class LLMClient:
    """
    LLM client with explicit backend selection.

    backend:
      - "hf"      → HuggingFace (Colab, cloud)
      - "ollama"  → Ollama (local only)
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        backend: str = "hf"   # ← ВАЖЛИВО
    ):
        self.model_name = model_name
        self.backend = backend

        if backend == "ollama":
            from langchain_community.llms import Ollama

            self.llm: LLM = Ollama(
                model=model_name,
                temperature=temperature,
                num_predict=max_tokens
            )

        elif backend == "hf":
            hf_pipeline = pipeline(
                "text-generation",
                model="google/flan-t5-base",
                max_new_tokens=max_tokens,
                temperature=temperature
            )

            self.llm: LLM = HuggingFacePipeline(
                pipeline=hf_pipeline
            )

        else:
            raise ValueError(f"Unsupported LLM backend: {backend}")

    def generate(self, prompt: str) -> str:
        return self.llm.invoke(prompt)
