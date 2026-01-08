from typing import Optional

from langchain_core.language_models.llms import LLM

# Ollama (local only)
try:
    from langchain_community.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# HuggingFace fallback
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline


class LLMClient:
    """
    Універсальний LLM-клієнт.

    - Ollama → для локального запуску
    - HuggingFace → fallback для Google Colab
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        backend: str = "auto"  # auto | ollama | hf
    ):
        self.model_name = model_name
        self.backend = backend

        self.llm: Optional[LLM] = None

        if backend in ("auto", "ollama") and OLLAMA_AVAILABLE:
            try:
                self.llm = Ollama(
                    model=model_name,
                    temperature=temperature,
                    num_predict=max_tokens
                )
                return
            except Exception:
                pass  # silently fallback

        # ---- HuggingFace fallback ----
        hf_pipeline = pipeline(
            "text-generation",
            model="google/flan-t5-base",
            max_new_tokens=max_tokens,
            temperature=temperature
        )

        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

    def generate(self, prompt: str) -> str:
        """
        Єдина точка генерації.
        """
        return self.llm.invoke(prompt)
