from langchain_core.language_models.llms import LLM
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline


class LLMClient:
    """
    Універсальний LLM-клієнт.

    backend:
      - "hf"      → HuggingFace (Colab / cloud)
      - "ollama"  → Ollama (local only)
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        backend: str = "hf"
    ):
        self.model_name = model_name
        self.backend = backend

        # ---------------- OLLAMA (LOCAL) ----------------
        if backend == "ollama":
            from langchain_community.llms import Ollama

            self.llm: LLM = Ollama(
                model=model_name,
                temperature=temperature,
                num_predict=max_tokens
            )

        # ---------------- HUGGINGFACE (COLAB) ----------------
        elif backend == "hf":
            # FLAN-T5 → text2text-generation, NOT text-generation
            hf_pipeline = pipeline(
                task="text2text-generation",
                model="google/flan-t5-base",
                max_new_tokens=max_tokens,
                do_sample=False,          # ← заміна temperature=0
                truncation=True
            )

            self.llm: LLM = HuggingFacePipeline(
                pipeline=hf_pipeline
            )

        else:
            raise ValueError(f"Unsupported LLM backend: {backend}")

    # --------------------------------------------------

    def generate(self, prompt: str) -> str:
        """
        Єдина точка генерації.
        """
        return self.llm.invoke(prompt)
