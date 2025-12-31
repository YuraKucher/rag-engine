from langchain.llms.base import LLM
from langchain.llms import Ollama


class LLMClient:
    """
    Клієнт для роботи з LLM.
    Єдина точка генерації тексту.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 512
    ):
        self.model_name = model_name
        self.llm: LLM = Ollama(
            model=model_name,
            temperature=temperature,
            num_predict=max_tokens
        )

    def generate(self, prompt: str) -> str:
        """
        prompt: готовий prompt string
        return: відповідь LLM
        """
        return self.llm(prompt)
