from enum import Enum


class ReasoningStrategy(Enum):
    """
    Стратегії побудови контексту та reasoning.
    """

    SIMPLE = "simple"          # пряме використання чанків
    QA = "qa"                  # питання-відповідь
    SUMMARIZATION = "summary"  # узагальнення (на майбутнє)
