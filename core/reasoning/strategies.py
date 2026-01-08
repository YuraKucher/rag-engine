from enum import Enum


class ReasoningStrategy(Enum):
    """
    Стратегії побудови контексту та reasoning.

    ВАЖЛИВО:
    - кожна стратегія або підтримана, або явно заборонена
    """

    SIMPLE = "simple"   # пряме використання чанків
    QA = "qa"           # grounded question answering
