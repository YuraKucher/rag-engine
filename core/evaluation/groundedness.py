from typing import List, Dict


class GroundednessEvaluator:
    """
    Перевіряє, наскільки відповідь опирається на контекст.
    """

    def score(self, answer: str, chunks: List[Dict]) -> float:
        """
        Дуже проста евристика:
        частка chunk-ів, слова з яких зустрічаються у відповіді.
        """

        if not chunks:
            return 0.0

        answer_text = answer.lower()
        hits = 0

        for chunk in chunks:
            content = chunk["content"].lower()
            if any(word in answer_text for word in content.split()[:20]):
                hits += 1

        score = hits / len(chunks)
        return round(score, 3)
