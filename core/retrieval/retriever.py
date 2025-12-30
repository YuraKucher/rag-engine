"""
Retriever — політика пошуку знань.

Відповідальність:
- прийняти запит
- вибрати індекс
- визначити k
- застосувати rerank
- повернути релевантні чанки
"""

from typing import List, Optional


class Retriever:
    def retrieve(self, query: str, context: Optional[dict] = None) -> List[dict]:
        """
        Retrieve relevant chunks for a given query.

        :param query: user question
        :param context: optional conversation or system context
        :return: list of chunks (dicts following chunk.schema.json)
        """
        raise NotImplementedError
