class QueryRewriter:
    """
    Компонент переписування запитів.
    За замовчуванням — identity.
    """

    def rewrite(self, query: str) -> str:
        """
        Явна точка розширення.
        """
        return query.strip()
