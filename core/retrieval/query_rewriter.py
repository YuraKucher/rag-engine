class QueryRewriter:
    """
    Компонент переписування запитів.
    За замовчуванням — identity.
    """

    def rewrite(self, query: str) -> str:
        # Базова реалізація — без змін
        # Розширення можливе пізніше
        return query
