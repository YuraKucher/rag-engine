class RetrievalPolicy:
    """
    Політики retrieval.
    НІЯКОЇ логіки — лише параметри.
    """

    def __init__(
        self,
        top_k: int = 5,
        rerank_k: int = 3,
        use_query_rewrite: bool = False
    ):
        self.top_k = top_k
        self.rerank_k = rerank_k
        self.use_query_rewrite = use_query_rewrite
