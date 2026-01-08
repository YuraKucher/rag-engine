class RetrievalPolicy:
    """
    Політики retrieval.
    ЛИШЕ параметри, але з базовою валідацією.
    """

    def __init__(
        self,
        top_k: int = 5,
        rerank_k: int = 3,
        use_query_rewrite: bool = False
    ):
        if rerank_k > top_k:
            raise ValueError("rerank_k cannot be greater than top_k")

        self.top_k = top_k
        self.rerank_k = rerank_k
        self.use_query_rewrite = use_query_rewrite
