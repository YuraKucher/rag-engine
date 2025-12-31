class RAGService:

    def answer(self, query: str, user_feedback=None):

        # 1. CACHE SEARCH
        cached = self.cache.search(query)
        if cached:
            evaluation = self.evaluator.evaluate_cached(cached)

            self.trainer.learn(
                evaluation=evaluation,
                context=cached["chunk_ids"],
                feedback=user_feedback
            )
            return cached["answer"]

        # 2. RETRIEVAL
        docs = self.retriever.retrieve(query)

        # 3. REASONING
        context = self.reasoner.build_context(query, docs)

        # 4. GENERATION
        answer = self.generator.generate(query, context)

        # 5. EVALUATION
        evaluation = self.evaluator.evaluate(
            query=query,
            answer=answer,
            context=context
        )

        # 6. CACHE STORE
        if evaluation.passed:
            self.cache.store(
                query=query,
                answer=answer,
                chunk_ids=context.chunk_ids,
                evaluation=evaluation
            )

        # 7. LEARNING
        self.trainer.learn(
            evaluation=evaluation,
            context=context.chunk_ids,
            feedback=user_feedback
        )

        return answer
