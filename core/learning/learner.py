from typing import Optional


class Learner:
    def __init__(self, chunk_store):
        self.chunk_store = chunk_store

    def learn(
        self,
        evaluation: dict,
        chunk_ids: list[str],
        feedback: Optional[str] = None
    ):
        if evaluation["passed"]:
            self._positive_update(chunk_ids)
        else:
            self._negative_update(chunk_ids)

        if feedback == "dislike":
            self._user_negative(chunk_ids)

    def _positive_update(self, chunk_ids):
        for cid in chunk_ids:
            stats = self.chunk_store.get(cid)
            stats.times_used += 1
            stats.positive_feedback += 1
            stats.trust_score = min(1.0, stats.trust_score + 0.02)

    def _negative_update(self, chunk_ids):
        for cid in chunk_ids:
            stats = self.chunk_store.get(cid)
            stats.negative_feedback += 1
            stats.trust_score = max(0.0, stats.trust_score - 0.05)

    def _user_negative(self, chunk_ids):
        for cid in chunk_ids:
            stats = self.chunk_store.get(cid)
            stats.trust_score = max(0.0, stats.trust_score - 0.1)
