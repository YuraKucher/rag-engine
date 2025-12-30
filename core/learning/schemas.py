from dataclasses import dataclass


@dataclass
class ChunkStats:
    chunk_id: str
    times_retrieved: int = 0
    times_used: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    trust_score: float = 0.5
