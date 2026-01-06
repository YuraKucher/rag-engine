import time


class ProgressTracker:
    def __init__(self, stage: str, total: int):
        self.stage = stage
        self.total = total
        self.start_time = time.time()
        self.current = 0

    def step(self, n: int = 1):
        self.current += n
        elapsed = time.time() - self.start_time
        speed = self.current / elapsed if elapsed > 0 else 0

        return {
            "stage": self.stage,
            "current": self.current,
            "total": self.total,
            "percent": self.current / self.total if self.total else 1.0,
            "speed": speed,          # items / second
            "elapsed": elapsed
        }
