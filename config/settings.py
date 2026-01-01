from pathlib import Path
import yaml


BASE_PATH = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_PATH / "config"


class Settings:
    def __init__(self):
        self.models = self._load_yaml("models.yaml")

    def _load_yaml(self, filename: str) -> dict:
        path = CONFIG_PATH / filename
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


settings = Settings()
