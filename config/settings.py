from pathlib import Path
import yaml


BASE_PATH = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_PATH / "config"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Settings:
    """
    Єдина точка доступу до всієї конфігурації.
    """

    def __init__(self):
        self.models = _load_yaml(CONFIG_PATH / "models.yaml")
        self.generation = _load_yaml(CONFIG_PATH / "generation.yaml")
        self.thresholds = _load_yaml(CONFIG_PATH / "thresholds.yaml")
        self.paths = _load_yaml(CONFIG_PATH / "paths.yaml")
        self.system = _load_yaml(CONFIG_PATH / "system.yaml")


settings = Settings()
