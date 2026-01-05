import os
import json
import pandas as pd


def load_evaluations(path: str) -> pd.DataFrame:
    records = []

    if not os.path.exists(path):
        return pd.DataFrame()

    for fname in os.listdir(path):
        if fname.endswith(".json"):
            with open(os.path.join(path, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
                record = {
                    "evaluation_id": data["evaluation_id"],
                    "created_at": data["created_at"],
                    "relevance": data["metrics"].get("relevance"),
                    "groundedness": data["metrics"].get("groundedness"),
                    "answerability": data["metrics"].get("answerability"),
                }
                records.append(record)

    df = pd.DataFrame(records)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def load_feedback(path: str) -> pd.DataFrame:
    records = []

    if not os.path.exists(path):
        return pd.DataFrame()

    for fname in os.listdir(path):
        if fname.endswith(".json"):
            with open(os.path.join(path, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
                records.append(data)

    return pd.DataFrame(records)
