import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data"

CACHE_PATH = ROOT / "cache"


def load_data():
    with open(DATA_PATH / "movies.json", "r") as f:
        return json.load(f)["movies"]


def load_stop_words():
    with open(DATA_PATH / "stopwords.txt", "r") as f:
        data = f.readlines()
        data = map(str.strip, data)
    return data
