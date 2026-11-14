"""Shared paths, I/O and evidence matching for Day 7."""
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
# Day 6 repo (Chroma store, pages, benchmark). Override with env DAY6_DIR.
DAY6_DIR = Path(os.environ.get(
    "DAY6_DIR",
    ROOT.parent / "K5-L3A-DAY06-NguyenKhacKhanh-2A202603472-DataIngestionEngineering",
)) / "starter"
CHROMA_DIR = DAY6_DIR / "chroma_db"
DAY6_COLLECTION = "chunks_B"  # index chosen in Day 6 (strategy C = B + filter)
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def norm(text: str) -> str:
    """Same rule as Day 6 evaluation: NFKC, collapse whitespace, lowercase."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def hit(chunk: dict, gold: list[dict]) -> bool:
    """A chunk is a hit if it is from the gold source/page and contains a whole gold evidence."""
    text = norm(chunk["text"])
    return any(chunk["source_id"] == g["source_id"] and chunk["page"] == g["page"]
               and norm(g["evidence"]) in text for g in gold)
