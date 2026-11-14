"""Export Day 6 chunks (Chroma) and benchmark into the Day 7 data format.

    python -m src.export_day6

Writes data/chunks.json and data/questions.json (answerable questions).
Unanswerable cases are hand-written in data/unanswerable.json; data/cases.json
merges both for the agent. Gold is only used here for validation and later
for scoring, never inside retrieval or the agent.
"""
import chromadb

from .common import CHROMA_DIR, DATA, DAY6_COLLECTION, DAY6_DIR, hit, read_json, write_json


def export_chunks() -> list[dict]:
    col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(DAY6_COLLECTION)
    res = col.get(include=["documents", "metadatas"])
    chunks = []
    for cid, text, meta in zip(res["ids"], res["documents"], res["metadatas"]):
        chunks.append({
            "id": cid, "text": text, "source_id": meta["document_id"],
            "source": meta["source"], "page": int(meta["page"]),
            "metadata": {"chapter": meta["chapter"], "content_hash": meta["content_hash"]},
        })
    chunks.sort(key=lambda c: (c["source_id"], c["page"], c["id"]))
    if len({c["id"] for c in chunks}) != len(chunks):
        raise ValueError("duplicate chunk id")
    return chunks


def convert_questions(chunks: list[dict]) -> list[dict]:
    out = []
    for q in read_json(DAY6_DIR / "benchmark" / "questions.json"):
        golds = q.get("gold") or [{"source_id": q["gold_source_id"], "page": q["gold_page"],
                                    "evidence_quote": q["evidence_quote"]}]
        gold = [{"source_id": g["source_id"], "page": g["page"], "evidence": g["evidence_quote"]}
                for g in golds]
        # Keep only gold evidences that lie entirely in one chunk of the index (Day 7 schema).
        valid = [g for g in gold if any(hit(c, [g]) for c in chunks)]
        dropped = [g for g in gold if g not in valid]
        if not valid:
            raise ValueError(f"{q['id']}: no gold evidence lies within a single chunk of {DAY6_COLLECTION}")
        item = {"id": q["id"], "split": q["split"], "type": q["type"], "question": q["question"],
                "expected_answer": q["expected_answer"], "gold": valid, "answerable": True,
                "expected_behavior": "search_and_answer", "filter": q.get("filter")}
        if dropped:
            item["dropped_gold"] = dropped
        out.append(item)
    return out


def main() -> None:
    chunks = export_chunks()
    questions = convert_questions(chunks)
    write_json(DATA / "chunks.json", chunks)
    write_json(DATA / "questions.json", questions)
    unans_path = DATA / "unanswerable.json"
    unanswerable = read_json(unans_path) if unans_path.exists() else []
    cases = [{k: q[k] for k in ("id", "split", "question", "answerable", "expected_behavior", "filter")}
             for q in questions]
    cases += [{k: u.get(k) for k in ("id", "split", "question", "answerable", "expected_behavior",
                                      "absence_reason", "filter")} for u in unanswerable]
    write_json(DATA / "cases.json", cases)
    pages = {(c["source_id"], c["page"]) for c in chunks}
    print(f"chunks: {len(chunks)} from {len(pages)} pages -> data/chunks.json")
    for split in ("dev", "test"):
        a = sum(q["split"] == split for q in questions)
        u = sum(x["split"] == split for x in unanswerable)
        print(f"{split}: {a} answerable + {u} unanswerable")
    for q in questions:
        if "dropped_gold" in q:
            print(f"  note {q['id']}: dropped {len(q['dropped_gold'])} gold not inside one chunk")


if __name__ == "__main__":
    main()
