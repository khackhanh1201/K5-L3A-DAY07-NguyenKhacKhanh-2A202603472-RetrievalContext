"""Benchmark retrieval methods on one split (no LLM involved).

    python -m src.eval_retrieval --split dev
    python -m src.eval_retrieval --split dev --methods dense hybrid --top-k 5 --budget 4000 --tag k5

Metrics on answerable questions only (unanswerable ones are excluded from the
denominator, as BENCHMARK.md requires):
- Hit@1, Evidence Hit@3: a gold evidence lies whole in a top-1 / top-3 chunk.
- MRR@10: reciprocal rank of the first hit within top-10.
- Context Evidence Rate: a hit chunk survives packing of the top-k into the char budget.
Gold is read only after retrieval, to mark hits.
"""
import argparse
import statistics
import time

from .common import DATA, RESULTS, hit, read_json, write_json
from .retrievers import NeighborExpander, build


def pack_context(results: list[dict], budget: int) -> tuple[str, list[str]]:
    """Same format as starter/baseline.py: keep chunks in rank order while they fit."""
    blocks, ids = [], []
    for c in results:
        if c["id"] in ids:
            continue
        block = f"[{c['id']} | {c['source']} | page {c['page']}]\n{c['text']}"
        if len("\n\n".join(blocks + [block])) <= budget:
            blocks.append(block)
            ids.append(c["id"])
    return "\n\n".join(blocks), ids


def evaluate(method: str, retriever, questions: list[dict], top_k: int, budget: int, use_filter: bool,
             expander=None):
    traces = []
    for q in questions:
        filters = q.get("filter") if use_filter else None
        start = time.perf_counter()
        top10 = retriever.search(q["question"], filters=filters, top_k=max(10, top_k))
        elapsed_ms = (time.perf_counter() - start) * 1000
        selected = top10[:top_k]
        ctx_list = expander.expand(selected) if expander else selected
        context, ids = pack_context(ctx_list, budget)
        ctx_chunks = {c['id']: c for c in ctx_list}
        marked = [{"id": c["id"], "rank": c["rank"], "score": round(c["score"], 5),
                   "source_id": c["source_id"], "page": c["page"], "chapter": c["metadata"].get("chapter"),
                   "hit": hit(c, q["gold"]), "text": c["text"]} for c in top10]
        first = next((m["rank"] for m in marked if m["hit"]), None)
        traces.append({
            "query_id": q["id"], "split": q["split"], "type": q.get("type"), "question": q["question"],
            "method": method, "filter": filters, "top_k": top_k, "budget_chars": budget,
            "results_top10": marked, "context_ids": ids, "context_chars": len(context),
            "elapsed_ms": round(elapsed_ms, 1),
            "hit1": bool(marked and marked[0]["hit"]),
            "hit3": any(m["hit"] for m in marked[:3]),
            "mrr10": 1.0 / first if first else 0.0,
            "context_hit": any(hit(ctx_chunks[i], q["gold"]) for i in ids),
            "gold": q["gold"],
        })
    return traces


def summarize(method: str, traces: list[dict], cfg: dict) -> dict:
    n = len(traces)
    count = lambda key: sum(bool(t[key]) for t in traces)
    return {
        "method": method, **cfg, "n_answerable": n,
        "hit1": f"{count('hit1')}/{n}", "hit1_rate": round(count("hit1") / n, 3),
        "hit3": f"{count('hit3')}/{n}", "hit3_rate": round(count("hit3") / n, 3),
        "mrr10": round(sum(t["mrr10"] for t in traces) / n, 3),
        "context_evidence": f"{count('context_hit')}/{n}", "context_evidence_rate": round(count("context_hit") / n, 3),
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces)),
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 1),
        "misses_at3": [t["query_id"] for t in traces if not t["hit3"]],
        "service_cost_usd": 0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=("dev", "test"))
    p.add_argument("--methods", nargs="+", default=["dense", "bm25", "hybrid", "rerank"])
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--budget", type=int, default=2400, help="context budget in characters")
    p.add_argument("--no-filter", action="store_true", help="ignore user-provided filters")
    p.add_argument("--neighbors", type=int, default=0, help="add N adjacent same-page chunks per hit to context")
    p.add_argument("--tag", default="", help="suffix for the output folder")
    args = p.parse_args()

    chunks = read_json(DATA / "chunks.json")
    questions = [q for q in read_json(DATA / "questions.json") if q["split"] == args.split]
    cfg = {"split": args.split, "top_k": args.top_k, "budget_chars": args.budget,
           "use_filter": not args.no_filter, "neighbors": args.neighbors}
    expander = NeighborExpander(chunks, args.neighbors) if args.neighbors else None
    rows = []
    for method in args.methods:
        t0 = time.perf_counter()
        retriever = build(method, chunks)
        load_s = time.perf_counter() - t0
        traces = evaluate(method, retriever, questions, args.top_k, args.budget, not args.no_filter, expander)
        summary = {**summarize(method, traces, cfg), "load_seconds": round(load_s, 1)}
        out = RESULTS / "retrieval" / f"{args.split}-{method}{'-' + args.tag if args.tag else ''}"
        write_json(out / "traces.json", traces)
        write_json(out / "summary.json", summary)
        rows.append(summary)

    print(f"\nsplit={args.split} top_k={args.top_k} budget={args.budget} filter={not args.no_filter} neighbors={args.neighbors}")
    print(f"{'method':<14}{'Hit@1':>8}{'Hit@3':>8}{'MRR@10':>8}{'Context':>9}{'ms':>8}  misses@3")
    for r in rows:
        print(f"{r['method']:<14}{r['hit1']:>8}{r['hit3']:>8}{r['mrr10']:>8}{r['context_evidence']:>9}"
              f"{r['median_ms']:>8}  {','.join(r['misses_at3'])}")


if __name__ == "__main__":
    main()
