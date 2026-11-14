"""Compare agent runs on one split with automatic metrics (manual grading still required).

    python -m src.compare_agent --split dev
    python -m src.compare_agent --split dev --runs dev-dense dev-rerank

Per run (results/agent/<run>/traces.json):
- tool use, error count, search-count distribution
- Agent Context Evidence Rate: answerable cases whose final context holds a gold chunk
- Answered-with-gold-citation: answered and >= 1 citation is a gold-evidence chunk (proxy, not a grade)
- False abstention on answerable, correct abstention on unanswerable
- tokens, median time, estimated cost (prices from --price-in/--price-out per 1M tokens)
Gold is read here only for scoring finished runs. Incomplete runs are flagged.
"""
import argparse
import statistics
import sys

from .common import DATA, RESULTS, hit, read_json, write_json


def score_run(traces: list[dict], cases: dict, questions: dict, chunks: dict, price_in: float, price_out: float):
    ans_ids = [i for i, c in cases.items() if c["answerable"]]
    un_ids = [i for i, c in cases.items() if not c["answerable"]]
    T = {t["query_id"]: t for t in traces}
    ok = [t for t in traces if t["status"] != "error"]

    def gold_hit(cid, qid):
        return cid in chunks and hit(chunks[cid], questions[qid]["gold"])

    ctx = [q for q in ans_ids if q in T and any(gold_hit(c, q) for c in T[q]["final_context_ids"])]
    cited = [q for q in ans_ids if q in T and T[q]["status"] == "answered"
             and any(gold_hit(c, q) for c in T[q]["citations"])]
    false_abs = [q for q in ans_ids if q in T and T[q]["status"] == "abstain"]
    good_abs = [q for q in un_ids if q in T and T[q]["status"] == "abstain"]
    p_tok = sum(t["prompt_tokens"] for t in traces)
    c_tok = sum(t["completion_tokens"] for t in traces)
    searches = [t["n_searches"] for t in ok]
    return {
        "complete": len(traces) == len(cases) and all(t["status"] != "error" for t in traces),
        "n_run": len(traces), "errors": [t["query_id"] for t in traces if t["status"] == "error"],
        "tool_use": f"{sum(s > 0 for s in searches)}/{len(ok)}",
        "avg_searches": round(statistics.mean(searches), 2) if searches else None,
        "context_evidence": f"{len(ctx)}/{len(ans_ids)}",
        "answered_gold_citation": f"{len(cited)}/{len(ans_ids)}",
        "false_abstention": f"{len(false_abs)}/{len(ans_ids)}", "false_abstention_ids": false_abs,
        "abstention_correct": f"{len(good_abs)}/{len(un_ids)}",
        "no_gold_citation_ids": [q for q in ans_ids if q in T and T[q]["status"] == "answered" and q not in cited],
        "avg_prompt_tokens": round(p_tok / len(traces)) if traces else 0,
        "median_s": round(statistics.median(t["elapsed_s"] for t in ok), 2) if ok else None,
        "est_cost_usd": round((p_tok * price_in + c_tok * price_out) / 1e6, 5),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=("dev", "test"))
    p.add_argument("--runs", nargs="*", help="folder names under results/agent (default: all for split)")
    p.add_argument("--price-in", type=float, default=0.15, help="USD per 1M prompt tokens (check provider page)")
    p.add_argument("--price-out", type=float, default=0.60, help="USD per 1M completion tokens")
    args = p.parse_args()

    cases = {c["id"]: c for c in read_json(DATA / "cases.json") if c["split"] == args.split}
    questions = {q["id"]: q for q in read_json(DATA / "questions.json")}
    chunks = {c["id"]: c for c in read_json(DATA / "chunks.json")}
    runs = args.runs or sorted(d.name for d in (RESULTS / "agent").iterdir()
                               if d.name.startswith(args.split + "-") and (d / "traces.json").exists()
                               and "smoke" not in d.name)
    rows = {}
    for run in runs:
        rows[run] = score_run(read_json(RESULTS / "agent" / run / "traces.json"), cases, questions, chunks,
                              args.price_in, args.price_out)
    write_json(RESULTS / "agent" / f"comparison-{args.split}.json",
               {"price_per_1M": [args.price_in, args.price_out], "runs": rows})
    hdr = f"{'run':<24}{'done':>6}{'tool':>7}{'srch':>6}{'ctx':>7}{'cite':>7}{'F-abs':>7}{'abst':>6}{'tok/q':>7}{'s':>6}{'$':>9}"
    print(hdr)
    for run, r in rows.items():
        print(f"{run:<24}{'yes' if r['complete'] else 'NO':>6}{r['tool_use']:>7}{str(r['avg_searches']):>6}"
              f"{r['context_evidence']:>7}{r['answered_gold_citation']:>7}{r['false_abstention']:>7}"
              f"{r['abstention_correct']:>6}{r['avg_prompt_tokens']:>7}{str(r['median_s']):>6}{r['est_cost_usd']:>9}")
    for run, r in rows.items():
        print(f"{run}: false abstain {r['false_abstention_ids']}, answered w/o gold citation "
              f"{r['no_gold_citation_ids']}, errors {r['errors']}")


if __name__ == "__main__":
    main()
