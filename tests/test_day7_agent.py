"""Code tests for src/agent.py with a scripted controller (not a benchmark)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent import pack_round_robin, run_agent  # noqa: E402


def chunk(i, page=1, text=None):
    return {"id": f"doc:p{page}:{i}", "text": text or f"text {i}", "source_id": "doc",
            "source": "x", "page": page, "metadata": {}}


class FakeRetriever:
    name = "fake"

    def __init__(self, pool):
        self.pool, self.calls = pool, []

    def search(self, query, *, filters=None, top_k=3):
        self.calls.append((query, filters))
        return [{**c, "score": 1.0 / r, "rank": r} for r, c in enumerate(self.pool[:top_k], 1)]


class ScriptedController:
    def __init__(self, actions):
        self.actions, self.seen = list(actions), []

    def decide(self, question, searches, context, allow_search):
        self.seen.append({"searches": list(searches), "context": context, "allow": allow_search})
        meta = {"latency_ms": 1, "attempts": 1, "prompt_tokens": 10, "completion_tokens": 2, "cost_usd": 0}
        return {**self.actions.pop(0), "llm": meta}


CASE = {"id": "q1", "split": "dev", "question": "Q?", "filter": {"chapter": "4"}}


class AgentTests(unittest.TestCase):
    def test_search_then_answer_maps_labels_to_ids(self):
        r = FakeRetriever([chunk(1), chunk(2)])
        c = ScriptedController([{"action": "search", "query": "q"},
                                {"action": "answer", "answer": "Yes [S2].", "citations": ["S2"]}])
        t = run_agent(CASE, c, r, max_searches=2, top_k=2, budget=1000)
        self.assertEqual(t["status"], "answered")
        self.assertEqual(t["citations"], ["doc:p1:2"])
        self.assertIn("[doc:p1:2]", t["answer"])
        self.assertEqual(r.calls, [("q", {"chapter": "4"})])  # filter comes from the case
        self.assertIn("[S1]", c.seen[1]["context"])
        self.assertEqual(t["prompt_tokens"], 20)

    def test_citation_outside_context_becomes_abstain(self):
        r = FakeRetriever([chunk(1)])
        c = ScriptedController([{"action": "search", "query": "q"},
                                {"action": "answer", "answer": "Yes [S9].", "citations": ["S9"]}])
        t = run_agent(CASE, c, r, max_searches=2, top_k=1, budget=1000)
        self.assertEqual(t["status"], "abstain")
        self.assertEqual(t["abstain_reason"], "invalid_citations")
        self.assertEqual(t["events"][1]["raw_answer"], "Yes [S9].")

    def test_listed_but_not_inline_citation_is_appended_and_flagged(self):
        r = FakeRetriever([chunk(1)])
        c = ScriptedController([{"action": "search", "query": "q"},
                                {"action": "answer", "answer": "Yes.", "citations": ["S1"]}])
        t = run_agent(CASE, c, r, max_searches=2, top_k=1, budget=1000)
        self.assertEqual(t["status"], "answered")
        self.assertEqual(t["answer"], "Yes. [doc:p1:1]")
        self.assertEqual(t["events"][1]["citations_appended"], ["S1"])

    def test_search_limit_removes_search_tool(self):
        r = FakeRetriever([chunk(1)])
        c = ScriptedController([{"action": "search", "query": "a"}, {"action": "abstain", "reason": "none"}])
        t = run_agent(CASE, c, r, max_searches=1, top_k=1, budget=1000)
        self.assertEqual([s["allow"] for s in c.seen], [True, False])
        self.assertEqual(t["status"], "abstain")
        self.assertEqual(t["n_searches"], 1)

    def test_provider_error_is_recorded(self):
        class Boom:
            def decide(self, *a):
                raise RuntimeError("429")
        t = run_agent(CASE, Boom(), FakeRetriever([]), max_searches=1, top_k=1, budget=100)
        self.assertEqual(t["status"], "error")
        self.assertIsNone(t["answer"])

    def test_round_robin_lets_later_search_add_evidence(self):
        s1 = [{**chunk(1), "text": "a" * 300}, {**chunk(2), "text": "b" * 300}]
        s2 = [{**chunk(3), "text": "c" * 300}]
        kept = pack_round_robin([s1, s2], budget=750)
        self.assertEqual([c["id"] for c in kept], ["doc:p1:1", "doc:p1:3"])


class NeighborTests(unittest.TestCase):
    def test_adds_same_page_neighbours_only(self):
        from src.retrievers import NeighborExpander
        chunks = [chunk(0, 1), chunk(1, 1), chunk(2, 1), chunk(0, 2)]
        ex = NeighborExpander(chunks, window=1)
        out = ex.expand([{**chunks[1], "score": 1.0, "rank": 1}])
        self.assertEqual([c["id"] for c in out], ["doc:p1:1", "doc:p1:2", "doc:p1:0"])
        self.assertEqual(out[1]["neighbor_of"], "doc:p1:1")
        out = ex.expand([{**chunks[3], "score": 1.0, "rank": 1}])
        self.assertEqual([c["id"] for c in out], ["doc:p2:0"])


if __name__ == "__main__":
    unittest.main()
