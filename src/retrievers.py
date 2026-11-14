"""Retrieval methods over the Day 6 corpus. All share one interface:

    retriever.search(query, filters=None, top_k=3) -> list[dict]

Each result is a chunk (id, text, source_id, source, page, metadata) plus
``score``, ``rank`` (1-based) and ``method``. ``filters`` may only come from the
user/question (e.g. {"chapter": "4"}), never from gold labels.

Methods
- dense:  Day 6 Chroma collection ``chunks_B`` + multilingual-e5-small (query: prefix), cosine.
- bm25:   BM25Okapi over the same chunks (lexical).
- hybrid: Reciprocal Rank Fusion of dense and bm25 candidate lists (no score mixing).
- rerank: hybrid candidates re-scored by a cross-encoder.
"""
import re
from typing import Any

from .common import CHROMA_DIR, DAY6_COLLECTION, EMBEDDING_MODEL, norm

ALLOWED_FILTERS = {"chapter", "source_id"}
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _check_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    filters = filters or {}
    bad = set(filters) - ALLOWED_FILTERS
    if bad:
        raise ValueError(f"filter not allowed: {sorted(bad)}")
    return filters


def _matches(chunk: dict, filters: dict[str, Any]) -> bool:
    for key, value in filters.items():
        actual = chunk["source_id"] if key == "source_id" else chunk["metadata"].get(key)
        if str(actual) != str(value):
            return False
    return True


def _ranked(chunks: list[dict], scores: list[float], method: str, top_k: int) -> list[dict]:
    order = sorted(range(len(chunks)), key=lambda i: (-scores[i], chunks[i]["id"]))[:top_k]
    return [{**chunks[i], "score": float(scores[i]), "rank": r, "method": method}
            for r, i in enumerate(order, start=1)]


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks: list[dict], embedder=None):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.by_id = {c["id"]: c for c in chunks}
        self.model = embedder or SentenceTransformer(EMBEDDING_MODEL)
        self.collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(DAY6_COLLECTION)
        dim = self.model.get_embedding_dimension()
        if dim != self.collection.metadata.get("dimensions"):
            raise ValueError(f"query embedding dim {dim} != index dim {self.collection.metadata}")

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict]:
        filters = _check_filters(filters)
        clauses = [{("document_id" if k == "source_id" else k): v} for k, v in filters.items()]
        where = None if not clauses else clauses[0] if len(clauses) == 1 else {"$and": clauses}
        vec = self.model.encode([f"query: {query}"], normalize_embeddings=True)[0].tolist()
        res = self.collection.query(query_embeddings=[vec], n_results=top_k, where=where,
                                    include=["distances"])
        out = []
        for rank, (cid, dist) in enumerate(zip(res["ids"][0], res["distances"][0]), start=1):
            if cid not in self.by_id:
                raise KeyError(f"chunk {cid} in Chroma but not in data/chunks.json; re-run export")
            out.append({**self.by_id[cid], "score": 1.0 - float(dist), "rank": rank, "method": self.name})
        return out


STOPWORDS = set("""a an and are as at be by does do did for from how in is it its of on or that the
their this to was what when where which who why will with""".split())


def bm25_tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", norm(text)) if t not in STOPWORDS]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.bm25 = BM25Okapi([bm25_tokenize(c["text"]) for c in chunks], k1=k1, b=b)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict]:
        filters = _check_filters(filters)
        scores = self.bm25.get_scores(bm25_tokenize(query))
        keep = [i for i, c in enumerate(self.chunks) if _matches(c, filters) and scores[i] > 0]
        return _ranked([self.chunks[i] for i in keep], [scores[i] for i in keep], self.name, top_k)


class HybridRetriever:
    """RRF: score(d) = sum over lists of 1 / (rrf_k + rank_in_list)."""
    name = "hybrid"

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, candidates: int = 20, rrf_k: int = 60):
        self.dense, self.bm25, self.candidates, self.rrf_k = dense, bm25, candidates, rrf_k

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict]:
        fused: dict[str, float] = {}
        chunks: dict[str, dict] = {}
        for retriever in (self.dense, self.bm25):
            for r in retriever.search(query, filters=filters, top_k=self.candidates):
                fused[r["id"]] = fused.get(r["id"], 0.0) + 1.0 / (self.rrf_k + r["rank"])
                chunks[r["id"]] = {k: v for k, v in r.items() if k not in ("score", "rank", "method")}
        ids = list(fused)
        return _ranked([chunks[i] for i in ids], [fused[i] for i in ids], self.name, top_k)


class RerankRetriever:
    """Take ``candidates`` chunks from a first-stage retriever and re-score with a cross-encoder."""
    name = "rerank"

    def __init__(self, first_stage, candidates: int = 20, model_name: str = RERANK_MODEL):
        from sentence_transformers import CrossEncoder

        self.first_stage, self.candidates = first_stage, candidates
        self.model = CrossEncoder(model_name)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict]:
        cands = self.first_stage.search(query, filters=filters, top_k=self.candidates)
        if not cands:
            return []
        scores = self.model.predict([(query, c["text"]) for c in cands]).tolist()
        base = [{k: v for k, v in c.items() if k not in ("score", "rank", "method")} for c in cands]
        return _ranked(base, scores, self.name, top_k)


class NeighborExpander:
    """Context step: after each retrieved chunk, add up to ``window`` adjacent chunks of the same page.

    Chunk IDs end with the paragraph index (``...:par800:<i>``), so adjacency is
    read from the ID order within (source_id, page). Neighbours are marked
    ``neighbor_of`` and never change the primary ranking used for Hit@k.
    """

    def __init__(self, chunks: list[dict], window: int = 1):
        self.window = window
        groups: dict[tuple, list[dict]] = {}
        for c in chunks:
            groups.setdefault((c["source_id"], c["page"]), []).append(c)
        self.pos: dict[str, tuple[list[dict], int]] = {}
        for group in groups.values():
            group.sort(key=lambda c: int(c["id"].rsplit(":", 1)[-1]))
            for i, c in enumerate(group):
                self.pos[c["id"]] = (group, i)

    def expand(self, results: list[dict]) -> list[dict]:
        out, seen = [], set()
        for r in results:
            if r["id"] not in seen:
                seen.add(r["id"])
                out.append(r)
            group, i = self.pos[r["id"]]
            for j in [i + d for d in range(1, self.window + 1)] + [i - d for d in range(1, self.window + 1)]:
                if 0 <= j < len(group) and group[j]["id"] not in seen:
                    seen.add(group[j]["id"])
                    out.append({**group[j], "score": 0.0, "rank": r["rank"], "method": "neighbor",
                                "neighbor_of": r["id"]})
        return out


def build(method: str, chunks: list[dict], _cache: dict = {}) -> Any:
    """Build a retriever by name, sharing the dense/bm25 instances across methods."""
    if "dense" not in _cache and method in ("dense", "hybrid", "rerank", "dense_rerank"):
        _cache["dense"] = DenseRetriever(chunks)
    if "bm25" not in _cache and method in ("bm25", "hybrid", "rerank"):
        _cache["bm25"] = BM25Retriever(chunks)
    if method in ("dense", "bm25"):
        return _cache[method]
    if method == "hybrid":
        return _cache.setdefault("hybrid", HybridRetriever(_cache["dense"], _cache["bm25"]))
    if method == "rerank":
        hybrid = _cache.setdefault("hybrid", HybridRetriever(_cache["dense"], _cache["bm25"]))
        return _cache.setdefault("rerank", RerankRetriever(hybrid))
    if method == "dense_rerank":
        r = _cache.setdefault("dense_rerank", RerankRetriever(_cache["dense"]))
        r.name = "dense_rerank"
        return r
    raise ValueError(f"unknown method {method}")
