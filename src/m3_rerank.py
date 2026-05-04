"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import logging
import os
import re
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FLASHRANK_CACHE_DIR,
    FLASHRANK_MAX_LENGTH,
    FLASHRANK_MODEL,
    RERANK_MODEL,
    RERANK_TOP_K,
)

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "và", "là", "của", "cho", "trong", "một", "những", "các", "được",
    "có", "theo", "với", "từ", "này", "đó", "khi", "thì", "lại", "ra",
    "về", "để", "ở", "như", "nên", "do", "đó", "nếu", "hoặc", "đến",
    "what", "which", "who", "when", "where", "why", "how", "is", "are",
    "the", "a", "an", "of", "to", "for", "in", "on", "and", "or", "be",
}


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    tokens = re.findall(r"[0-9A-Za-zÀ-ỹ]+", text.lower())
    return [tok for tok in tokens if tok not in _STOPWORDS]


def _heuristic_score(query: str, text: str) -> float:
    query_tokens = set(_tokenize(query))
    doc_tokens = set(_tokenize(text))
    if not query_tokens or not doc_tokens:
        return 0.0

    overlap = len(query_tokens & doc_tokens)
    if overlap == 0:
        return 0.0

    coverage = overlap / len(query_tokens)
    jaccard = overlap / len(query_tokens | doc_tokens)
    phrase_bonus = 0.0
    lowered_text = text.lower()
    for token in query_tokens:
        if len(token) >= 4 and token in lowered_text:
            phrase_bonus += 0.03
    return min(1.0, 0.7 * coverage + 0.3 * jaccard + phrase_bonus)


def _heuristic_rerank(query: str, documents: list[dict], top_k: int) -> list[RerankResult]:
    scored = []
    for idx, doc in enumerate(documents):
        text = doc.get("text", "")
        score = _heuristic_score(query, text)
        scored.append((score, idx, doc))

    scored.sort(key=lambda item: (-item[0], item[1]))
    results: list[RerankResult] = []
    for rank, (score, _, doc) in enumerate(scored[:top_k]):
        results.append(
            RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=rank,
            )
        )
    return results


class FlashrankReranker:
    """Optional local reranker fallback powered by flashrank when available."""

    def __init__(
        self,
        model_name: str = FLASHRANK_MODEL,
        max_length: int = FLASHRANK_MAX_LENGTH,
        cache_dir: str = FLASHRANK_CACHE_DIR,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.cache_dir = cache_dir.strip() or None
        self._ranker = None
        self._request_cls = None
        self._available: bool | None = None

    def _load_model(self):
        if self._available is False:
            return None
        if self._ranker is not None:
            return self._ranker

        try:
            from flashrank import Ranker, RerankRequest
        except ImportError:
            self._available = False
            logger.warning("flashrank is not installed; using heuristic reranker fallback.")
            return None

        kwargs: dict[str, object] = {"model_name": self.model_name, "max_length": self.max_length}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir

        try:
            self._ranker = Ranker(**kwargs)
            self._request_cls = RerankRequest
            self._available = True
            return self._ranker
        except TypeError:
            # Some flashrank versions accept a smaller constructor surface.
            try:
                self._ranker = Ranker(max_length=self.max_length)
                self._request_cls = RerankRequest
                self._available = True
                return self._ranker
            except Exception as exc:
                self._available = False
                logger.warning("flashrank initialization failed: %s", exc)
                return None
        except Exception as exc:
            self._available = False
            logger.warning("flashrank initialization failed: %s", exc)
            return None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []

        ranker = self._load_model()
        if ranker is None or self._request_cls is None:
            return _heuristic_rerank(query, documents, top_k)

        passages = [
            {
                "id": idx,
                "text": doc.get("text", ""),
                "meta": doc.get("metadata", {}),
            }
            for idx, doc in enumerate(documents)
        ]

        try:
            request = self._request_cls(query=query, passages=passages)
            ranked = ranker.rerank(request)
        except Exception as exc:
            logger.warning("flashrank rerank failed; falling back to heuristic reranker: %s", exc)
            return _heuristic_rerank(query, documents, top_k)

        results: list[RerankResult] = []
        by_id = {idx: doc for idx, doc in enumerate(documents)}
        by_text = {doc.get("text", ""): doc for doc in documents}

        for rank, hit in enumerate(ranked[:top_k]):
            hit_id = hit.get("id") if isinstance(hit, dict) else getattr(hit, "id", None)
            hit_text = hit.get("text") if isinstance(hit, dict) else getattr(hit, "text", "")
            original_doc = by_id.get(hit_id) if hit_id is not None else by_text.get(hit_text, {})
            score = hit.get("score") if isinstance(hit, dict) else getattr(hit, "score", 0.0)
            results.append(
                RerankResult(
                    text=original_doc.get("text", hit_text),
                    original_score=float(original_doc.get("score", 0.0)),
                    rerank_score=float(score or 0.0),
                    metadata=original_doc.get("metadata", {}),
                    rank=rank,
                )
            )

        return results


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        self.model_name = model_name
        self._model = None
        self._fallback = FlashrankReranker()

    def _load_model(self):
        if self._model is None:
            import cohere
            import dotenv

            dotenv.load_dotenv()
            api_key = os.environ.get("COHERE_API_KEY")
            if not api_key:
                raise ValueError("COHERE_API_KEY không được tìm thấy trong .env")
            self._model = cohere.Client(api_key)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []

        try:
            client = self._load_model()
            docs_for_cohere = [doc.get("text", "") for doc in documents]

            response = client.rerank(
                model=self.model_name,
                query=query,
                documents=docs_for_cohere,
                top_n=top_k,
            )

            results = []
            for i, hit in enumerate(response.results):
                doc_idx = hit.index
                original_doc = documents[doc_idx]
                results.append(
                    RerankResult(
                        text=original_doc.get("text", ""),
                        original_score=float(original_doc.get("score", 0.0)),
                        rerank_score=float(hit.relevance_score),
                        metadata=original_doc.get("metadata", {}),
                        rank=i,
                    )
                )
            return results
        except Exception as exc:
            logger.warning("Cohere rerank failed; falling back to Flashrank/heuristic reranker: %s", exc)
            return self._fallback.rerank(query, documents, top_k)


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    import numpy as np

    times = []
    reranker.rerank(query, documents)

    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)  # ms

    return {
        "avg_ms": float(np.mean(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
