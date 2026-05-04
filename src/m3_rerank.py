"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "rerank-multilingual-v3.0"):
        self.model_name = model_name
        self._model = None

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
            
        client = self._load_model()
        docs_for_cohere = [doc["text"] for doc in documents]
        
        # Gọi Cohere API
        response = client.rerank(
            model=self.model_name,
            query=query,
            documents=docs_for_cohere,
            top_n=top_k
        )
        
        results = []
        for i, hit in enumerate(response.results):
            doc_idx = hit.index
            original_doc = documents[doc_idx]
            results.append(
                RerankResult(
                    text=original_doc["text"],
                    original_score=float(original_doc.get("score", 0.0)),
                    rerank_score=float(hit.relevance_score),
                    metadata=original_doc.get("metadata", {}),
                    rank=i
                )
            )
        return results


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        # TODO (optional): from flashrank import Ranker, RerankRequest
        # model = Ranker(); passages = [{"text": d["text"]} for d in documents]
        # results = model.rerank(RerankRequest(query=query, passages=passages))
        return []


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    import numpy as np
    times = []
    # Warmup
    reranker.rerank(query, documents)
    
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)  # ms
        
    return {
        "avg_ms": float(np.mean(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times))
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
