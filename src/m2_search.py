"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os
import re
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K,
                    JINA_NORMALIZED, JINA_TRUNCATE)
from src.jina_embeddings import (
    JinaEmbeddingsClient,
    JINA_TASK_PASSAGE,
    JINA_TASK_QUERY,
    fallback_hash_embeddings,
)

_JINA_CLIENT = JinaEmbeddingsClient()


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except Exception:
        return " ".join(re.findall(r"[0-9A-Za-zÀ-ỹ]+", text))


def _tokenize(text: str) -> list[str]:
    return segment_vietnamese(text).split()


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi
        self.documents = chunks
        self.corpus_tokens = [_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if not self.bm25:
            return []
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for i in top_indices:
            results.append(SearchResult(
                text=self.documents[i]["text"],
                score=float(scores[i]),
                metadata=self.documents[i].get("metadata", {}),
                method="bm25"
            ))
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._fallback_docs: list[dict] = []
        self._fallback_vectors: list[list[float]] = []
        self._vector_dim = EMBEDDING_DIM
        self._embedding_mode = "jina"

    def _embed_texts(self, texts: list[str], task: str) -> list[list[float]] | None:
        vectors = _JINA_CLIENT.embed_texts(
            texts,
            model=EMBEDDING_MODEL,
            task=task,
            dimensions=self._vector_dim,
            normalized=JINA_NORMALIZED,
            truncate=JINA_TRUNCATE,
        )
        return vectors

    def _local_query_vector(self, query: str) -> list[float]:
        return fallback_hash_embeddings([query], dimensions=self._vector_dim)[0]

    def _vector_search(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        if not self._fallback_vectors or not self._fallback_docs:
            return []

        q = np.asarray(query_vector, dtype=float)
        scores = []
        for i, vector in enumerate(self._fallback_vectors):
            v = np.asarray(vector, dtype=float)
            denom = float(np.linalg.norm(q) * np.linalg.norm(v))
            score = float(np.dot(q, v) / denom) if denom else 0.0
            scores.append((score, i))
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [
            SearchResult(
                text=self._fallback_docs[i]["text"],
                score=float(score),
                metadata=self._fallback_docs[i].get("metadata", {}),
                method="dense",
            )
            for score, i in scores[:top_k]
        ]

    def _lexical_search(self, query: str, top_k: int) -> list[SearchResult]:
        query_tokens = set(_tokenize(query))
        if not query_tokens or not self._fallback_docs:
            return []

        scored = []
        for i, doc in enumerate(self._fallback_docs):
            doc_tokens = set(_tokenize(doc["text"]))
            overlap = len(query_tokens & doc_tokens)
            score = overlap / max(1.0, len(doc_tokens) ** 0.5)
            scored.append((score, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [
            SearchResult(
                text=self._fallback_docs[i]["text"],
                score=float(score),
                metadata=self._fallback_docs[i].get("metadata", {}),
                method="dense",
            )
            for score, i in scored[:top_k]
        ]

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PointStruct
        texts = [c["text"] for c in chunks]
        vectors = self._embed_texts(texts, task=JINA_TASK_PASSAGE)
        if vectors is None:
            vectors = fallback_hash_embeddings(texts, dimensions=self._vector_dim)
            self._embedding_mode = "local"

        if not vectors:
            return

        self._vector_dim = len(vectors[0])
        self._fallback_docs = chunks
        self._fallback_vectors = vectors

        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=self._vector_dim, distance=Distance.COSINE)
        )

        points = []
        for i, (text, vector) in enumerate(zip(texts, vectors)):
            points.append(PointStruct(
                id=i,
                vector=list(vector),
                payload={**chunks[i].get("metadata", {}), "text": text}
            ))
        self.client.upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if self._embedding_mode == "local":
            query_vector = self._local_query_vector(query)
            try:
                res = self.client.query_points(
                    collection_name=collection,
                    query=query_vector,
                    limit=top_k
                )
                hits = res.points
                return [
                    SearchResult(
                        text=hit.payload["text"],
                        score=hit.score,
                        metadata=hit.payload,
                        method="dense"
                    ) for hit in hits
                ]
            except Exception:
                return self._vector_search(query_vector, top_k) or self._lexical_search(query, top_k)

        query_vectors = self._embed_texts([query], task=JINA_TASK_QUERY)
        if query_vectors is None:
            return self._lexical_search(query, top_k)

        query_vector = query_vectors[0]
        try:
            res = self.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=top_k
            )
            hits = res.points
            return [
                SearchResult(
                    text=hit.payload["text"],
                    score=hit.score,
                    metadata=hit.payload,
                    method="dense"
                ) for hit in hits
            ]
        except Exception:
            return self._lexical_search(query, top_k)


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                            top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores = {}  # text -> {"score": float, "result": SearchResult}

    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {"score": 0.0, "result": result}
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)

    # Sort by score descending
    sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    final_results = []
    for item in sorted_items:
        res = item["result"]
        final_results.append(SearchResult(
            text=res.text,
            score=item["score"],
            metadata=res.metadata,
            method="hybrid"
        ))
    return final_results


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print("Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
