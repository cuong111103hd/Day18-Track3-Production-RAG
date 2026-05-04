"""Shared Jina Embeddings client with retry, rate-limit handling, and fallback."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import (
    EMBEDDING_DIM,
    JINA_API_KEY,
    JINA_BASE_URL,
    JINA_BATCH_SIZE,
    JINA_BACKOFF_INITIAL_SECONDS,
    JINA_BACKOFF_MAX_SECONDS,
    JINA_DIMENSIONS,
    JINA_EMBEDDING_TYPE,
    JINA_MAX_RETRIES,
    JINA_NORMALIZED,
    JINA_TASK_RETRIEVAL_PASSAGE,
    JINA_TASK_RETRIEVAL_QUERY,
    JINA_TASK_TEXT_MATCHING,
    JINA_TIMEOUT_SECONDS,
    JINA_TRUNCATE,
    JINA_USER_AGENT,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingRequestConfig:
    model: str
    task: str
    dimensions: int
    normalized: bool
    truncate: bool
    embedding_type: str


def _hash_embedding(text: str, dim: int) -> list[float]:
    """Deterministic lightweight fallback embedding."""
    vec = np.zeros(dim, dtype=np.float32)
    for token in _tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest, "big") % dim
        vec[idx] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec.tolist()


def _tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[0-9A-Za-zÀ-ỹ]+", (text or "").lower())


class JinaEmbeddingsClient:
    """Small client around Jina Embeddings API with local fallback."""

    def __init__(
        self,
        api_key: str = JINA_API_KEY,
        base_url: str = JINA_BASE_URL,
        batch_size: int = JINA_BATCH_SIZE,
        timeout_seconds: float = JINA_TIMEOUT_SECONDS,
        max_retries: int = JINA_MAX_RETRIES,
        backoff_initial_seconds: float = JINA_BACKOFF_INITIAL_SECONDS,
        backoff_max_seconds: float = JINA_BACKOFF_MAX_SECONDS,
        default_dimensions: int = JINA_DIMENSIONS,
        default_normalized: bool = JINA_NORMALIZED,
        default_truncate: bool = JINA_TRUNCATE,
        default_embedding_type: str = JINA_EMBEDDING_TYPE,
    ):
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.batch_size = max(1, batch_size)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(1, max_retries)
        self.backoff_initial_seconds = backoff_initial_seconds
        self.backoff_max_seconds = backoff_max_seconds
        self.default_dimensions = default_dimensions or EMBEDDING_DIM
        self.default_normalized = default_normalized
        self.default_truncate = default_truncate
        self.default_embedding_type = default_embedding_type
        self._disabled = False
        self._cache: dict[tuple[str, str, int, bool, bool, str], list[float]] = {}

    @property
    def available(self) -> bool:
        return bool(self.api_key) and not self._disabled

    def _build_request(self, texts: list[str], config: EmbeddingRequestConfig) -> urllib.request.Request:
        payload: dict[str, object] = {
            "model": config.model,
            "input": texts,
            "embedding_type": config.embedding_type,
            "normalized": config.normalized,
            "truncate": config.truncate,
        }
        if config.task:
            payload["task"] = config.task
        if config.dimensions:
            payload["dimensions"] = config.dimensions

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": JINA_USER_AGENT,
            "Authorization": f"Bearer {self.api_key}",
        }
        return urllib.request.Request(self.base_url, data=body, headers=headers, method="POST")

    def _request_embeddings(self, texts: list[str], config: EmbeddingRequestConfig) -> Optional[list[list[float]]]:
        request = self._build_request(texts, config)
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                data = json.loads(raw)
                vectors = [item["embedding"] for item in data.get("data", [])]
                if len(vectors) != len(texts):
                    raise ValueError("Jina response size mismatch")
                return vectors
            except urllib.error.HTTPError as exc:
                last_error = exc
                status = getattr(exc, "code", None)
                body = ""
                try:
                    body = exc.read().decode("utf-8", errors="ignore")
                except Exception:
                    body = ""

                if status in (401, 403):
                    self._disabled = True
                    logger.warning("Jina API authentication failed; falling back to local embeddings.")
                    return None

                if status != 429 and not (500 <= int(status or 0) < 600):
                    logger.warning("Jina API error %s: %s", status, body[:200])
                    return None

                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                if retry_after:
                    try:
                        sleep_for = float(retry_after)
                    except ValueError:
                        sleep_for = self.backoff_initial_seconds
                else:
                    sleep_for = min(
                        self.backoff_max_seconds,
                        self.backoff_initial_seconds * (2**attempt),
                    )
                sleep_for += random.uniform(0, 0.25)
                time.sleep(max(0.0, sleep_for))
            except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                sleep_for = min(
                    self.backoff_max_seconds,
                    self.backoff_initial_seconds * (2**attempt),
                )
                sleep_for += random.uniform(0, 0.25)
                time.sleep(max(0.0, sleep_for))

        logger.warning("Jina API request failed after retries: %s", last_error)
        return None

    def embed_texts(
        self,
        texts: list[str],
        *,
        model: str,
        task: str,
        dimensions: Optional[int] = None,
        normalized: Optional[bool] = None,
        truncate: Optional[bool] = None,
        embedding_type: Optional[str] = None,
    ) -> Optional[list[list[float]]]:
        """Embed a list of texts. Returns None when API is unavailable or fails."""
        if not texts:
            return []

        if not self.available:
            return None

        config = EmbeddingRequestConfig(
            model=model,
            task=task,
            dimensions=dimensions or self.default_dimensions,
            normalized=self.default_normalized if normalized is None else normalized,
            truncate=self.default_truncate if truncate is None else truncate,
            embedding_type=embedding_type or self.default_embedding_type,
        )

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            uncached_texts: list[str] = []
            batch_result: list[Optional[list[float]]] = [None] * len(batch)

            for idx, text in enumerate(batch):
                cache_key = (config.model, config.task, config.dimensions, config.normalized, config.truncate, text)
                if cache_key in self._cache:
                    batch_result[idx] = self._cache[cache_key]
                else:
                    uncached_texts.append(text)

            if not uncached_texts:
                vectors.extend([vec for vec in batch_result if vec is not None])
                continue

            batch_vectors = self._request_embeddings(uncached_texts, config)
            if batch_vectors is None:
                return None

            for text, vector in zip(uncached_texts, batch_vectors):
                cache_key = (config.model, config.task, config.dimensions, config.normalized, config.truncate, text)
                self._cache[cache_key] = vector

            for idx, text in enumerate(batch):
                cache_key = (config.model, config.task, config.dimensions, config.normalized, config.truncate, text)
                if batch_result[idx] is None:
                    batch_result[idx] = self._cache[cache_key]

            vectors.extend([vec for vec in batch_result if vec is not None])

        return vectors


def fallback_hash_embeddings(texts: list[str], dimensions: int = JINA_DIMENSIONS) -> list[list[float]]:
    """Use a deterministic local fallback when Jina API is unavailable."""
    return [_hash_embedding(text, dimensions) for text in texts]


def fallback_similarity(text_a: str, text_b: str) -> float:
    """Lightweight lexical similarity for local fallback in chunking."""
    tokens_a = set(_tokenize(text_a))
    tokens_b = set(_tokenize(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = len(tokens_a & tokens_b)
    if overlap == 0:
        return 0.0
    jaccard = overlap / len(tokens_a | tokens_b)
    coverage = overlap / min(len(tokens_a), len(tokens_b))
    return 0.65 * coverage + 0.35 * jaccard


JINA_TASK_QUERY = JINA_TASK_RETRIEVAL_QUERY
JINA_TASK_PASSAGE = JINA_TASK_RETRIEVAL_PASSAGE
JINA_TASK_TEXT_MATCH = JINA_TASK_TEXT_MATCHING
