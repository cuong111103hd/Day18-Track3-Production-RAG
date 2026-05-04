"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import glob
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD, SEMANTIC_EMBEDDING_MODEL,
                    JINA_DIMENSIONS, JINA_NORMALIZED, JINA_TRUNCATE)
from src.jina_embeddings import JinaEmbeddingsClient, JINA_TASK_TEXT_MATCH, fallback_similarity

_JINA_CLIENT = JinaEmbeddingsClient()


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}
    metadata = metadata or {}
    # 1. Split text into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []

    # 2. Encode sentences with Jina API; fall back to lexical similarity when unavailable.
    embeddings = _JINA_CLIENT.embed_texts(
        sentences,
        model=SEMANTIC_EMBEDDING_MODEL,
        task=JINA_TASK_TEXT_MATCH,
        dimensions=JINA_DIMENSIONS,
        normalized=JINA_NORMALIZED,
        truncate=JINA_TRUNCATE,
    )

    use_embeddings = embeddings is not None

    if use_embeddings:
        import numpy as np

        def cosine_sim(a, b):
            return float(np.dot(a, b))
    else:
        def cosine_sim(a, b):
            return fallback_similarity(a, b)

    # 3. Group sentences by similarity
    chunks = []
    current_group = [sentences[0]]

    for i in range(1, len(sentences)):
        if use_embeddings and embeddings is not None:
            sim = cosine_sim(embeddings[i - 1], embeddings[i])
        else:
            sim = cosine_sim(sentences[i - 1], sentences[i])
        if sim < threshold:
            chunks.append(Chunk(
                text=" ".join(current_group),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
            ))
            current_group = []
        current_group.append(sentences[i])

    if current_group:
        chunks.append(Chunk(
            text=" ".join(current_group),
            metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
        ))

    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    metadata = metadata or {}
    parents = []
    children = []

    # 1. Split text into parents (simple paragraph grouping)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    current_parent_text = ""
    p_index = 0

    for para in paragraphs:
        if len(current_parent_text) + len(para) > parent_size and current_parent_text:
            pid = f"{metadata.get('source', 'doc')}_p{p_index}"
            parent_chunk = Chunk(text=current_parent_text.strip(), metadata={**metadata, "chunk_type": "parent", "parent_id": pid})
            parents.append(parent_chunk)

            # 2. Split parent into children
            for c_idx in range(0, len(current_parent_text), child_size):
                child_text = current_parent_text[c_idx:c_idx + child_size].strip()
                if child_text:
                    children.append(Chunk(
                        text=child_text,
                        metadata={**metadata, "chunk_type": "child"},
                        parent_id=pid
                    ))

            current_parent_text = ""
            p_index += 1
        current_parent_text += para + "\n\n"

    if current_parent_text.strip():
        pid = f"{metadata.get('source', 'doc')}_p{p_index}"
        parents.append(Chunk(text=current_parent_text.strip(), metadata={**metadata, "chunk_type": "parent", "parent_id": pid}))
        for c_idx in range(0, len(current_parent_text), child_size):
            child_text = current_parent_text[c_idx:c_idx + child_size].strip()
            if child_text:
                children.append(Chunk(text=child_text, metadata={**metadata, "chunk_type": "child"}, parent_id=pid))

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}
    metadata = metadata or {}
    # 1. Split by markdown headers (H1, H2, H3)
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)

    chunks = []
    current_header = "Intro"
    current_content = ""

    for part in sections:
        if not part.strip():
            continue
        if re.match(r'^#{1,3}\s+', part):
            if current_content.strip():
                chunks.append(Chunk(
                    text=f"{current_header}\n{current_content}".strip(),
                    metadata={**metadata, "section": current_header, "strategy": "structure"}
                ))
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part

    if current_content.strip():
        chunks.append(Chunk(
            text=f"{current_header}\n{current_content}".strip(),
            metadata={**metadata, "section": current_header, "strategy": "structure"}
        ))

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    results = {}
    for strategy in ["basic", "semantic", "hierarchical", "structure"]:
        all_chunks = []
        for doc in documents:
            if strategy == "basic":
                all_chunks.extend(chunk_basic(doc["text"], metadata=doc["metadata"]))
            elif strategy == "semantic":
                all_chunks.extend(chunk_semantic(doc["text"], metadata=doc["metadata"]))
            elif strategy == "hierarchical":
                p, c = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
                all_chunks.extend(c) # Compare based on children (retrieval units)
            elif strategy == "structure":
                all_chunks.extend(chunk_structure_aware(doc["text"], metadata=doc["metadata"]))

        if not all_chunks:
            continue

        lengths = [len(c.text) for c in all_chunks]
        results[strategy] = {
            "num_chunks": len(all_chunks),
            "avg_len": int(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths)
        }

    # Print table
    print(f"{'Strategy':<15} | {'Chunks':<6} | {'Avg Len':<8} | {'Min':<5} | {'Max':<5}")
    print("-" * 50)
    for name, stats in results.items():
        print(f"{name:<15} | {stats['num_chunks']:<6} | {stats['avg_len']:<8} | {stats['min_len']:<5} | {stats['max_len']:<5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
