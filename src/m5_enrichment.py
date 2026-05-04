"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os
import re
import sys
from dataclasses import dataclass
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEFAULT_ENRICHMENT_METHODS,
    ENRICHMENT_DEFAULT_QUESTIONS,
    ENRICHMENT_SUMMARY_MAX_CHARS,
    ENRICHMENT_SUMMARY_SHORT_MAX_CHARS,
)


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


_VI_STOPWORDS = {
    "và", "là", "của", "cho", "trong", "một", "những", "các", "được",
    "có", "theo", "với", "từ", "này", "đó", "khi", "thì", "lại", "ra",
    "về", "để", "ở", "như", "nên", "do", "đó", "nếu", "hoặc", "đến",
    "nhân", "viên", "thông", "tin", "sau", "trước", "đó", "đoạn", "văn",
    "this", "that", "the", "and", "or", "for", "with", "from", "into",
}


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences while keeping Vietnamese punctuation simple."""
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def _tokenize(text: str) -> list[str]:
    """Tokenize text with a lightweight offline regex tokenizer."""
    if not text:
        return []
    return re.findall(r"[0-9A-Za-zÀ-ỹ]+", text.lower())


def _content_words(text: str) -> list[str]:
    """Return content words by removing a small set of stopwords."""
    return [tok for tok in _tokenize(text) if tok not in _VI_STOPWORDS and len(tok) > 1]


def _unique_keep_order(items: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        key = item.strip()
        if key and key.lower() not in seen:
            ordered.append(key)
            seen.add(key.lower())
    return ordered


def _join_with_limit(
    sentences: list[str],
    max_chars: int = ENRICHMENT_SUMMARY_MAX_CHARS,
) -> str:
    """Join sentences until a character budget is reached."""
    joined = []
    total = 0
    for sentence in sentences:
        if not sentence:
            continue
        next_total = total + len(sentence) + (1 if joined else 0)
        if joined and next_total > max_chars:
            break
        joined.append(sentence.rstrip("."))
        total = next_total
    if not joined and sentences:
        joined.append(sentences[0].rstrip("."))
    result = ". ".join(joined).strip()
    if result and not result.endswith((".", "!", "?")):
        result += "."
    return result


def _extract_salient_topic(text: str) -> str:
    words = _content_words(text)
    if not words:
        return "nội dung chung"
    counts = Counter(words)
    top_words = [word for word, _ in counts.most_common(3)]
    return " / ".join(top_words)


def _detect_language(text: str) -> str:
    if re.search(r"[àáảãạăâđêôơưèéẻẽẹìíỉĩịòóỏõọùúủũụỳýỷỹ]", text.lower()):
        return "vi"
    common_vi = {"và", "là", "của", "cho", "nhân", "viên", "được", "nghỉ"}
    tokens = set(_tokenize(text))
    return "vi" if len(tokens & common_vi) >= 2 else "en"


def _infer_category(text: str) -> str:
    lowered = text.lower()
    rules = [
        ("hr", ["nhân viên", "nghỉ phép", "thử việc", "lương", "phòng nhân sự", "benefit"]),
        ("it", ["mật khẩu", "password", "vpn", "server", "access", "token", "wireguard"]),
        ("policy", ["nghị định", "quy định", "chính sách", "tuân thủ", "điều khoản", "policy"]),
        ("finance", ["hóa đơn", "invoice", "thanh toán", "payment", "ngân sách", "budget"]),
        ("legal", ["luật", "pháp luật", "điều", "khoản", "decree", "article"]),
    ]
    scores = {name: 0 for name, _ in rules}
    for category, keywords in rules:
        for kw in keywords:
            if kw in lowered:
                scores[category] += 1
    best_category = max(scores, key=scores.get)
    return best_category if scores[best_category] > 0 else "general"


def _extract_entities(text: str) -> list[str]:
    entities = []
    # Multi-word capitalized phrases, e.g. "VinUni 2024"
    entities.extend(re.findall(r"(?:[A-ZÀ-Ỹ][\wÀ-ỹ-]*)(?:\s+(?:[A-ZÀ-Ỹ][\wÀ-ỹ-]*|[0-9]{2,4}))+",
                                text))
    # Acronyms
    entities.extend(re.findall(r"\b[A-Z]{2,}\b", text))
    # Dates / numbers with units
    entities.extend(re.findall(r"\b\d+(?:[./]\d+)*(?:\s?(?:ngày|tháng|năm|giờ|phút|%))?\b", text))
    return _unique_keep_order([entity.strip() for entity in entities if entity.strip()])


def _build_questions(text: str, n_questions: int) -> list[str]:
    """Generate simple hypothesis questions from salient cues in the text."""
    lowered = text.lower()
    topic = _extract_salient_topic(text)
    questions: list[str] = []

    # Number / quantity cues
    number_match = re.search(r"\b\d+(?:[./]\d+)?\b", text)
    if number_match:
        if "ngày" in lowered:
            questions.append(f"Nhân viên được hưởng bao nhiêu ngày theo nội dung về {topic}?")
        elif "năm" in lowered or "tháng" in lowered:
            questions.append(f"Quy định nào được nêu cho mốc thời gian trong đoạn về {topic}?")
        else:
            questions.append(f"Con số nào là quan trọng nhất trong đoạn về {topic}?")

    # Policy / action cues
    if any(word in lowered for word in ["phải", "cần", "được", "không được", "cho phép", "phê duyệt"]):
        questions.append(f"Quy định hoặc yêu cầu nào được nêu trong đoạn về {topic}?")

    if "?" not in " ".join(questions):
        questions.append(f"Đoạn văn này nói về nội dung gì liên quan đến {topic}?")

    if "nghỉ phép" in lowered or "leave" in lowered:
        questions.append("Nhân viên được nghỉ phép bao nhiêu ngày mỗi năm?")

    if "mật khẩu" in lowered or "password" in lowered:
        questions.append("Chính sách mật khẩu được nêu như thế nào?")

    if "vpn" in lowered or "wireguard" in lowered:
        questions.append("Hệ thống hoặc công nghệ nào được đề cập trong đoạn này?")

    if "ai" not in lowered and any(word.isdigit() for word in text):
        questions.append(f"Thông tin số liệu nào cần nhớ về {topic}?")

    questions = _unique_keep_order([q.strip() for q in questions if q.strip()])
    if not questions:
        questions = [f"Đoạn văn này đề cập đến điều gì về {topic}?"]

    while len(questions) < n_questions:
        questions.append(f"Đoạn văn này có thể trả lời câu hỏi nào về {topic}?")

    return questions[:n_questions]


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.

    Args:
        text: Raw chunk text.

    Returns:
        Summary string (2-3 câu).
    """
    sentences = _split_sentences(text)
    if not sentences:
        return ""

    # Offline extractive fallback: prefer the first two informative sentences.
    summary = _join_with_limit(sentences[:3], max_chars=ENRICHMENT_SUMMARY_MAX_CHARS)
    if summary and len(summary) >= len(text):
        summary = _join_with_limit(
            sentences[:2],
            max_chars=ENRICHMENT_SUMMARY_SHORT_MAX_CHARS,
        )
    return summary or sentences[0]


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(
    text: str,
    n_questions: int = ENRICHMENT_DEFAULT_QUESTIONS,
) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).

    Args:
        text: Raw chunk text.
        n_questions: Số câu hỏi cần generate.

    Returns:
        List of question strings.
    """
    # Prefer a deterministic offline generator so the lab still works
    # when the OpenAI API is not available.
    questions = _build_questions(text, n_questions=max(1, n_questions))
    return questions[:n_questions]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).

    Args:
        text: Raw chunk text.
        document_title: Tên document gốc.

    Returns:
        Text với context prepended.
    """
    summary = summarize_chunk(text)
    topic = _extract_salient_topic(text)

    header_bits = []
    if document_title.strip():
        header_bits.append(f"Trích từ {document_title.strip()}")
    if summary:
        header_bits.append(f"Đoạn này nói về {topic}: {summary}")
    elif topic:
        header_bits.append(f"Đoạn này nói về {topic}")

    if header_bits:
        prefix = ". ".join(header_bits).strip()
        if not prefix.endswith("."):
            prefix += "."
        return f"{prefix}\n\n{text}"
    return text


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.

    Args:
        text: Raw chunk text.

    Returns:
        Dict with extracted metadata fields.
    """
    topic = _extract_salient_topic(text)
    entities = _extract_entities(text)
    language = _detect_language(text)
    category = _infer_category(text)

    dates = re.findall(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b|\b\d{4}\b", text)
    keywords = _unique_keep_order(_content_words(text))[:8]

    metadata = {
        "topic": topic,
        "entities": entities,
        "category": category,
        "language": language,
        "keywords": keywords,
    }
    if dates:
        metadata["date_range"] = dates[:3]
    return metadata


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: List of methods to apply. Default: ["contextual", "hyqa", "metadata"]
                 Options: "summary", "hyqa", "contextual", "metadata", "full"

    Returns:
        List of EnrichedChunk objects.
    """
    if methods is None:
        methods = list(DEFAULT_ENRICHMENT_METHODS)

    method_set = set(methods)
    if "full" in method_set:
        method_set.update({"summary", "hyqa", "contextual", "metadata"})

    enriched: list[EnrichedChunk] = []

    for chunk in chunks:
        text = chunk["text"]
        metadata = dict(chunk.get("metadata", {}))

        summary = summarize_chunk(text) if "summary" in method_set else ""
        questions = generate_hypothesis_questions(text) if "hyqa" in method_set else []
        enriched_text = contextual_prepend(text, metadata.get("source", "")) if "contextual" in method_set else text
        auto_meta = extract_metadata(text) if "metadata" in method_set else {}

        merged_meta = {**metadata, **auto_meta}
        if summary:
            merged_meta["summary"] = summary
        if questions:
            merged_meta["hypothesis_questions"] = questions

        applied_methods = [m for m in ["summary", "hyqa", "contextual", "metadata"] if m in method_set]
        method_label = "+".join(applied_methods) if applied_methods else "raw"

        enriched.append(
            EnrichedChunk(
                original_text=text,
                enriched_text=enriched_text,
                summary=summary,
                hypothesis_questions=questions,
                auto_metadata=merged_meta,
                method=method_label,
            )
        )

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
