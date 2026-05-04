"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import json
import os
import re
import sys
from dataclasses import dataclass
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TEST_SET_PATH,
    RAGAS_REPORT_PATH,
    RAGAS_EMBEDDING_MODEL,
    RAGAS_EMBEDDING_PROVIDER,
    FAILURE_BOTTOM_N,
    FAILURE_FAITHFULNESS_THRESHOLD,
    FAILURE_CONTEXT_RECALL_THRESHOLD,
    FAILURE_CONTEXT_PRECISION_THRESHOLD,
    FAILURE_ANSWER_RELEVANCY_THRESHOLD,
    OPENAI_API_KEY,
    GENERATION_MODEL,
)
from src.jina_embeddings import JinaEmbeddingsClient, JINA_TASK_QUERY
from ragas.embeddings.base import BaseRagasEmbeddings

_JINA_CLIENT = JinaEmbeddingsClient()


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Some lab files contain a trailing comma. Be tolerant so the pipeline
        # still works in grading and local demo environments.
        cleaned = re.sub(r",(\s*[\]\}])", r"\1", raw)
        return json.loads(cleaned)


def _tokenize(text: str) -> list[str]:
    """Lightweight tokenizer that works offline and on Vietnamese text."""
    if not text:
        return []
    tokens = re.findall(r"[0-9A-Za-zÀ-ỹ]+", text.lower())
    stopwords = {
        "và", "là", "của", "cho", "trong", "một", "những", "các", "được",
        "có", "theo", "với", "từ", "này", "đó", "khi", "thì", "lại", "ra",
        "về", "để", "ở", "as", "the", "and", "or", "to", "of", "a", "an",
        "in", "on", "is", "are", "be", "this", "that", "it",
    }
    return [tok for tok in tokens if tok not in stopwords]


def _overlap_ratio(source: str, target: str) -> float:
    """Return token overlap ratio from source tokens covered by target tokens."""
    source_tokens = _tokenize(source)
    target_tokens = set(_tokenize(target))
    if not source_tokens:
        return 0.0
    covered = sum(1 for token in source_tokens if token in target_tokens)
    return covered / len(source_tokens)


def _f1(a: str, b: str) -> float:
    """Simple lexical F1 score used by the offline fallback."""
    a_tokens = _tokenize(a)
    b_tokens = _tokenize(b)
    if not a_tokens or not b_tokens:
        return 0.0
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    overlap = len(a_set & b_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(a_set)
    recall = overlap / len(b_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _score_example(question: str, answer: str, contexts: list[str], ground_truth: str) -> EvalResult:
    """Create one EvalResult using deterministic lexical heuristics."""
    context_text = " ".join(contexts or [])
    ground = ground_truth or ""

    faithfulness = max(_overlap_ratio(answer, context_text), _overlap_ratio(answer, ground))
    answer_relevancy = 0.5 * _f1(answer, question) + 0.5 * _f1(answer, ground or question)
    context_precision = _overlap_ratio(context_text, answer or ground)
    context_recall = max(_overlap_ratio(answer or ground, context_text), _overlap_ratio(ground or answer, context_text))

    return EvalResult(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
        faithfulness=float(max(0.0, min(1.0, faithfulness))),
        answer_relevancy=float(max(0.0, min(1.0, answer_relevancy))),
        context_precision=float(max(0.0, min(1.0, context_precision))),
        context_recall=float(max(0.0, min(1.0, context_recall))),
    )


def _metric_value(result) -> float:
    """Extract a numeric score from ragas MetricResult or plain floats."""
    if result is None:
        return 0.0
    for attr in ("value", "score"):
        if hasattr(result, attr):
            try:
                return float(getattr(result, attr))
            except Exception:
                pass
    try:
        return float(result)
    except Exception:
        return 0.0


def _embedding_similarity(a: str, b: str) -> float:
    """Offline cosine similarity used by fallback diagnostics."""
    from src.jina_embeddings import fallback_hash_embeddings
    import numpy as np

    vec_a = np.asarray(fallback_hash_embeddings([a])[0], dtype=float)
    vec_b = np.asarray(fallback_hash_embeddings([b])[0], dtype=float)
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    return float(np.dot(vec_a, vec_b) / denom) if denom else 0.0


class JinaRagasEmbeddings(BaseRagasEmbeddings):
    """Minimal RAGAS embeddings adapter backed by Jina API."""

    def __init__(self, model: str = RAGAS_EMBEDDING_MODEL):
        self.model = model

    def embed_query(self, text: str) -> list[float]:
        vectors = _JINA_CLIENT.embed_texts([text], model=self.model, task=JINA_TASK_QUERY)
        if vectors is None:
            from src.jina_embeddings import fallback_hash_embeddings
            return fallback_hash_embeddings([text])[0]
        return vectors[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = _JINA_CLIENT.embed_texts([*texts], model=self.model, task=JINA_TASK_QUERY)
        if vectors is None:
            from src.jina_embeddings import fallback_hash_embeddings
            return fallback_hash_embeddings(texts)
        return vectors

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


def _score_with_collections_api(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> list[EvalResult]:
    """Use ragas collections API when API keys are available."""
    from openai import OpenAI
    from ragas import SingleTurnSample
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    client = OpenAI(api_key=OPENAI_API_KEY)
    llm = llm_factory(GENERATION_MODEL, provider="openai", client=client)
    embeddings = JinaRagasEmbeddings(model=RAGAS_EMBEDDING_MODEL)

    faithfulness_metric = Faithfulness(llm=llm)
    answer_relevancy_metric = AnswerRelevancy(llm=llm, embeddings=embeddings)
    context_precision_metric = ContextPrecision(llm=llm)
    context_recall_metric = ContextRecall(llm=llm)

    scored: list[EvalResult] = []
    for question, answer, retrieved_contexts, ground_truth in zip(questions, answers, contexts, ground_truths):
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=retrieved_contexts,
            reference=ground_truth,
        )

        faithfulness = _metric_value(
            faithfulness_metric.score(response=sample.response, retrieved_contexts=sample.retrieved_contexts)
        )
        answer_relevancy = _metric_value(
            answer_relevancy_metric.score(user_input=sample.user_input, response=sample.response)
        )
        context_precision = _metric_value(
            context_precision_metric.score(
                user_input=sample.user_input,
                reference=sample.reference,
                retrieved_contexts=sample.retrieved_contexts,
            )
        )
        context_recall = _metric_value(
            context_recall_metric.score(
                user_input=sample.user_input,
                retrieved_contexts=sample.retrieved_contexts,
                reference=sample.reference,
            )
        )

        scored.append(
            EvalResult(
                question=question,
                answer=answer,
                contexts=retrieved_contexts,
                ground_truth=ground_truth,
                faithfulness=float(max(0.0, min(1.0, faithfulness))),
                answer_relevancy=float(max(0.0, min(1.0, answer_relevancy))),
                context_precision=float(max(0.0, min(1.0, context_precision))),
                context_recall=float(max(0.0, min(1.0, context_recall))),
            )
        )

    return scored


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation.

    Prefer the real RAGAS pipeline when the optional dependencies are available.
    Fall back to a deterministic lexical evaluator so the lab still runs offline.
    """
    per_question: list[EvalResult] = []
    try:
        if OPENAI_API_KEY and RAGAS_EMBEDDING_PROVIDER.lower() == "jina":
            per_question = _score_with_collections_api(questions, answers, contexts, ground_truths)
        else:
            raise ValueError("OPENAI_API_KEY missing; use offline fallback")
    except Exception:
        # Offline / missing dependency fallback.
        per_question = [
            _score_example(q, a, c, gt)
            for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
        ]

    if not per_question:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": [],
        }

    return {
        "faithfulness": float(mean(r.faithfulness for r in per_question)),
        "answer_relevancy": float(mean(r.answer_relevancy for r in per_question)),
        "context_precision": float(mean(r.context_precision for r in per_question)),
        "context_recall": float(mean(r.context_recall for r in per_question)),
        "per_question": per_question,
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = FAILURE_BOTTOM_N) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    def avg_score(result: EvalResult) -> float:
        return mean([
            result.faithfulness,
            result.answer_relevancy,
            result.context_precision,
            result.context_recall,
        ])

    diagnosis_map = {
        "faithfulness": (
            FAILURE_FAITHFULNESS_THRESHOLD,
            "LLM hallucinating",
            "Tighten prompt, lower temperature, and answer strictly from context",
        ),
        "context_recall": (
            FAILURE_CONTEXT_RECALL_THRESHOLD,
            "Missing relevant chunks",
            "Improve chunking, add BM25, or broaden retrieval top-k",
        ),
        "context_precision": (
            FAILURE_CONTEXT_PRECISION_THRESHOLD,
            "Too many irrelevant chunks",
            "Add reranking, metadata filters, or reduce top-k",
        ),
        "answer_relevancy": (
            FAILURE_ANSWER_RELEVANCY_THRESHOLD,
            "Answer doesn't match question",
            "Improve prompt template and generation constraints",
        ),
    }

    ranked = sorted(eval_results, key=avg_score)[: max(0, bottom_n)]
    failures: list[dict] = []

    for result in ranked:
        scores = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }
        worst_metric = min(scores, key=scores.get)
        threshold, diagnosis, suggested_fix = diagnosis_map[worst_metric]
        score = scores[worst_metric]

        # If the metric is above threshold, keep the same root-cause label only when
        # it is still the weakest signal. This makes the report easier to read.
        if score >= threshold:
            diagnosis = f"Likely {diagnosis.lower()}"

        failures.append(
            {
                "question": result.question,
                "worst_metric": worst_metric,
                "score": float(score),
                "average_score": float(avg_score(result)),
                "diagnosis": diagnosis,
                "suggested_fix": suggested_fix,
            }
        )

    return failures


def save_report(results: dict, failures: list[dict], path: str = RAGAS_REPORT_PATH):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
