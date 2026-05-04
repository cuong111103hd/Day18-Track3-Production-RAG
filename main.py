"""
Lab 18: Production RAG Pipeline — Main Entry Point
===================================================
Chạy toàn bộ pipeline: naive baseline → production → so sánh → report.

Usage:
    python main.py
"""

import json
import os
import time

from config import REPORT_DIR, RAGAS_REPORT_PATH, NAIVE_REPORT_PATH, PASS_THRESHOLD, EVAL_METRICS


def main():
    print("=" * 60)
    print("LAB 18: PRODUCTION RAG PIPELINE")
    print("=" * 60)
    start = time.time()

    os.makedirs(REPORT_DIR, exist_ok=True)

    # Step 1: Basic Baseline
    print("\n📌 STEP 1: Running Basic RAG Baseline...")
    print("-" * 40)
    from naive_baseline import main as run_baseline
    run_baseline()

    # Step 2: Production Pipeline
    print("\n📌 STEP 2: Running Production Pipeline...")
    print("-" * 40)
    from src.pipeline import build_pipeline, evaluate_pipeline
    search, reranker = build_pipeline()
    prod_results = evaluate_pipeline(search, reranker)

    # Move reports to reports/
    for f in [os.path.basename(RAGAS_REPORT_PATH), os.path.basename(NAIVE_REPORT_PATH)]:
        if os.path.exists(f):
            os.rename(f, os.path.join(REPORT_DIR, f))

    # Step 3: Comparison
    print("\n📌 STEP 3: Comparison")
    print("-" * 40)
    naive_path = NAIVE_REPORT_PATH
    prod_path = RAGAS_REPORT_PATH

    if os.path.exists(naive_path) and os.path.exists(prod_path):
        with open(naive_path, encoding="utf-8") as f:
            naive = json.load(f)
        with open(prod_path, encoding="utf-8") as f:
            prod = json.load(f)

        print(f"\n{'Metric':<25} {'Basic':>8} {'Production':>12} {'Δ':>8}")
        print("-" * 55)
        for m in EVAL_METRICS:
            n = naive.get("aggregate", {}).get(m, 0)
            p = prod.get("aggregate", {}).get(m, 0)
            d = p - n
            status = "✓" if p >= PASS_THRESHOLD else " "
            print(f"{status} {m:<23} {n:>8.4f} {p:>12.4f} {d:>+8.4f}")

    elapsed = time.time() - start
    print(f"\n⏱️  Total time: {elapsed:.1f}s")
    print("\n📋 Next steps:")
    print("  1. Điền analysis/failure_analysis.md")
    print("  2. Điền analysis/group_report.md")
    print("  3. Viết analysis/reflections/reflection_[Tên].md")
    print("  4. Chạy: python check_lab.py")


if __name__ == "__main__":
    main()
