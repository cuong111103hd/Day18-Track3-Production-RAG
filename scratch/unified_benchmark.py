import os, sys, json, time
import numpy as np
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_basic, chunk_structure_aware, chunk_hierarchical
from src.m2_search import HybridSearch, BM25Search, DenseSearch
from src.m3_rerank import CrossEncoderReranker

def run_benchmark():
    # 1. Setup
    print("🚀 Initializing Benchmark...")
    docs = load_documents()
    with open("scratch/golden_test_set.json", encoding="utf-8") as f:
        test_set = json.load(f)
    
    reranker = CrossEncoderReranker()
    
    results = []

    # --- Config 1: Baseline (Basic + BM25) ---
    print("\n--- Running Config 1: Basic Chunking + BM25 ---")
    basic_chunks = []
    for d in docs:
        basic_chunks.extend(chunk_basic(d["text"], metadata=d["metadata"]))
    
    bm25_only = BM25Search()
    bm25_only.index([{"text": c.text, "metadata": c.metadata} for c in basic_chunks])
    
    results.append(evaluate_config("Baseline (Basic+BM25)", bm25_only, None, test_set))

    # --- Config 2: Step 1 (+M1 Structure-Aware) ---
    print("\n--- Running Config 2: Structure-Aware + BM25 ---")
    struct_chunks = []
    for d in docs:
        struct_chunks.extend(chunk_structure_aware(d["text"], metadata=d["metadata"]))
    
    bm25_struct = BM25Search()
    bm25_struct.index([{"text": c.text, "metadata": c.metadata} for c in struct_chunks])
    
    results.append(evaluate_config("Step 1 (M1+BM25)", bm25_struct, None, test_set))

    # --- Config 3: Step 2 (+M2 Hybrid) ---
    print("\n--- Running Config 3: Structure-Aware + Hybrid ---")
    hybrid_search = HybridSearch()
    hybrid_search.index([{"text": c.text, "metadata": c.metadata} for c in struct_chunks])
    
    results.append(evaluate_config("Step 2 (M1+M2 Hybrid)", hybrid_search, None, test_set))

    # --- Config 4: Step 3 (+M3 Rerank) ---
    print("\n--- Running Config 4: Structure-Aware + Hybrid + Rerank ---")
    results.append(evaluate_config("Step 3 (M1+M2+M3)", hybrid_search, reranker, test_set))

    # 2. Summary
    print("\n" + "="*80)
    print("🏆 UNIFIED RETRIEVAL BENCHMARK RESULTS")
    print("="*80)
    print(tabulate(results, headers="keys", tablefmt="grid"))

def evaluate_config(name, searcher, reranker, test_set):
    hit_at_3 = 0
    hit_at_5 = 0
    latencies = []
    
    for item in test_set:
        start = time.perf_counter()
        
        # Retrieval
        hits = searcher.search(item["question"], top_k=10)
        
        # Rerank if available
        if reranker:
            # Convert hits to dict format for reranker
            doc_dicts = [{"text": h.text, "score": h.score, "metadata": h.metadata} for h in hits]
            reranked = reranker.rerank(item["question"], doc_dicts, top_k=5)
            final_texts = [r.text for r in reranked]
        else:
            final_texts = [h.text for h in hits]
            
        latencies.append((time.perf_counter() - start) * 1000)
        
        # Check Hit (simple substring match for ground truth context)
        gt_context = item["context_search"]
        if any(gt_context in t for t in final_texts[:3]):
            hit_at_3 += 1
        if any(gt_context in t for t in final_texts[:5]):
            hit_at_5 += 1
            
    return {
        "Configuration": name,
        "Hit@3": f"{hit_at_3/len(test_set):.1%}",
        "Hit@5": f"{hit_at_5/len(test_set):.1%}",
        "Avg Latency (ms)": f"{np.mean(latencies):.2f}"
    }

if __name__ == "__main__":
    run_benchmark()
