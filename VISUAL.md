# Lab 18 — Visual Architecture & Work Breakdown

## System Architecture

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                    PRODUCTION RAG SYSTEM                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

INPUT: HR Policy Documents (3 Markdown files)
       ├─ Chính sách Nghỉ phép
       ├─ Chính sách An toàn Thông tin  
       └─ Chính sách Lương & Phúc lợi

        ↓↓↓ M1: CHUNKING ↓↓↓

    ┌──────────────────────────────────────┐
    │ Document Chunking Strategy           │
    ├──────────────────────────────────────┤
    │ ✓ Baseline: Paragraph split (500)    │
    │ • Semantic: Cosine similarity (0.85) │  ← Advanced
    │ • Hierarchical: Parent 2K / Child256 │  ← BEST PRACTICE
    │ • Structure-aware: Parse markdown    │
    └──────────────────────────────────────┘
         ↓
    Output: ~150-200 child chunks
         ↓↓↓ M5: ENRICHMENT (optional) ↓↓↓
    
    ┌──────────────────────────────────────┐
    │ Chunk Enrichment                     │
    ├──────────────────────────────────────┤
    │ • Summarize: 2-3 câu ngắn gọn       │
    │ • HyQA: Generate hypothesis questions
    │ • Contextual: Prepend context info   │
    └──────────────────────────────────────┘
         ↓
    Enriched chunks → better embedding
         ↓↓↓ M2: INDEXING & SEARCH ↓↓↓

    ┌──────────────────────────────────────┐
    │ Hybrid Search Pipeline               │
    ├──────────────────────────────────────┤
    │                                       │
    │  Query: "Bao nhiêu ngày nghỉ?"      │
    │    ├─ BM25 Search (Vietnamese)      │
    │    │  └─ Tokenize: ["bao nhiêu",... │
    │    │  └─ Top 20 keyword matches     │
    │    │                                 │
    │    ├─ Dense Search (BGE-M3)         │
    │    │  └─ Embed: [0.23, -0.45, ...]  │
    │    │  └─ Top 20 semantic matches    │
    │    │                                 │
    │    └─ RRF Fusion                    │
    │       └─ Merge & rank: Top 20       │
    │       └─ Formula: 1/(k + rank)     │
    │                                       │
    └──────────────────────────────────────┘
         ↓
    20 candidate documents (from M2)
         ↓↓↓ M3: RERANKING ↓↓↓

    ┌──────────────────────────────────────┐
    │ Cross-Encoder Reranking              │
    ├──────────────────────────────────────┤
    │ Input: 20 docs + query               │
    │ Model: bge-reranker-v2-m3            │
    │ Output: Rerank scores (0-1)          │
    │                                       │
    │ Filter irrelevant:                   │
    │  "12 ngày nghỉ" → 0.98 ✓            │
    │  "VPN 6 tháng"  → 0.15 ✗            │
    │  "Phép kết hôn"  → 0.82 ✓            │
    │                                       │
    │ Return: Top 3 → Top-3contexts       │
    └──────────────────────────────────────┘
         ↓
    3 most relevant contexts
         ↓ (Optional: LLM Generation)

    ┌──────────────────────────────────────┐
    │ LLM Generation (GPT-4o-mini)        │
    ├──────────────────────────────────────┤
    │ Prompt:                              │
    │  "Trả lời CHỈ từ context dưới đây" │
    │  Context: [3 documents]              │
    │  Question: [user query]              │
    │                                       │
    │ Output: Curated answer              │
    └──────────────────────────────────────┘
         ↓
    ANSWER: "Nhân viên được 12 ngày nghỉ phép hàng năm"
         
         ↓↓↓ M4: RAGAS EVALUATION ↓↓↓

    ┌──────────────────────────────────────┐
    │ RAGAS 4 Metrics                      │
    ├──────────────────────────────────────┤
    │                                       │
    │ 1. Faithfulness → Measure halluc     │
    │    Score: 0.78 (gốc ≥0.85 → +5 🎁) │
    │                                       │
    │ 2. Answer Relevancy → Match question │
    │    Score: 0.81 (tốt, trả lời đúng)  │
    │                                       │
    │ 3. Context Precision → No noise docs │
    │    Score: 0.75 (reranking helping)   │
    │                                       │
    │ 4. Context Recall → Have all info    │
    │    Score: 0.88 (hybrid search rocks) │
    │                                       │
    └──────────────────────────────────────┘
         ↓
    Aggregate Scores
         ↓↓↓ FAILURE ANALYSIS ↓↓↓

    ┌──────────────────────────────────────┐
    │ Bottom-5 Failing Questions           │
    ├──────────────────────────────────────┤
    │ Q: "Được bao nhiêu ngày?"            │
    │   Score: 0.52 ✗                      │
    │   Worst metric: Faithfulness (0.38)  │
    │   Diagnosis: LLM hallucinating       │
    │   Fix: Lower temperature, tighter    │
    │        prompt constraint             │
    │                                       │
    │ Q: "Thưởng năm là gì?"              │
    │   Score: 0.68 ✗                      │
    │   Worst metric: Context Recall (0.45)
    │   Diagnosis: Search missed chunks    │
    │   Fix: Improve chunking strategy     │
    │        or add more BM25              │
    │                                       │
    │ ... (3 more) ...                     │
    │                                       │
    └──────────────────────────────────────┘

OUTPUT: RAGAS Report (ragas_report.json)
        ├─ aggregate: {faithfulness, answer_relevancy, ...}
        ├─ per_question: [question, score, diagnosis, ...]
        └─ failures: [bottom-N with root causes]
```

---

## Workload Breakdown (2 hours)

```
┌─ PHẦN A: CÁ NHÂN (60 pts, 1.5 hr) ─────────────────────┐
│                                                         │
│  👤 Person 1: MODULE 1 (Chunking) — 45 min             │
│  ├─ Read: ANALYSIS.md § "Module 1" (10min)            │
│  ├─ Implement: chunk_semantic() (12min)               │
│  ├─ Implement: chunk_hierarchical() (15min)           │
│  ├─ Implement: chunk_structure_aware() (8min)         │
│  ├─ Polish: Comments + tests (5min)                   │
│  └─ pytest tests/test_m1.py ✓                          │
│                                                         │
│  👤 Person 2: MODULE 2 (Search) — 50 min              │
│  ├─ Read: ANALYSIS.md § "Module 2" (10min)            │
│  ├─ Implement: segment_vietnamese() (8min)            │
│  ├─ Implement: BM25Search (15min)                     │
│  ├─ Implement: DenseSearch (15min)                    │
│  ├─ Implement: RRF (7min)                             │
│  └─ pytest tests/test_m2.py ✓                          │
│                                                         │
│  👤 Person 3: MODULE 3 (Reranking) — 40 min           │
│  ├─ Read: ANALYSIS.md § "Module 3" (8min)             │
│  ├─ Implement: CrossEncoderReranker (20min)           │
│  ├─ Implement: benchmark_reranker() (10min)           │
│  └─ pytest tests/test_m3.py ✓                          │
│                                                         │
│  👤 Person 4: MODULE 4 (Evaluation) — 35 min          │
│  ├─ Read: ANALYSIS.md § "Module 4" (7min)             │
│  ├─ Implement: evaluate_ragas() (12min)               │
│  ├─ Implement: failure_analysis() (12min)             │
│  └─ pytest tests/test_m4.py ✓                          │
│                                                         │
└─────────────────────────────────────────────────────────┘

┌─ PHẦN B: NHÓM (40 pts, 30 min) ────────────────────────┐
│                                                         │
│  👥 All: Integrate Pipeline (10 min)                   │
│  ├─ Edit src/pipeline.py (5min)                       │
│  ├─ python main.py (3min)                             │
│  └─ Check reports/ ✓ (2min)                            │
│                                                         │
│  👥 All: Failure Analysis (10 min)                     │
│  ├─ Read ragas_report.json (2min)                     │
│  ├─ Fill analysis/failure_analysis.md (5min)          │
│  └─ Fill analysis/group_report.md (3min)              │
│                                                         │
│  👥 Each: Individual Reflection (5 min)                │
│  ├─ Write reflection_[Name].md (5min)                 │
│                                                         │
│  👥 All: Presentation (5 min prep)                     │
│  ├─ Organize findings (5min)                          │
│                                                         │
└─────────────────────────────────────────────────────────┘

TOTAL: 2 hours ✓
```

---

## Testing Strategy

```
┌─ UNIT TESTS (per module) ────────────────────────┐
│                                                  │
│  M1: pytest tests/test_m1.py -v                 │
│  ├─ test_chunk_semantic: returns list[Chunk]   │
│  ├─ test_chunk_hierarchical: parent_id valid   │
│  ├─ test_chunk_structure_aware: keeps headers  │
│  └─ test_compare_strategies: stats dict        │
│                                                  │
│  M2: pytest tests/test_m2.py -v                 │
│  ├─ test_vietnamese_segmentation: tokens work  │
│  ├─ test_bm25_search: top result has keyword   │
│  ├─ test_dense_search: loads BGE-M3            │
│  └─ test_hybrid_search: RRF merges scores      │
│                                                  │
│  M3: pytest tests/test_m3.py -v                 │
│  ├─ test_cross_encoder_rerank: scores sorted   │
│  └─ test_latency: < 5 seconds                   │
│                                                  │
│  M4: pytest tests/test_m4.py -v                 │
│  ├─ test_evaluate_ragas: 4 metrics returned    │
│  └─ test_failure_analysis: has diagnosis       │
│                                                  │
└──────────────────────────────────────────────────┘

┌─ INTEGRATION TEST (full pipeline) ──────────────┐
│                                                  │
│  python naive_baseline.py                       │
│  └─ Output: naive_baseline_report.json          │
│                                                  │
│  python main.py                                 │
│  ├─ Builds pipeline ✓                           │
│  ├─ Runs 20 test questions ✓                    │
│  ├─ Generates reports/ ✓                        │
│  └─ Prints comparison table ✓                   │
│                                                  │
│  Expected Improvement:                          │
│  ┌─────────────────────────────────────────┐    │
│  │ Attribute       │ Naive │ Prod  │ Δ     │    │
│  ├─────────────────────────────────────────┤    │
│  │ Faithfulness    │ 0.50  │ 0.78  │ +0.28 │    │
│  │ Answer Relev.   │ 0.60  │ 0.81  │ +0.21 │    │
│  │ Context Prec.   │ 0.55  │ 0.75  │ +0.20 │    │
│  │ Context Recall  │ 0.70  │ 0.88  │ +0.18 │    │
│  └─────────────────────────────────────────┘    │
│                                                  │
└──────────────────────────────────────────────────┘

┌─ FINAL CHECK ────────────────────────────────────┐
│                                                  │
│  python check_lab.py                            │
│  ├─ analysis/failure_analysis.md ✓              │
│  ├─ analysis/group_report.md ✓                  │
│  ├─ analysis/reflections/*.md ✓ (1 per person) │
│  ├─ reports/*.json ✓ (both)                     │
│  └─ All tests pass ✓                            │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## Success Metrics

```
INDIVIDUAL (60 pts):
┌──────────────────────────────────┬────────┐
│ Implementation correct           │ 15 pts │
│ Pytest pass                      │ 15 pts │
│ Vietnamese-specific logic        │ 10 pts │
│ Code quality + comments          │ 10 pts │
│ All TODOs completed              │ 10 pts │
├──────────────────────────────────┼────────┤
│ TOTAL                            │ 60 pts │
└──────────────────────────────────┴────────┘

GROUP (40 pts):
┌──────────────────────────────────┬────────┐
│ Pipeline end-to-end              │ 10 pts │
│ RAGAS ≥ 0.75 (any metric)        │ 10 pts │
│ Failure analysis + insights      │ 10 pts │
│ Presentation (5 min)             │ 10 pts │
├──────────────────────────────────┼────────┤
│ TOTAL                            │ 40 pts │
└──────────────────────────────────┴────────┘

BONUS (+10 max):
┌──────────────────────────────────┬────────┐
│ Faithfulness ≥ 0.85              │ +5 pts │
│ M5 Enrichment implemented        │ +3 pts │
│ Latency breakdown report         │ +2 pts │
└──────────────────────────────────┴────────┘
```

---

## Decision Tree: "Nên chọn Module nào?"

```
┌─ Your Experience Level?
├─ 👶 Beginner (new to ML)
│  └─> Module 1: Chunking ✓
│       • Learns text processing
│       • Tangible outputs (chunks)
│       • 200 lines, easy test
│
├─ 👨‍💻 Intermediate (know ML basics)
│  ├─> Module 2: Hybrid Search ✓
│  │    • Learn NLP + vector DB
│  │    • Vietnamese-specific tricks
│  │    • More complex (~250 lines)
│  │
│  └─> Module 3: Reranking ✓
│       • Learn about models
│       • Ranking + benchmarking
│       • Shorter (~150 lines)
│
└─ 🚀 Advanced (comfort with ML)
   ├─> Module 2 or 3 ✓
   │    • Optimize for production
   │    • Add bonus features
   │
   └─> Any module + M5 Enrichment ✓
        • Push for bonus points
```

---

## Failure Diagnosis Tree

```
Question fails? (avg score < 0.75)
│
├─ Worst metric: Faithfulness?
│  ├─ LLM making up info?
│  │  └─ Fix: Lower temperature, tighter constraint
│  │
│  └─ Context too noisy?
│     └─ Fix: Better reranking (M3)
│
├─ Worst metric: Answer Relevancy?
│  ├─ Answer off-topic?
│  │  └─ Fix: Improve system prompt
│  │
│  └─ Question ambiguous?
│     └─ Fix: Query rewriting (HyQA)
│
├─ Worst metric: Context Precision?
│  ├─ Too many irrelevant docs?
│  │  └─ Fix: Add/improve M3 reranking ← Biggest impact
│  │
│  └─ Chunking too large?
│     └─ Fix: Reduce M1 chunk size
│
└─ Worst metric: Context Recall?
   ├─ Missing relevant chunks?
   │  ├─ Fix: Better M2 search (add BM25)
   │  │
   │  └─ Fix: Better M1 chunking (semantic grouping)
   │
   └─ Query not matching?
      └─ Fix: Query expansion / HyQA
```

---

## File Editing Checklist

```
Essential edits for each person:

👤 Person 1: Module 1
├─ src/m1_chunking.py
│  ├─ chunk_semantic() — full impl
│  ├─ chunk_hierarchical() — full impl
│  ├─ chunk_structure_aware() — full impl
│  └─ compare_strategies() — full impl
│
├─ tests/test_m1.py — should pass
└─ analysis/reflections/reflection_Person1.md — write

👤 Person 2: Module 2
├─ src/m2_search.py
│  ├─ segment_vietnamese() — full impl
│  ├─ BM25Search.index() — full impl
│  ├─ BM25Search.search() — full impl
│  ├─ DenseSearch.index() — full impl
│  ├─ DenseSearch.search() — full impl
│  └─ reciprocal_rank_fusion() — full impl
│
├─ tests/test_m2.py — should pass
└─ analysis/reflections/reflection_Person2.md — write

👤 Person 3: Module 3
├─ src/m3_rerank.py
│  ├─ CrossEncoderReranker._load_model() — full impl
│  ├─ CrossEncoderReranker.rerank() — full impl
│  └─ benchmark_reranker() — full impl
│
├─ tests/test_m3.py — should pass
└─ analysis/reflections/reflection_Person3.md — write

👤 Person 4: Module 4
├─ src/m4_eval.py
│  ├─ evaluate_ragas() — full impl
│  └─ failure_analysis() — full impl
│
├─ tests/test_m4.py — should pass
└─ analysis/reflections/reflection_Person4.md — write

👥 ALL (Group integration):
├─ src/pipeline.py — edit build_pipeline() & evaluate_pipeline()
├─ analysis/failure_analysis.md — fill in bottom-5 analysis
├─ analysis/group_report.md — fill in 4 sections
├─ .env — fill in API keys
└─ tests/ — run all pytest
```

---

## Docker Quick Reference

```
# Build
docker build -t rag-lab18 .

# Run single command
docker run --rm -v $(pwd):/app rag-lab18 python main.py

# Run with compose
docker-compose up -d qdrant
docker-compose run --rm rag python main.py

# Clean up
docker-compose down
docker rmi rag-lab18
```

---

**Happy coding! 🚀**
