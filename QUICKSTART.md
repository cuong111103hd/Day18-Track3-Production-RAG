# Lab 18: Production RAG Pipeline — Tóm tắt Nhanh gọn

## 🎯 Lab này những cái gì?

**Mục tiêu**: Build 1 Production RAG system từ đầu → cuối
- Naive RAG (basic) ➜ Production RAG (advanced) ➜ So sánh RAGAS
- Implement 4 modules: Chunking, Search, Reranking, Evaluation
- Evaluate end-to-end trên 20 test questions (Q&A về HR policy)

---

## 📊 Architecture tổng quát

```
┌─────────────────────────────────────────────────────┐
│ Lab 18: Production RAG (2 giờ)                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  PHẦN A (60 điểm, 1.5h): Cá nhân chọn 1 module     │
│                                                      │
│  ┌─ M1: Chunking (3 strategies)                    │
│  │   ├─ Semantic (DL nhóm câu)                    │
│  │   ├─ Hierarchical (parent-child) ⭐            │
│  │   └─ Structure-aware (parse markdown)          │
│  │                                                 │
│  ├─ M2: Hybrid Search (BM25 + Dense + RRF)        │
│  │   ├─ Vietnamese tokenization                   │
│  │   ├─ BM25 indexing & search                    │
│  │   ├─ Dense (BGE-M3 + Qdrant)                   │
│  │   └─ RRF fusion                                │
│  │                                                 │
│  ├─ M3: Reranking (Cross-encoder)                 │
│  │   ├─ Load bge-reranker-v2-m3                   │
│  │   ├─ Score + rank top-20 → top-3              │
│  │   └─ Benchmark latency                         │
│  │                                                 │
│  └─ M4: Evaluation (RAGAS)                        │
│      ├─ Load test_set (20 Q&A)                    │
│      ├─ Run 4 metrics                             │
│      │   ├─ Faithfulness (LLM không hallucinate?)  │
│      │   ├─ Answer Relevancy (trả lời đúng?)     │
│      │   ├─ Context Precision (chunks relevant?)  │
│      │   └─ Context Recall (đủ info?)            │
│      └─ Failure analysis                          │
│                                                    │
│  PHẦN B (40 điểm, 30min): Nhóm ghép 4 modules    │
│                                                    │
│  ├─ Build pipeline.py (M1→M2→M3→M4)             │
│  ├─ Run RAGAS on pipeline                         │
│  ├─ Compare Naive vs Production                   │
│  ├─ Analyze failures (bottom-5 questions)        │
│  └─ Present 5 minutes                             │
│                                                    │
└─────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start (5 phút)

```bash
# 1. Setup
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Copy config
cp .env.example .env   # Fill in API keys

# 3. Start Qdrant
docker-compose up -d

# 4. Run baseline
python naive_baseline.py

# 5. Implement module (mỗi cá nhân làm 1)
# Edit: src/m1_chunking.py (hoặc m2, m3, m4)
# Test: pytest tests/test_m1.py

# 6. Nhóm ghép pipeline
# Edit: src/pipeline.py
python main.py    # Full pipeline + comparison
```

---

## 📁 Cấu trúc Project

```
lab18-production-rag/
│
├── 📄 README.md                    # File gốc (file này)
├── 📄 SETUP.md                     # ★ Hướng dẫn setup chi tiết
├── 📄 ANALYSIS.md                  # ★ Phân tích depp-dive (200+ dòng)
├── 📄 RUBRIC.md                    # Hệ thống chấm điểm
├── 📄 requirements.txt             # ★ Dependencies (đã có)
├── 📄 Dockerfile                   # ★ Docker image (đã có)
├── 📄 docker-compose.yml           # ★ Qdrant + RAG container (đã có)
│
├── 📄 config.py                    # Shared config
├── 📄 main.py                      # Entry: Naive + Production + Compare
├── 📄 check_lab.py                 # Kiểm tra định dạng trước nộp
├── 📄 naive_baseline.py            # Baseline (run trước)
│
├── src/                            # ★ Core implementation (TODO markers)
│   ├── __init__.py
│   ├── m1_chunking.py              # ★ M1: Chunking strategies
│   ├── m2_search.py                # ★ M2: Hybrid Search
│   ├── m3_rerank.py                # ★ M3: Reranking
│   ├── m4_eval.py                  # ★ M4: Evaluation
│   ├── m5_enrichment.py            # M5: Enrichment (bonus)
│   └── pipeline.py                 # ★ Nhóm ghép tất cả
│
├── tests/                          # Auto-grading
│   ├── __init__.py
│   ├── test_m1.py
│   ├── test_m2.py
│   ├── test_m3.py
│   └── test_m4.py
│
├── data/                           # ★ Sample documents (có sẵn 3 MD)
│   ├── sample_01.md                # Chính sách Nghỉ phép
│   ├── sample_02.md                # Chính sách An toàn Thông tin
│   ├── sample_03.md                # Chính sách Lương & Phúc lợi
│   └── BCTC.pdf / Nghi_dinh.pdf    # (Original PDF files)
│
├── test_set.json                   # ★ 20 Q&A pairs (đã mở rộng)
│
├── analysis/                       # ★ Deliverable (nhóm điền)
│   ├── failure_analysis.md         # Bottom-5 questions + diagnosis
│   ├── group_report.md             # 4 phần: scores, wins, case study, next steps
│   └── reflections/                # Individual reflections
│       ├── reflection_template.md
│       └── reflection_[Name].md    # Mỗi người viết 1 file
│
├── reports/                        # ★ Auto-generated (sau khi run)
│   ├── naive_baseline_report.json  # Naive baseline scores
│   └── ragas_report.json           # Production pipeline scores
│
├── templates/                      # Templates (backup)
│   ├── failure_analysis.md
│   └── group_report.md
│
├── .env.example                    # ★ Config template (updated)
├── .env                            # (Create from .env.example, add your keys)
│
└── .gitignore
```

---

## 🎓 Phần A: Lựa chọn & Implement (60 điểm)

### How to choose which module?

| Nếu bạn... | Chọn module | Tại sao |
|-----------|-----------|--------|
| Mới học AI, focus text | M1: Chunking | Easy, 200 dòng, nhiều concept |
| Thích Vietnamese NLP | M2: Search | Hybrid, BM25, Qdrant |
| Muốn viết model code | M3: Reranking | Cross-encoder, benchmarking |
| Mới, muốn quick win | M4: Evaluation | Ngắn, 100 dòng, tools ready |

### Phần Implementation:

1. **Mở file**: `src/m[1-4]_*.py`
2. **Tìm `# TODO:`** (2-6 TODO markers mỗi module)
3. **Implement** theo hướng dẫn comment
4. **Test**: `pytest tests/test_m*.py`
5. **Polish**: Add comments, type hints

### Example: M1 Chunking

```python
# ---- MODULE 1: Chunking ----

# Đã có sẵn:
def chunk_basic(text: str, chunk_size: int = 500):
    # baseline chunking (paragraph split)
    ...

# TODO 1: Semantic Chunking
def chunk_semantic(text: str, threshold: float = 0.85):
    """Split by sentence similarity."""
    # TODO Implementation:
    # 1. Split text → sentences
    # 2. Encode sentences
    # 3. Compare consecutive sentences (cosine_sim > threshold?)
    # 4. If < threshold → new chunk
    # 5. Return list[Chunk]
    pass

# TODO 2: Hierarchical Chunking  ⭐ PRODUCTION BEST PRACTICE
def chunk_hierarchical(text: str, parent_size: int = 2048, child_size: int = 256):
    """
    Parent (big) + Child (small) hierarchy.
    Index: children (accurate) | Retrieve: parents (context)
    """
    # TODO Implementation:
    # 1. Split into parent chunks (~2000 chars)
    # 2. For each parent, slide-window into children (~250 chars)
    # 3. Set parent_id on children
    # 4. Return (parents, children)
    pass

# Tương tự M2, M3, M4...
```

**Test locally:**
```bash
pytest tests/test_m1.py -v
# test_chunk_semantic PASSED
# test_chunk_hierarchical PASSED
# test_chunk_structure_aware PASSED
# test_compare_strategies PASSED
```

---

## 🔗 Phần B: Ghép Pipeline (40 điểm)

Sau khi 4 modules implement xong → nhóm ghép thành 1 pipeline:

```
Document → M1: Chunk → M5: Enrich → M2: Search → M3: Rerank → LLM Gen → Answer
                                                                              ↓
                                                              M4: RAGAS Eval ← Contexts
```

### Implementation Steps:

**1. Edit `src/pipeline.py`:**

```python
def build_pipeline():
    # M1: Load & Chunk
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], ...)
        all_chunks.extend(children)
    
    # M5: Enrichment (optional)
    enriched = enrich_chunks(all_chunks)
    
    # M2: Index
    search = HybridSearch()
    search.index(enriched or all_chunks)
    
    # M3: Reranker
    reranker = CrossEncoderReranker()
    
    return search, reranker

def evaluate_pipeline(search, reranker):
    test_set = load_test_set()
    
    questions, answers, contexts, ground_truths = [], [], [], []
    
    for item in test_set:
        # M2: Search
        results = search.search(item["question"], top_k=20)
        docs = [{"text": r.text, "score": r.score, ...} for r in results]
        
        # M3: Rerank
        reranked = reranker.rerank(item["question"], docs, top_k=3)
        
        # Generate answer (TODO: use LLM for better score)
        answer = reranked[0].text if reranked else "Not found"
        
        questions.append(item["question"])
        answers.append(answer)
        contexts.append([r.text for r in reranked])
        ground_truths.append(item["ground_truth"])
    
    # M4: Evaluate
    results = evaluate_ragas(questions, answers, contexts, ground_truths)
    failures = failure_analysis(results.get("per_question", []))
    
    save_report(results, failures)
    return results
```

**2. Run pipeline:**

```bash
python main.py

# Output:
# [1/3] Running Basic RAG Baseline...
# [2/3] Running Production Pipeline...
# [3/3] Comparison
#
# Metric                  Basic      Production      Δ
# ──────────────────────────────────────────────────
# ✓ faithfulness         0.5234      0.8234        +0.3000
# ✓ answer_relevancy     0.4892      0.7891        +0.3000
# ...
```

### Expected Improvements:

| Metric | Baseline | Production | Why |
|--------|----------|-----------|-----|
| Context Precision | 0.55 | 0.75 | M3: Reranking filters irrelevant |
| Context Recall | 0.70 | 0.85 | M2: Hybrid search more comprehensive |
| Faithfulness | 0.50 | 0.80 | LLM generation (M4 measures) |
| Answer Relevancy | 0.60 | 0.80 | Better prompt + context |

---

## 📋 Failure Analysis (Nhóm)

**Khi nào question bị "fail"?**

```
Score Threshold: Nếu avg(4_metrics) < 0.75 → "Fail"

Example:
Question: "Bao nhiêu ngày nghỉ phép?"
  - Faithfulness: 0.45 ✗ (LLM hallucinating)
  - Answer Relevancy: 0.80 ✓
  - Context Precision: 0.60 ✗ (irrelevant chunks)
  - Context Recall: 0.70 ✗ (missing chunks)
  - Avg: 0.64 ✗ FAIL
```

**Diagnostic Tree:**

```
Q: Tại sao failed?

├─ Faithfulness < 0.85?
│  └─ LLM Hallucinating
│     └─ Fix: "Trả lời CHỈ từ context. Không tự tạo."
│
├─ Answer Relevancy < 0.80?
│  └─ Answer không match question
│     └─ Fix: "Rephrase prompt để specific hơn"
│
├─ Context Precision < 0.75?
│  └─ Có irrelevant chunks
│     └─ Fix: "Thêm M3 Reranking hoặc metadata filter"
│
└─ Context Recall < 0.75?
   └─ Thiếu relevant chunks
      └─ Fix: "Tốt hóa M1 chunking hoặc M2 search"
```

**Template: `analysis/failure_analysis.md`**

```markdown
## Q: "Bao nhiêu ngày nghỉ phép?"

**Scores:**
- Faithfulness: 0.45 (LLM making up numbers)
- Answer Relevancy: 0.80
- Context Precision: 0.60 (got 5 docs, only 1 relevant)
- Context Recall: 0.70 (missing info about conditions)
- **Avg: 0.64** ✗ FAIL

**Root Cause:** Worst metric = Faithfulness (0.45)
- LLM generated "15 ngày" (không có trong context)

**Suggested Fix:**
1. System prompt: "Trả lời CHỈ dựa trên context provided"
2. Temperature: 0 (deterministic)
3. Add constraint: "If not in context → say 'Cannot answer'"

---

## Q: "VPN certificate khi nào hết hạn?"

[Tương tự...]
```

**`analysis/group_report.md`** (4 phần)

```markdown
# Group Report

## 1. RAGAS Scores Comparison

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.50 | 0.78 | +0.28 |
| ...

## 2. Biggest Win

**Module M2 (Hybrid Search)** improved Context Recall from 0.70 → 0.85.
- Before: Dense-only search missed keyword-heavy queries
- After: BM25 catches keyword queries, Dense catches semantic
- RRF merges both → best of both worlds

## 3. Case Study: Question XYZ

[Deep-dive into 1 question]

## 4. Next Steps

If we had +1 hour:
1. Implement M5 Enrichment (contextual prepend)
2. Train on domain-specific data
3. Add caching for 10x speedup
```

---

## ✅ Checklist Trước nộp

**Toàn bộ nhóm:**

```bash
# Sau mỗi bước, run:
pytest tests/ -v              # All tests pass (hoặc mostly pass)
python naive_baseline.py      # Baseline scores
python main.py                # Full pipeline

# Check files:
ls -R analysis/               # failure_analysis.md, group_report.md
ls -R reports/                # naive_baseline_report.json, ragas_report.json
python check_lab.py           # Format validation
```

**Cá nhân:**

- [ ] Module implement (all TODO done)
- [ ] Tests pass
- [ ] Code với comments + type hints
- [ ] `analysis/reflections/reflection_[Name].md` viết xong

---

## 🚀 Bonus Points (+10 max)

| Bonus | +Points | Điều kiện |
|-------|---------|----------|
| **Faithfulness ≥ 0.85** | +5 | Good prompt + low temperature |
| **M5 Enrichment** | +3 | Contextual prepend hoặc HyQA |
| **Latency Breakdown** | +2 | Report thời gian từng bước |

---

## 📚 Files to Read First

1. **[SETUP.md](SETUP.md)** — Step-by-step setup & installation
2. **[ANALYSIS.md](ANALYSIS.md)** — Deep-dive vào mỗi module (200+ dòng)
3. **[ASSIGNMENT_INDIVIDUAL.md](ASSIGNMENT_INDIVIDUAL.md)** — Chi tiết Phần A
4. **[ASSIGNMENT_GROUP.md](ASSIGNMENT_GROUP.md)** — Chi tiết Phần B
5. **[RUBRIC.md](RUBRIC.md)** — Hệ thống chấm điểm

---

## 🎯 Success Criteria

| Tiêu chí | Cá nhân | Nhóm |
|----------|--------|------|
| **Minimum** | Implement M1-4 module được | Pipeline chạy + RAGAS ≥0.60 |
| **Good** | Tests pass + type hints | RAGAS ≥0.75 + failure analysis |
| **Excellent** | Clean code + bonuses | RAGAS ≥0.85 + insights |

---

## 💬 Resources

- **RAGAS Docs**: https://docs.ragas.io/
- **BGE-M3**: https://huggingface.co/BAAI/bge-m3
- **Qdrant**: https://qdrant.tech/
- **Underthesea**: https://underthesea.readthedocs.io/

---

**Good luck! 🎉**

*Last updated: 2024-05-04*
