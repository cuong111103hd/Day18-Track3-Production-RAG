# Lab 18: Production RAG Pipeline — Phân tích Chi tiết

**Lab này**: Day 18 · Production RAG  
**Thời gian**: 2 giờ (1.5h cá nhân + 30min nhóm)  
**Điểm**: 100 (60 cá nhân + 40 nhóm)

---

## 📋 Tương Quan Công việc

```
┌─────────────────────────────────────────────────────┐
│  LAB 18: PRODUCTION RAG PIPELINE                     │
├──────────────────────────────────────────────────────┤
│                                                       │
│  PHẦN A: CÁ NHÂN (60 điểm • 1.5 giờ)               │
│  ├─ M1: Chunking           ✓ Implement 3 strategies│
│  ├─ M2: Hybrid Search      ✓ BM25 + Dense + RRF  │
│  ├─ M3: Reranking          ✓ Cross-encoder + latency
│  └─ M4: Evaluation         ✓ RAGAS + failure analysis
│                                                       │
│  PHẦN B: NHÓM (40 điểm • 30min)                    │
│  ├─ Build pipeline         (Module 1+2+3+4+5)       │
│  ├─ Run RAGAS              (20 test questions)      │
│  ├─ Compare scores         (Naive vs Production)    │
│  ├─ Failure analysis       (Bottom-5 → diagnosis)  │
│  └─ Presentation           (5 phút/nhóm)           │
│                                                       │
│  BONUS                                               │
│  ├─ M5: Enrichment         (+3 điểm)               │
│  ├─ Latency breakdown      (+2 điểm)               │
│  └─ Faithfulness ≥ 0.85    (+5 điểm)               │
└──────────────────────────────────────────────────────┘
```

---

## 🎯 Phần A: Cá nhân — Chọn 1 Module

### **Lựa chọn theo năng lực**

| Module | Độ khó | Dung lượng | Người phù hợp |
|--------|--------|-----------|-------|
| **M1: Chunking** | ⭐⭐ | ~200 dòng | Người mới, focus vào text processing |
| **M2: Hybrid Search** | ⭐⭐⭐ | ~250 dòng | Người thích Vietnamese NLP + vector DB |
| **M3: Reranking** | ⭐⭐ | ~150 dòng | Người muốn học LLM models |
| **M4: Evaluation** | ⭐ | ~100 dòng | Người mới, nhiều tools sẵn có |

### **M1: Advanced Chunking (200 dòng)**

**Mục tiêu**: Cắt document thành chunks **hợp lý** để embedding chính xác

**Các strategy cần implement:**

1. **Semantic Chunking** (70 dòng)
   - Split câu theo cosine similarity (threshold = 0.85)
   - Nếu similarity < 0.85 → tách chunk mới
   - Dùng `SentenceTransformer("all-MiniLM-L6-v2")` để encode
   - Input: "Nhân viên được 12 ngày nghỉ. Cha được 5 ngày."
   - Output: 2 chunks (vì 2 ý khác nhau)

2. **Hierarchical Chunking** (60 dòng) ⭐ **QUAN TRỌNG cho production**
   - Parent: 2048 chars (đủ context cho LLM)
   - Child: 256 chars (nhỏ, embedding chính xác)
   - Flow: Index children → Search → Return parent để prompt
   - Example:
     ```
     Parent: "Chính sách Nghỉ phép ... [2000 chars] ... và công ty."
     Child 1: "Nhân viên được 12 ngày nghỉ phép hàng năm"
     Child 2: "Điều kiện: hoàn thành thử việc 3 tháng"
     ```

3. **Structure-Aware Chunking** (50 dòng)
   - Parse markdown headers (`# `, `## `, `###`)
   - Chunk theo section logic (header + content)
   - Giữ table, code block nguyên vẹn
   - Example:
     ```markdown
     # Phần 1: Nghỉ phép
     ## 1.1 Quyền nghỉ phép
     [content] → 1 chunk
     ## 1.2 Điều kiện
     [content] → 1 chunk
     ```

4. **Compare Strategies** (20 dòng)
   - Chạy cả 4 chunking trên cùng 1 document
   - In bảng so sánh: strategy, #chunks, avg chunk size, latency

**Tests cần pass:**
```python
✓ chunk_semantic() → list[Chunk], groups by topic
✓ chunk_hierarchical() → (parents, children), parent_id valid
✓ chunk_structure_aware() → keeps headers, metadata["section"]
✓ compare_strategies() → dict với stats
```

---

### **M2: Hybrid Search (250 dòng)**

**Mục tiêu**: Tìm documents **chính xác** từ 2 angle khác nhau

**Khái niệm chính:**

1. **BM25 Search** (80 dòng)
   - **Keyword-based**: Tìm words chính xác trong document
   - **Vietnamese-specific**: Phải tokenize tiếng Việt
   - Flow:
     ```
     Query: "Nhân viên nghỉ phép bao nhiêu ngày?"
     → Tokenize: ["nhân viên", "nghỉ phép", "bao nhiêu", "ngày"]
     → BM25 score trên corpus
     → Top 20 documents
     ```
   - Why: "nghỉ phép" = 1 word (not 2), BM25 hiểu context tốt hơn
   - Library: `underthesea` + `rank_bm25.BM25Okapi`

2. **Dense Search** (100 dòng)
   - **Semantic**: Encode query + docs thành vectors (1024 dims)
   - **Model**: BGE-M3 (Multilingual, tiếng Việt)
   - Flow:
     ```
     Query: "Bao nhiêu ngày phép?"
     → Encode: [0.23, -0.45, 0.67, ...] (1024 dims)
     → Cosine similarity vs all document vectors
     → Top 20 by similarity
     ```
   - Why: Hiểu **ý** query, không chỉ keywords
   - Library: `sentence-transformers` + `qdrant-client`

3. **Reciprocal Rank Fusion (RRF)** (30 dòng)
   - **Merge** BM25 + Dense scores
   - Formula: `score(doc) = Σ 1/(k + rank_i(doc))`
   - Example:
     ```
     BM25 rank: [doc1(rank=1), doc2(rank=2), doc3(rank=5)]
     Dense rank: [doc2(rank=1), doc1(rank=3), doc5(rank=2)]
     
     RRF:
     doc1: 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
     doc2: 1/(60+2) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323
     doc5: 1/(60+2) = 0.0159
     
     Top-3: doc1, doc2, doc5
     ```
   - Why: Kết hợp 2 method → tốt hơn 1 method

4. **Segment Vietnamese** (40 dòng)
   - Tokenize tiếng Việt: "Tôi đi học" → ["Tôi", "đi", "học"]
   - Library: `underthesea.word_tokenize(text, format="text")`
   - Important: Phải làm trước indexing BM25

**Tests cần pass:**
```python
✓ segment_vietnamese() → "nhân viên được nghỉ" = "nhân_viên được nghỉ"
✓ BM25.search("nghỉ phép") → top result chứa "nghỉ phép"
✓ DenseSearch.search() → load BGE-M3, create collection
✓ RRF merge 2 lists → sorted by score
```

---

### **M3: Reranking (150 dòng)**

**Mục tiêu**: Lọc top 20 → top 3 **documents most relevant**

**Khái niệm:**

1. **Cross-Encoder Reranker** (80 dòng)
   - Input: Query + 20 documents
   - Model: BGE-Reranker-v2-m3 (Tiếng Việt)
   - Output: Rerank score (0-1)
   - Flow:
     ```
     Query: "Được nghỉ phép bao lâu"
     Docs: [
       "Nhân viên được 12 ngày/năm" (search_score=0.8),
       "Mật khẩu thay đổi 90 ngày" (search_score=0.75),
       "Công tác > 7 ngày được nghỉ" (search_score=0.70),
     ]
     
     Reranker:
     doc1: rerank_score = 0.95 ✓ (match perfectly)
     doc2: rerank_score = 0.15 ✗ (about password, not leave)
     doc3: rerank_score = 0.82 ✓ (about leave, less specific)
     
     Top-3: [doc1, doc3, doc2]  (re-ranked by cross-encoder)
     ```
   - Library: `FlagEmbedding.FlagReranker` hoặc `sentence-transformers.CrossEncoder`

2. **Benchmark Latency** (70 dòng)
   - Đo thời gian reranking 5 documents
   - Chạy n_runs=5 lần, tính avg/min/max
   - Target: < 5 giây (first load likely 2-3s)
   - Code:
     ```python
     times = []
     for _ in range(5):
       start = time.perf_counter()
       reranker.rerank(query, docs, top_k=3)
       times.append((time.perf_counter() - start) * 1000)  # ms
     
     print(f"Latency: {np.mean(times):.2f}ms (avg)")
     ```

**Tests cần pass:**
```python
✓ rerank() trả về ≤ 3 RerankResult
✓ Sorted by rerank_score descending
✓ Doc về "nghỉ phép" ranked cao hơn "VPN"
✓ Latency < 5s
```

---

### **M4: Evaluation (100 dòng)**

**Mục tiêu**: Đo **chất lượng** RAG pipeline bằng 4 metrics

**4 RAGAS Metrics:**

1. **Faithfulness** (0-1)
   - Câu hỏi: Answer có nhớng từ từ context không? (không hallucinate)
   - Score cao = LLM trả lời dựa vào context, không tự tạo
   - Target: ≥ 0.85 (Bonus +5)
   - Example:
     ```
     Context: "Nhân viên được 12 ngày nghỉ"
     Question: "Được bao nhiêu ngày?"
     Good answer: "12 ngày" → Faithfulness ≈ 0.95
     Bad answer: "15 ngày" → Faithfulness ≈ 0.1 (hallucinate)
     ```

2. **Answer Relevancy** (0-1)
   - Câu hỏi: Answer có trả lời question không?
   - Score cao = Answer trực tiếp, không lạc đề
   - Example:
     ```
     Question: "Được bao nhiêu ngày?"
     Good answer: "12 ngày" → Relevancy ≈ 0.95
     Bad answer: "Công ty có tổng 200 NV" → Relevancy ≈ 0.2
     ```

3. **Context Precision** (0-1)
   - Câu hỏi: Contexts có nhiều irrelevant chunks không?
   - Score cao = Khi search, lọc được mỗi chunk relevant
   - ← Impact của **Reranking (M3)**
   - Example:
     ```
     Question: "Bao nhiêu ngày nghỉ?"
     
     With Reranking OFF:
     Contexts: [
       "12 ngày nghỉ phép" ✓,
       "VPN certificate mỗi 6 tháng", ✗
       "Mật khẩu 8 ký tự", ✗
     ]
     Precision ≈ 0.33 (1/3 relevant)
     
     With Reranking ON:
     Contexts: [
       "12 ngày nghỉ phép" ✓,
       "Điều kiện: 3 tháng thử việc" ✓,
       "Thưởng: 1 tháng" ✓,
     ]
     Precision ≈ 1.0 (3/3 relevant)
     ```

4. **Context Recall** (0-1)
   - Câu hỏi: Contexts có bao gồm đủ info để trả lời không?
   - Score cao = Search tìm được ALL relevant chunks
   - ← Impact của **Chunking (M1) + Search (M2)**
   - Example:
     ```
     Question: "Bao nhiêu ngày nghỉ? Điều kiện gì?"
     
     Good search result:
     Contexts: [
       "12 ngày nghỉ",
       "Điều kiện: hoàn thành 3 tháng thử việc"
     ]
     Recall ≈ 1.0 (đủ info để trả lời)
     
     Bad search result:
     Contexts: [
       "12 ngày nghỉ"
     ]
     Recall ≈ 0.5 (thiếu info về điều kiện)
     ```

**Implementation:**

```python
def evaluate_ragas(questions, answers, contexts, ground_truths):
    # 1. Chuẩn bị dataset
    from datasets import Dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,  # list[list[str]]
        "ground_truth": ground_truths,
    })
    
    # 2. Chạy RAGAS
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    
    result = evaluate(dataset, metrics=[
        faithfulness, answer_relevancy, context_precision, context_recall
    ])
    
    # 3. Extract scores
    df = result.to_pandas()
    f1 = df['faithfulness'].mean()
    ar = df['answer_relevancy'].mean()
    cp = df['context_precision'].mean()
    cr = df['context_recall'].mean()
    
    return {
        'faithfulness': f1,
        'answer_relevancy': ar,
        'context_precision': cp,
        'context_recall': cr,
        'per_question': [EvalResult(...) for row in df.iterrows()]
    }
```

**Failure Analysis:**

```python
def failure_analysis(eval_results, bottom_n=10):
    # 1. Tính avg score cho mỗi question
    scores = [
        (r.question, np.mean([r.faithfulness, r.answer_relevancy, 
                              r.context_precision, r.context_recall]))
        for r in eval_results
    ]
    
    # 2. Sort ascending, lấy bottom-10
    failures = sorted(scores, key=lambda x: x[1])[:bottom_n]
    
    # 3. Diagnose từng failure
    diagnostics = []
    for question, avg_score in failures:
        r = next(r for r in eval_results if r.question == question)
        worst_metric = min([
            ('faithfulness', r.faithfulness),
            ('answer_relevancy', r.answer_relevancy),
            ('context_precision', r.context_precision),
            ('context_recall', r.context_recall),
        ], key=lambda x: x[1])
        
        diagnosis_map = {
            'faithfulness': ('LLM hallucinating', 'Tighten prompt, lower temperature'),
            'answer_relevancy': ('Answer mismatch', 'Improve prompt template'),
            'context_precision': ('Too many irrelevant chunks', 'Add reranking'),
            'context_recall': ('Missing chunks', 'Improve chunking + search'),
        }
        
        diagnosis, fix = diagnosis_map[worst_metric[0]]
        
        diagnostics.append({
            'question': question,
            'worst_metric': worst_metric[0],
            'score': worst_metric[1],
            'diagnosis': diagnosis,
            'suggested_fix': fix,
        })
    
    return diagnostics
```

**Tests cần pass:**
```python
✓ evaluate_ragas() trả về dict với 4 metrics
✓ Tất cả scores là 0-1 float
✓ failure_analysis() trả về list với diagnosis + suggested_fix
✓ Mỗi diagnosis có suggested_fix hợp lý
```

---

## 🔗 Phần B: Nhóm — Ghép thành Pipeline

**Nếu nhóm chỉ có 3 người:**
- M1 + M4 cho 1 người (2 module)
- M2, M3 cho 2 người khác

**Flow pipeline:**

```
1. Load Documents
   └─> M1: Chunk (hierarchical recommended)

2. Preparation
   └─> M5: Enrichment (optional, +3 bonus)

3. Indexing
   └─> M2: HybridSearch (BM25 + Dense)

4. Per Query:
   Query → M2: Search (top 20) → M3: Rerank (top 3) → LLM Gen → Answer

5. Evaluation
   └─> M4: RAGAS (4 metrics) + Failure Analysis

6. Report
   ├─> Compare Naive vs Production
   ├─> Analyze failures
   └─> Present 5 min
```

**Điểm nhóm breakdown:**

| Bước | Điểm | Tiêu chí |
|------|------|---------|
| Pipeline end-to-end | 10 | Chạy được, không error |
| RAGAS ≥ 0.75 | 10 | Ít nhất 1 metric ≥ 0.75 |
| Failure analysis | 10 | Bottom-5 có diagnosis hợp lý |
| Presentation | 10 | 4 điểm + có data |

**Bonus:**
- Faithfulness ≥ 0.85: +5
- M5 Enrichment: +3
- Latency breakdown: +2

---

## 📊 So sánh Naive vs Production

### What changes from naive?

| Bước | Naive | Production |
|------|-------|------------|
| **Chunking** | paragraph split (500 chars) | hierarchical (2048 parent / 256 child) + semantic |
| **Search** | Dense only (BGE-M3) | Hybrid (BM25 + Dense + RRF) |
| **Reranking** | None | Cross-encoder (bge-reranker-v2-m3) |
| **Enrichment** | Raw text | Contextual prepend (optional) |
| **LLM Gen** | Context[0] as answer | Full LLM generation (optional) |

### Expected improvement:

```
Metric                  Naive       Production      Δ          Target
─────────────────────────────────────────────────────────────────────
Faithfulness            0.50        0.75+           +0.25      ✓ +5 bonus if ≥0.85
Answer Relevancy        0.60        0.80+           +0.20      ✓
Context Precision       0.55        0.75+           +0.20      ← From Reranking
Context Recall          0.70        0.85+           +0.15      ← From Better Search
```

---

## 📋 Checklist Trước khi Nộp

**Cả nhóm:**

- [ ] Tất cả 4 modules implement xong (hoặc có fallback)
- [ ] `pytest tests/ -v` pass (hoặc mostly pass)
- [ ] `python naive_baseline.py` chạy thành công
- [ ] `python main.py` chạy end-to-end
- [ ] `reports/naive_baseline_report.json` chứa scores
- [ ] `reports/ragas_report.json` chứa production scores
- [ ] `analysis/failure_analysis.md` nội dung hợp lý
- [ ] `analysis/group_report.md` có 4 phần nhập
- [ ] `analysis/reflections/reflection_*.md` mỗi thành viên

**Cá nhân:**

- [ ] Module được implement trọn vẹn (tất cả TODO xong)
- [ ] Tests pass cho module đó
- [ ] Code có comments + type hints
- [ ] Reflection viết xong

**Trước nộp:**

```bash
python check_lab.py   # Kiểm tra định dạng
```

---

## 💡 Tips & Tricks

### Performance Optimization:

1. **Embedding cache**: Encode chunks 1 lần, lưu/reload
   ```python
   import pickle
   embeddings = encoder.encode(texts, show_progress_bar=False)
   with open("cached_embeddings.pkl", "wb") as f:
       pickle.dump(embeddings, f)
   ```

2. **Batch processing**: Chunk nhỏ → encode theo batch
   ```python
   embeddings = encoder.encode(texts, batch_size=64)
   ```

3. **Model optimization**: Dùng quantized version
   ```python
   from sentence-transformers import SentenceTransformer
   model = SentenceTransformer("BAAI/bge-m3")
   model.eval()  # faster
   ```

### Debugging:

1. **Print đúng position**: vào pipeline nào, step nào
   ```python
   print(f"[M2] Query: {query}, Top-k results: {len(results)}")
   ```

2. **Check intermediate outputs:**
   ```python
   if len(chunks) == 0:
       print("⚠️  No chunks indexed! Check M1 output.")
   ```

3. **RAGAS mock test**: Nếu API slow
   ```python
   mock_results = {
       'faithfulness': 0.8,
       'answer_relevancy': 0.75,
       'context_precision': 0.70,
       'context_recall': 0.80,
   }
   ```

---

## 🎓 Learning Goals

Sau lab này, bạn sẽ hiểu:

- ✓ **RAG pipeline**: Từ document → embedding → search → rank → generate
- ✓ **Chunking strategies**: Semantic grouping, hierarchical retrieval
- ✓ **Hybrid search**: Kết hợp keyword + semantic
- ✓ **Reranking**: Lọc top-K kết quả
- ✓ **RAGAS evaluation**: Đo chất lượng RAG, diagnose failures
- ✓ **Vietnamese NLP**: Word tokenization, multilingual models
- ✓ **Production readiness**: Caching, benchmarking, error handling

---

## 🆘 Common Issues & Solutions

| Lỗi | Nguyên nhân | Fix |
|-----|-----------|-----|
| `ModuleNotFoundError: underthesea` | Chưa cài | `pip install underthesea` |
| `ConnectionError: Qdrant` | Qdrant không chạy | `docker-compose up -d` |
| `CUDA OOM` | GPU memory đầy | Giảm batch_size |
| `RRF scores sum to 0` | Nhầm formula | Check k parameter (thường k=60) |
| `Faithfulness = 0` | LLM always hallucinate | Tighten system prompt |

---

## 📚 Modern RAG Best Practices

1. **Always hierarchical chunk** (not fixed size)
2. **Hybrid search** (keyword + semantic) > single method
3. **Rerank trước LLM** (filter irrelevant)
4. **Evaluate rigorously** (RAGAS on validation set)
5. **Cache embeddings** (expensive to recompute)
6. **Monitor latency** (end-to-end < 1s target)
7. **Handle edge cases** (empty query, no results, etc.)

---

**Good luck với lab! 🚀**
