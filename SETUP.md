# Lab 18: Production RAG Pipeline — Hướng dẫn Setup

## 📦 Quick Start (5 phút)

### 1. Clone và chuẩn bị environment

```bash
# Tạo virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Cấu hình API Keys

```bash
# Copy template
cp .env.example .env

# Điền vào .env:
# OPENAI_API_KEY=sk-...   (từ https://platform.openai.com/api-keys)
# COHERE_API_KEY=...      (từ https://dashboard.cohere.com/)
```

### 3. Khởi động Qdrant (Vector Database)

**Cách 1: Dùng Docker Compose (Recommended)**

```bash
docker-compose up -d

# Check status
curl http://localhost:6333/health
# Expected: {"status":"ok"}
```

**Cách 2: Cài Qdrant locally (without Docker)**

```bash
# macOS
brew install qdrant

# Linux
wget https://github.com/qdrant/qdrant/releases/download/v2.7.0/qdrant-x86_64-unknown-linux-musl.tar.gz
tar xz && ./qdrant

# Windows: Download từ https://github.com/qdrant/qdrant/releases
```

### 4. Chạy Baseline trước

```bash
# QUAN TRỌNG: Chạy naive baseline để có điểm so sánh
python naive_baseline.py

# Output: naive_baseline_report.json (scores đầu tiên)
# Giữ lại để so sánh với production pipeline sau
```

---

## 🎯 Phần A: Implement Cá nhân (60 điểm)

Mỗi người chọn **1 module** từ 4 module dưới đây:

### **Module 1: Chunking (src/m1_chunking.py)**

**Mục tiêu**: Implement 3 advanced chunking strategies

```python
# Các TODO cần làm:
# 1. chunk_semantic()         - Nhóm câu theo cosine similarity
# 2. chunk_hierarchical()     - Parent-child hierarchy (recommended)
# 3. chunk_structure_aware()  - Parse markdown headers
# 4. compare_strategies()     - Compare all 4 strategies
```

**Test locally:**

```bash
pytest tests/test_m1.py -v

# Expected output:
# test_chunk_semantic PASSED
# test_chunk_hierarchical PASSED
# test_chunk_structure_aware PASSED
```

**Dependencies chính:**
- `sentence-transformers` — SentenceTransformer("all-MiniLM-L6-v2")
- `numpy` — cosine similarity

---

### **Module 2: Hybrid Search (src/m2_search.py)**

**Mục tiêu**: BM25 (Vietnamese) + Dense + RRF fusion

```python
# Các TODO cần làm:
# 1. segment_vietnamese()    - underthesea word tokenize
# 2. BM25Search.index()      - rank_bm25.BM25Okapi
# 3. BM25Search.search()     - BM25 scoring
# 4. DenseSearch.index()     - Embed + Qdrant upload
# 5. DenseSearch.search()    - Vector similarity search
# 6. reciprocal_rank_fusion() - Merge BM25 + Dense scores
```

**Test locally:**

```bash
pytest tests/test_m2.py -v

# Expected:
# test_vietnamese_segmentation PASSED
# test_bm25_search PASSED
# test_dense_search PASSED
# test_hybrid_search PASSED
```

**Dependencies chính:**
- `underthesea` — Vietnamese word tokenization
- `rank_bm25` — BM25 indexing
- `sentence-transformers` — "BAAI/bge-m3"
- `qdrant-client` — Vector DB

---

### **Module 3: Reranking (src/m3_rerank.py)**

**Mục tiêu**: Cross-encoder reranker + latency benchmark

```python
# Các TODO cần làm:
# 1. CrossEncoderReranker._load_model()  - Load bge-reranker-v2-m3
# 2. CrossEncoderReranker.rerank()       - Score + sort
# 3. FlashrankReranker (optional)        - Lightweight alternative
# 4. benchmark_reranker()                - Measure latency
```

**Test locally:**

```bash
pytest tests/test_m3.py -v

# Expected:
# test_cross_encoder_rerank PASSED
# test_latency_benchmark PASSED
```

**Dependencies chính:**
- `FlagEmbedding` — FlagReranker("BAAI/bge-reranker-v2-m3")
- `sentence-transformers` — CrossEncoder alternative
- `time` — perf_counter()

---

### **Module 4: Evaluation (src/m4_eval.py)**

**Mục tiêu**: RAGAS evaluation + failure analysis

```python
# Các TODO cần làm:
# 1. evaluate_ragas()    - Run 4 RAGAS metrics
# 2. failure_analysis()  - Bottom-10 questions + diagnosis
# 3. save_report()       - JSON output (đã sẵn)
```

**Test locally:**

```bash
pytest tests/test_m4.py -v

# Expected:
# test_evaluate_ragas PASSED
# test_failure_analysis PASSED
```

**Dependencies chính:**
- `ragas` — RAGAS evaluation framework
- `datasets` — Dataset.from_dict()

---

## 🔗 Phần B: Ghép nhóm (40 điểm)

Sau khi cá nhân implement xong, **nhóm ghép 4 modules thành pipeline hoàn chỉnh**.

### Bước 1: Integrate các modules vào pipeline

File: `src/pipeline.py`

```python
def build_pipeline():
    # M1: Load & Chunk
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"])
        all_chunks.extend(children)
    
    # M5: Enrichment (optional, +3 bonus)
    enriched = enrich_chunks(all_chunks, methods=["contextual"])
    
    # M2: Index
    search = HybridSearch()
    search.index(all_chunks)
    
    # M3: Reranker load
    reranker = CrossEncoderReranker()
    
    return search, reranker

def evaluate_pipeline(search, reranker):
    # M4: Run RAGAS evaluation
    # TODO: Replace answer generation with LLM (see pipeline.py comments)
```

### Bước 2: Chạy full pipeline

```bash
python src/pipeline.py

# Output:
# [1/3] Chunking documents...
# [2/4] Enriching chunks...
# [3/4] Indexing...
# [4/4] Loading reranker...
# [Eval] Running RAGAS...
# Production RAG Scores:
#   ✓ faithfulness: 0.8234
#   ... (other metrics)
```

### Bước 3: So sánh với naive baseline

```bash
python main.py

# Output:
# Metric                    Basic        Production        Δ
# ---------------------------------------------------------
# ✓ faithfulness         0.5234        0.8234        +0.3000
# ✓ answer_relevancy     0.4892        0.7891        +0.3000
# ...
```

### Bước 4: Failure analysis

1. Mở `reports/ragas_report.json`
2. Tìm bottom-5 questions (lowest scores)
3. Điền vào `analysis/failure_analysis.md` với:
   - Question + scores
   - Root cause (từ Diagnostic Tree)
   - Suggested fix

Ví dụ template:

```markdown
## Q: "Nhân viên được nghỉ phép bao nhiêu ngày?"
- Faithfulness: 0.45 → LLM hallucinating → Tighten prompt
- Context Recall: 0.60 → Missing relevant chunks → Improve search
- Root cause: Hybrid search không tìm thấy document chính xác
- Fix: Thêm BM25 hoặc semantic chunking
```

### Bước 5: Group report

Điền vào `analysis/group_report.md`:

```markdown
## 1. RAGAS Scores
[So sánh naive vs production]

## 2. Biggest Win
[Module nào cải thiện nhiều nhất? Tại sao?]

## 3. Case Study
[1 question cụ thể check qua Error Tree]

## 4. Next Steps
[Nếu có +1 giờ, tối ưu gì tiếp theo?]
```

### Bước 6: Individual Reflections

Mỗi người viết 1-2 trang:

- Điều khó nhất khi implement module?
- Metric nào quan trọng nhất?
- Nếu có cơ hội, sẽ improve gì?

Lưu vào `analysis/reflections/reflection_[TênNhân].md`

---

## ✅ Kiểm tra trước khi nộp

```bash
# 1. Chạy tất cả tests
pytest tests/ -v

# 2. Kiểm tra định dạng
python check_lab.py

# 3. Generate reports
python main.py

# 4. Verify submission structure
ls -R analysis/ reports/
```

---

## 🐳 Docker Setup (Optional)

Để chạy trong container:

```bash
# Build image
docker build -t rag-lab18 .

# Run container
docker run -v $(pwd):/app rag-lab18 python main.py

# Hoặc dùng docker-compose
docker-compose run --rm rag python main.py
```

---

## 📊 Performance Targets (Bonus)

| Metric | Target | Bonus |
|--------|--------|-------|
| Faithfulness ≥ 0.85 | +5 |
| Enrichment pipeline | +3 |
| Latency breakdown | +2 |

---

## 🆘 Troubleshooting

### "ModuleNotFoundError: No module named 'underthesea'"

```bash
pip install underthesea
# Nếu lỗi: pip install underthesea --upgrade
```

### "ConnectionError: Failed to connect to Qdrant"

```bash
# Kiểm tra Qdrant running
docker ps | grep qdrant

# Nếu không chạy
docker-compose up -d

# Check health
curl http://localhost:6333/health
```

### "CUDA out of memory" (khi embed chunks)

```python
# Trong m2_search.py, thêm:
encoder = SentenceTransformer("BAAI/bge-m3")
embeddings = encoder.encode(texts, batch_size=32)  # Giảm batch size
```

### "OPENAI_API_KEY not found"

- [x] Copy `.env.example` → `.env`
- [x] Điền API key từ https://platform.openai.com/api-keys
- [x] Chuẩn bị ít nhất $5 credit

---

## 📚 References

- **RAGAS**: https://docs.ragas.io/
- **BGE-M3**: https://huggingface.co/BAAI/bge-m3
- **Qdrant**: https://qdrant.tech/documentation/
- **Underthesea**: https://underthesea.readthedocs.io/

---

## 💬 Contact

- Giảng viên: M.Sc Trần Minh Tú
- Discord: #lab18-support
- Office hours: Thứ 5, 15:00-16:00
