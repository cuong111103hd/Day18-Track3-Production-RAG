# Individual Reflection — Lab 18

**Tên:** Nguyễn Đức Cường  
**Module phụ trách:** M1/Chunking + M2/Search + M3/Reranking

---

## 1. Đóng góp kỹ thuật

- Module đã implement:
  - M1: Advanced Chunking (Semantic, Hierarchical, Structure-Aware)
  - M2: Hybrid Search (BM25 + Qdrant Dense Vector)
  - M3: Reranking (Cohere API)
- Các hàm/class chính đã viết:
  - `chunk_hierarchical`, `chunk_semantic`, `chunk_structure_aware` (Xử lý đặc thù cho bảng biểu và tiêu đề).
  - `HybridSearch.search`: Kết hợp điểm số RRF (Reciprocal Rank Fusion) giữa BM25 và Dense Search.
  - `CohereReranker.rerank`: Tích hợp Cohere API `rerank-multilingual-v3.0` để tối ưu hóa kết quả cuối cùng.
- Số tests pass:
  - `23/23` khi chạy `tests/test_m1.py`, `tests/test_m2.py`, và `tests/test_m3.py`.

## 2. Kiến thức học được

- Khái niệm mới:
  - Hiểu sâu về sự kết hợp giữa **Lexical Search (BM25)** và **Semantic Search (Dense)** để giải quyết bài toán tìm kiếm từ khóa tiếng Việt (tên riêng, mã số thuế, số chỉ tiêu).
  - Kỹ thuật **Structure-Aware Chunking** để bảo toàn ngữ cảnh của các bảng biểu phức tạp trong báo cáo tài chính.
- Điều bất ngờ nhất:
  - Việc chuyển từ search thông thường sang dùng **Reranker** giúp tăng Context Precision rõ rệt ngay cả khi tập dữ liệu bị nhiễu.
- Kết nối với bài giảng:
  - Áp dụng kiến thức về Retrieval Strategy và Hybrid Search từ bài giảng Production RAG vào thực tế xử lý văn bản quy phạm pháp luật Việt Nam.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất:
  - Lỗi kết nối và deprecation của Qdrant Client (chuyển đổi từ `.search()` sang `.query_points()`).
  - Lỗi logic mapping `parent_id` trong Hierarchical Chunking khiến con bị lạc mất cha.
- Cách giải quyết:
  - Đọc kỹ tài liệu API mới nhất của Qdrant v1.17+ để cập nhật code.
  - Thực hiện debug từng bước (step-by-step) trên bộ test M1 để fix lỗi metadata key.
  - Implement cơ chế **Rate-limit Retry** cho Cohere API khi sử dụng bản Trial (10 requests/minute).

## 4. Nếu làm lại

- Sẽ làm khác điều gì:
  - Sẽ tối ưu phần tiền xử lý (Preprocessing) cho tiếng Việt tốt hơn nữa (tách từ, loại bỏ stopword chuyên sâu) trước khi đưa vào BM25.
- Module nào muốn thử tiếp:
  - Muốn thử nghiệm GraphRAG hoặc tích hợp Knowledge Graph vào M2 để xử lý các truy vấn bắc cầu phức tạp.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 5 |
| Teamwork | 5 |
| Problem solving | 5 |
