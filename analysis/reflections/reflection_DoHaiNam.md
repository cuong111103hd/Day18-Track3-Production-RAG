# Individual Reflection — Lab 18

**Tên:** Đỗ Hải Nam  
**Module phụ trách:** M4/Evaluation + M5/Enrichment

---

## 1. Đóng góp kỹ thuật

- Module đã implement:
  - M4: Evaluation
  - M5: Enrichment
- Các hàm/class chính đã viết:
  - `load_test_set`, `evaluate_ragas`, `failure_analysis`, `save_report`
  - `summarize_chunk`, `generate_hypothesis_questions`, `contextual_prepend`, `extract_metadata`, `enrich_chunks`
- Số tests pass:
  - `14/14` khi chạy `tests/test_m4.py` và `tests/test_m5.py`

## 2. Kiến thức học được

- Khái niệm mới nhất:
  - RAGAS evaluation, đặc biệt là `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`
- Điều bất ngờ nhất:
  - `faithfulness` và `context_recall` có thể rất cao dù answer vẫn chưa thật sự tốt, nên cần nhìn tổng thể nhiều metric
- Kết nối với bài giảng (slide nào):
  - Slide về retrieval quality, reranking, answer generation, và evaluation loop trong RAG pipeline
- Ngoài các kiến thức học được từ việc triển khai m4, m5, mình còn được tiếp thu thêm kiến thức từ m1, m2, m3 của thành viên trong nhóm.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất:
  - Rate limit / lỗi API từ provider ngoài làm pipeline chậm hoặc fallback nhiều lần
- Cách giải quyết:
  - Tách config ra `.env`/`config.py`
  - Thêm retry/backoff cho Cohere
  - Tạo fallback heuristic để pipeline vẫn chạy được khi API lỗi
- Thời gian debug:
  - Chủ yếu ở M4/M5 và phần integrate pipeline end-to-end

## 4. Nếu làm lại

- Sẽ làm khác điều gì:
  - Thiết kế answer generation sớm hơn và kiểm soát output chặt hơn thay vì để fallback heuristic quá lâu
- Module nào muốn thử tiếp:
  - M3 reranking với provider ổn định hơn hoặc local reranker mạnh hơn
- Tối ưu RAG accuracy để các metric ổn định hơn và tốt hơn

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 5 |
| Teamwork | 5 |
| Problem solving | 5 |
