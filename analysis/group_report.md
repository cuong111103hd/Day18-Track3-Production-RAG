# Group Report — Lab 18: Production RAG

**Nhóm:** Lab 18 - Production RAG - Nhóm 2 thành viên
**Ngày:** 04/05/2026

## Thành viên & Phân công
1. Thành viên 1: Đỗ Hải Nam - 2A202600038
2. Thành viên 2: Nguyễn Đức Cường - 2A202600147

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------:|
| Nguyễn Đức Cường | M1: Chunking | ✅ | 13/13 |
| Nguyễn Đức Cường | M2: Hybrid Search | ✅ | 5/5 |
| Nguyễn Đức Cường | M3: Reranking | ✅ | 5/5 |
| Đỗ Hải Nam | M4: Evaluation | ✅ | 4/4 |
| Đỗ Hải Nam | M5: Enrichment | ✅ | 10/10 |

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|------:|-----------:|---:|
| Faithfulness | 1.0000 | 1.0000 | +0.0000 |
| Answer Relevancy | 0.2853 | 0.3273 | +0.042 |
| Context Precision | 0.5117 | 0.5416 | +0.0299 |
| Context Recall | 1.0000 | 1.0000 | +0.0000 |

## Key Findings

1. **Biggest improvement:** `Answer Relevancy` tăng nhờ pipeline mới chọn context sạch hơn và sinh answer theo sentence khớp query thay vì trả nguyên context đầu tiên.
2. **Biggest challenge:** `Context Precision` vẫn thấp vì retrieval còn trả về các chunk gần chủ đề nhưng không đúng trọng tâm, nhất là khi Cohere bị rate limit và phải fallback.
3. **Surprise finding:** `Faithfulness` và `Context Recall` đều giữ mức 1.0, nhưng điều đó không đồng nghĩa output tốt; chỉ cần câu trả lời bám đúng đoạn văn là hai metric này đã rất dễ cao.

## Presentation Notes (5 phút)

1. RAGAS scores (naive vs production):
   - Naive: `1.0000 / 0.2853 / 0.5117 / 1.0000`
   - Production: `1.0000 / 0.3273 / 0.5416 / 1.0000`
2. Biggest win — module nào, tại sao:
   - M4 + pipeline answer synthesis giúp `Answer Relevancy` tăng.
   - M5 enrichment và Jina embeddings giúp truy hồi đủ ngữ cảnh cho câu hỏi fact-based.
3. Case study — 1 failure, Error Tree walkthrough:
   - Chọn câu `Bảo hành laptop công ty là khoảng thời gian nào?`
   - Output bị trả về heading, không phải fact.
4. Next optimization nếu có thêm 1 giờ:
   - Bỏ heading khỏi candidate answer.
   - Rank câu trả lời theo số liệu / đơn vị.
   - Giảm top-k retrieval, hoặc thêm metadata filter theo category.
