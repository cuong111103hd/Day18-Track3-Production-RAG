# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Lab 18 - Production RAG  
**Thành viên:** Nguyễn Đức Cường → M1/M2/M3 · Đỗ Hải Nam → M4/M5

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------:|-----------:|---:|
| Faithfulness | 1.0000 | 1.0000 | +0.0000 |
| Answer Relevancy | 0.2853 | 0.3273 | +0.042 |
| Context Precision | 0.5117 | 0.5416 | +0.0299 |
| Context Recall | 1.0000 | 1.0000 | +0.0000 |

## Bottom-5 Failures

### #1
- **Question:** Bảo hành laptop công ty là khoảng thời gian nào?
- **Expected:** Bảo hành laptop là 3 năm từ ngày nhân viên nhận.
- **Got:** `Thiết bị Công ty`
- **Worst metric:** Context Precision
- **Error Tree:** Output sai → Context đúng? `khá đúng` → Query OK? `có` → Root cause: answer synthesis lấy nhầm heading thay vì câu chứa số liệu.
- **Root cause:** Context truy hồi đúng section, nhưng bước sinh answer ưu tiên title/heading ngắn, nên không trích ra câu chứa thông tin thực.
- **Suggested fix:** Loại bỏ heading khi chọn answer, ưu tiên sentence có số liệu / đơn vị, và dùng prompt generation thật thay heuristic khi có thể.

### #2
- **Question:** Thời gian thử việc cần bao lâu để hưởng nghỉ phép?
- **Expected:** Đã hoàn thành thời gian thử việc 3 tháng là một trong các điều kiện hưởng nghỉ phép.
- **Got:** `Quyền Nghỉ phép Hàng năm`
- **Worst metric:** Answer Relevancy
- **Error Tree:** Output sai → Context đúng? `một phần` → Query OK? `có` → Root cause: câu trả lời bị trượt sang title của section gần nhất.
- **Root cause:** Query tokens overlap với section lớn về nghỉ phép, nhưng answer generation không khóa vào câu chứa "3 tháng".
- **Suggested fix:** Thêm sentence ranking theo số/mốc thời gian, giảm trọng số heading, và tăng ràng buộc sinh câu trả lời theo câu hỏi.

### #3
- **Question:** Lương trong 30 ngày đầu khi ốm đau là bao nhiêu?
- **Expected:** Hưởng 100% lương cơ bản trong 30 ngày đầu khi nhân viên bị bệnh.
- **Got:** `- **Trong 30 ngày đầu**: Hưởng 10`
- **Worst metric:** Context Precision
- **Error Tree:** Output sai → Context đúng? `đúng một phần` → Query OK? `có` → Root cause: context có thông tin đúng nhưng answer bị cắt cụt / thiếu hoàn chỉnh.
- **Root cause:** Trích xuất câu từ context vẫn còn phụ thuộc định dạng markdown bullet, nên câu trả lời có thể bị truncate ở giữa số liệu.
- **Suggested fix:** Chuẩn hóa markdown/bullet trước khi sinh answer, hoặc dùng LLM answer generation với output constraint rõ ràng.

### #4
- **Question:** Mẹ được nghỉ phép sinh con bao lâu?
- **Expected:** Mẹ được 29 tuần nghỉ phép sinh con, bao gồm cả kỳ nghỉ trước sinh.
- **Got:** `### Phép Sinh con:`
- **Worst metric:** Context Precision
- **Error Tree:** Output sai → Context đúng? `đúng section` → Query OK? `có` → Root cause: answer trả về heading của mục thay vì nội dung chi tiết.
- **Root cause:** Rerank / context selection đã vào đúng section, nhưng answer synthesis không ưu tiên dòng có dữ kiện định lượng.
- **Suggested fix:** Bỏ heading khỏi candidate answer, ưu tiên câu có con số / đơn vị, và giảm top-k context khi section đã rõ.

### #5
- **Question:** Cha được nghỉ phép sinh con bao nhiêu ngày?
- **Expected:** Cha được 5 ngày làm việc được hưởng lương 100% sau khi con được sinh.
- **Got:** `- **5 ngày làm việc** khi cha, mẹ k`
- **Worst metric:** Context Precision
- **Error Tree:** Output sai → Context đúng? `khá đúng` → Query OK? `có` → Root cause: câu trả lời đúng ý nhưng bị cắt ngắn và vẫn lẫn sang bullet khác.
- **Root cause:** Cụm trả lời bị cắt ở giữa bullet markdown; answer generation chưa có bước làm sạch/hoàn thiện câu.
- **Suggested fix:** Normalize bullet text, tách câu rõ hơn, và thêm post-processing để hoàn chỉnh số liệu + ngữ cảnh.

## Case Study (cho presentation)

**Question chọn phân tích:** Bảo hành laptop công ty là khoảng thời gian nào?

**Error Tree walkthrough:**
1. Output đúng? → Chưa đúng, vì model trả về heading `Thiết bị Công ty` thay vì câu có dữ kiện.
2. Context đúng? → Đúng phần lớn, context có section `Bão hành: 3 năm từ ngày nhân viên nhận`.
3. Query rewrite OK? → OK, query rõ và hỏi đúng một fact.
4. Fix ở bước: answer synthesis và post-processing context.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Tách heading khỏi body trước khi rank sentence.
- Thêm rule ưu tiên câu có số, đơn vị, và keyword khớp query.
- Giảm `DEFAULT_CONTEXT_TOP_K` / `ANSWER_CONTEXT_TOP_K` để bớt nhiễu.
- Dùng LLM generation thật cho bước cuối thay vì heuristic.
