# Thử nghiệm trên dev và chọn cấu hình (bước 5)

Tiêu chí chọn được chốt trước trong [selection_criteria.md](selection_criteria.md). Test **chưa chạy**.
Cố định: corpus Day 6 `chunks_B` (265 chunk, 56 trang), embedding `intfloat/multilingual-e5-small` 384d, Chroma 1.5.9 cosine,
LLM `openai/gpt-4o-mini` qua OpenRouter, temperature 0, `max_searches=3`, `max_tokens=1024`, filter lấy từ trường `filter` của câu hỏi.

## 1. Retrieval một lượt, query gốc (15 câu dev có đáp án)

Lệnh: `python -m src.eval_retrieval --split dev --methods dense bm25 hybrid rerank --budget 4000`
(+ `--top-k 5 --budget 6000 --tag k5`, `--budget 6000 --neighbors 1 --tag nb1`). Không gọi LLM, chi phí 0.

| Phương pháp | Hit@1 | Evidence Hit@3 | MRR@10 | Context (top-3, 4000 ký tự) | Context top-5 / 6000 | Context top-3 + neighbors / 6000 | ms/query (trung vị) |
|---|---|---|---|---|---|---|---|
| Dense (Chroma Day 6) | 6/15 (40%) | 9/15 (60%) | 0.509 | 9/15 | 10/15 | 10/15 | ~45 |
| BM25 | 11/15 (73%) | 12/15 (80%) | 0.767 | 12/15 | 12/15 | 12/15 | ~1 |
| Hybrid RRF (dense+BM25, 20+20, k=60) | 7/15 (47%) | 9/15 (60%) | 0.580 | 9/15 | – | – | ~20 |
| Hybrid + rerank (cross-encoder ms-marco-MiniLM-L-6-v2, 20 candidate) | 10/15 (67%) | **14/15 (93%)** | **0.796** | **14/15** | 14/15 | 14/15 | ~800 |

Dense trùng kết quả Day 6 chiến lược C (Hit@1 6/15, Hit@3 9/15), xác nhận kết nối đúng index.

## 2. Agent (17 câu dev: 15 có đáp án + 2 thiếu evidence)

Lệnh: `python -m src.agent --split dev --method <m> [...]`, so sánh: `python -m src.compare_agent --split dev`.
Chi phí ước tính theo $0.15 / $0.60 cho 1M token vào/ra (cần đối chiếu giá OpenRouter).

| Cấu hình | Giả thuyết / thay đổi | Tool use | Lượt tìm TB | Context có evidence | Trả lời + citation gold | Từ chối nhầm | Abstention đúng | Token prompt/câu | s/câu | $ (17 câu) |
|---|---|---|---|---|---|---|---|---|---|---|
| v0: rerank, top-3 (cho phép từ chối ngay) | bản đầu | 13/17 | 1.00 | 10/15 | 10/15 | 5/15 | 2/2 | 1251 | 2.80 | 0.0040 |
| **Baseline: dense, top-3, 4000** | phương pháp 1 | 17/17 | 1.29 | 11/15 | 11/15 | 3/15 | 2/2 | 1438 | 2.85 | 0.0046 |
| BM25, top-3, 4000 | phương pháp 2 | 17/17 | 1.29 | 12/15 | 12/15 | 2/15 | 2/2 | 1421 | 2.82 | 0.0045 |
| Rerank, top-3, 4000 | cải thiện 1: cross-encoder sửa lỗi xếp hạng của dense | 17/17 | 1.24 | 13/15 | 13/15 | 2/15 | 2/2 | 1383 | 3.01 | 0.0044 |
| Rerank, top-5, 6000 | cải thiện 2: gold ở hạng 4–5 sẽ vào context | 17/17 | 1.24 | 13/15 | 13/15 | 2/15 | 2/2 | 1704 | 3.11 | 0.0053 |
| **Rerank, top-3 + neighbors ±1, 6000** | cải thiện 3: evidence nằm ở chunk kề của chunk được lấy | 17/17 | 1.24 | **14/15** | **14/15** | **1/15** | 2/2 | 1920 | 3.39 | 0.0060 |

"Trả lời + citation gold" là chỉ số tự động (citation trỏ tới chunk chứa evidence), chưa phải điểm grounded chấm tay.

## 3. Phân tích lỗi dev

1. **Từ chối khi chưa tìm (v0: q08, q10, q12, u02).** Model gọi `abstain` ngay bước đầu, trong khi retrieval một lượt tìm thấy evidence của q08/q10/q12. Nguyên nhân: bước đầu cung cấp cả `answer`/`abstain`.
   Sửa: bước đầu chỉ có tool `search`. Kết quả: tool use 13→17/17, context 10→13/15, từ chối nhầm 5→2. u02 vẫn từ chối đúng nhưng nay sau khi đã tìm.
2. **Citation chỉ liệt kê, không chèn inline (q01, smoke test).** Model trả lời đúng nhưng không viết `[S1]` trong câu, nên bị kiểm tra citation loại.
   Sửa: nối nhãn vào cuối và ghi cờ `citations_appended` trong trace; citation ngoài context vẫn bị loại.
3. **q06 — đúng trang, sai chunk.** Agent viết lại query thành "Kepler third law masses of the Sun and planet"; rerank lấy `p99:par800:1` (giới thiệu khối lượng trong định luật 3), còn câu trả lời ở `p99:par800:2` ("How did Kepler miss this factor?…"). Ở top-5 chunk này vẫn không xuất hiện → **giả thuyết top-5 bị bác**. Với neighbors ±1, `p99:2` được kéo theo `p99:1` → agent trả lời đúng có citation → **giả thuyết neighbors được xác nhận**.
4. **q15 — trang gold không được lấy.** Gold `p141:par800:1` ("These show that the Moon is moving away by 3.8 centimeters per year"). Câu hỏi diễn đạt "distance… changing, how fast"; cả 3 query của agent chỉ lấy trang 134/136/143. Neighbors không giúp vì không chunk nào của trang 141 được lấy (giả thuyết "chunk kề p141:0" sai). Agent còn lặp lại cùng một query ("Moon distance from Earth changing rate") hai lần. Hướng tiếp theo (chưa thử): gợi ý model dùng thuật ngữ khác khi query lặp, hoặc thêm dense vào candidate của rerank (dense một lượt xếp gold hạng 3).
5. **q14 (chỉ dense) — nhiễu cùng chương.** Dense lấy trang 138 (lực thủy triều) thay vì 140 (angular momentum), giống phân tích Day 6. BM25 và rerank xếp đúng trang 140 hạng 1.

Quan sát thêm: agent giúp dense (context 11/15 so với 9/15 một lượt) nhờ tự viết lại query, nhưng làm hỏng q06 với rerank (một lượt 14/15 → agent 13/15). Hybrid RRF trọng số đều kém BM25 vì dense yếu kéo xếp hạng xuống; chỉ có ích khi đi kèm rerank.

## 4. Quyết định

Áp dụng tiêu chí đã chốt:
1. Điều kiện bắt buộc: mọi cấu hình (trừ v0) đạt abstention 2/2, tool use 17/17, 0 lỗi.
2. Context Evidence Rate: **rerank + neighbors 14/15** > rerank 13/15 = rerank top-5 13/15 > BM25 12/15 > dense 11/15.
3. Chênh lệch với rerank top-3 là 1 câu nên xét luật chi phí: rerank top-3 dùng ít hơn 28% token prompt (1383 so với 1920), **dưới ngưỡng 30%** → giữ kết quả bước 2.

**Cấu hình chọn:** `--method rerank --top-k 3 --budget 6000 --neighbors 1` (xem [selected_config.json](selected_config.json)).
**Baseline cho test:** `--method dense --top-k 3 --budget 4000`.

Đánh đổi: so với baseline, tăng 3 câu có evidence trong context (11→14/15), tốn thêm ~34% token prompt, ~0.5 s/câu (rerank ~0.8 s/query trên CPU), chi phí ~$0.0060 so với $0.0046 cho 17 câu. Chênh lệch giữa các cấu hình rerank chỉ 1 câu trên 15, chưa đủ ý nghĩa thống kê.
