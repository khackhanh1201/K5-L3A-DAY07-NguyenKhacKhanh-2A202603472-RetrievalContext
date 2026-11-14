# Báo cáo Day 7 — Nguyễn Khắc Khánh (2A202603472)

Chi tiết số liệu: [dev_analysis.md](../../results/dev_analysis.md) · [test_results.md](../../results/test_results.md) · tiêu chí chọn [selection_criteria.md](../../results/selection_criteria.md) · cấu hình chốt [selected_config.json](../../results/selected_config.json).

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** dùng lại corpus Day 6: OpenStax *Astronomy 2e* (CC BY 4.0, Fraknoi, Morrison, Wolff et al., OpenStax/Rice University). Lấy chương 3 *Orbits and Gravity* (trang PDF vật lý 85–109) và chương 4 *Earth, Moon, and Sky* (117–147), tổng **56 trang nội dung**. Bản PDF dùng là `astronomy-2e_-_WEB.pdf` (1151 trang, CreationDate 2026-04-20). Index Day 6 `chunks_B` (chia theo đoạn, tối đa 800 ký tự) có **265 chunk**, được xuất bằng `python -m src.export_day6` ra [data/chunks.json](../../data/chunks.json). Mỗi chunk giữ `id`, `text`, `source_id`, `source`, `page`, `metadata.chapter` và `content_hash`.
- **Vector store, embedding model, phiên bản và số chiều:** Chroma **1.5.9** `PersistentClient` của Day 6 (`../K5-L3A-DAY06-…/starter/chroma_db`, có thể đổi bằng biến môi trường `DAY6_DIR`), collection `chunks_B`, metric cosine, index HNSW (tìm gần đúng). Embedding `intfloat/multilingual-e5-small` chạy local qua sentence-transformers 5.7.0 / torch 2.15.0 CPU, **384 chiều**, query có tiền tố `query: `, vector chuẩn hóa L2. `DenseRetriever` kiểm tra số chiều query bằng số chiều index trước khi tìm. Chi phí 0.
- **Agent model, cách chạy và phân chia dev/test:**
  - Model `openai/gpt-4o-mini` qua OpenRouter (SDK `openai` 3.51.0), tool calling, temperature 0, `max_tokens=1024`. Python 3.12.3.
  - Lệnh chạy: `python -m src.agent --split <dev|test> --method <...>`.
  - Benchmark có **24 câu**: dev gồm 15 câu có đáp án (q01–q15) + 2 câu thiếu evidence (u01, u02); test gồm 5 câu có đáp án (q16–q20) + 2 câu thiếu evidence (u03, u04).
  - Các câu có đáp án lấy nguyên từ Day 6, giữ đúng split cũ. Day 6 chỉ chạy test sau khi chốt cấu hình, nhưng báo cáo Day 6 đã phân tích lỗi của q16 và q20, nên test này **không hoàn toàn "mù"**. Các câu đó không được dùng để chỉnh cấu hình Day 7.
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - Câu có đáp án ghi `gold = [{source_id, page, evidence}]`. Exporter kiểm tra mỗi evidence nằm trọn trong ít nhất một chunk của `chunks_B`, sau khi chuẩn hóa NFKC, gộp khoảng trắng và lowercase (cùng quy tắc Day 6). Kết quả 20/20 hợp lệ. q04 có 2 evidence thay thế.
  - Câu thiếu evidence ([data/unanswerable.json](../../data/unanswerable.json)) được chọn gần chủ đề: mặt trăng của Sao Hỏa, giây nhuận, ngày nhật thực tiếp theo ở Việt Nam, khối lượng trạm ISS. Mỗi câu ghi các từ khóa đã tìm (không phân biệt hoa thường) trên toàn bộ 56 trang và kết quả 0 trang. u03 là bẫy: trang 143 nói danh sách nhật thực tương lai nằm ở Appendix H, ngoài corpus.
  - Một chunk là **hit** khi đúng nguồn, đúng trang và chứa trọn một evidence. Câu thiếu evidence không tính vào mẫu số của retrieval metrics.
  - Câu trả lời được chấm 0/1/2 và citation được đọc đối chiếu theo [BENCHMARK.md](../../BENCHMARK.md).

## Retrieval và thử nghiệm

Bốn phương pháp nằm trong [src/retrievers.py](../../src/retrievers.py), dùng chung interface `search(query, filters, top_k)`, trả về chunk kèm `score`, `rank` và `method`:
1. **Dense:** Chroma Day 6 + e5.
2. **BM25:** `rank-bm25` 0.2.2, k1=1.5, b=0.75, tách từ `\w+` sau khi chuẩn hóa, bỏ 36 stopword tiếng Anh.
3. **Hybrid RRF:** lấy 20 candidate từ dense và 20 từ BM25, gộp bằng `Σ 1/(60+rank)`. RRF không cộng score thô.
4. **Hybrid + rerank:** 20 candidate từ hybrid, xếp lại bằng cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`.

Ngoài ra có bước context **neighbors ±1**: kèm chunk liền kề trên cùng trang, xác định theo chỉ số đoạn trong ID. Bước này không đổi ranking dùng để tính Hit@k.

Filter chỉ lấy từ trường `filter` của câu hỏi. Có 4 câu có filter (q14–q17), tương ứng câu do người dùng nói rõ "In the chapter on…". Filter không bao giờ lấy từ gold, và model không tự đặt filter.

Bảng dưới gồm hai phần:
- **Dev:** Hit@1, Hit@3 và Context Evidence Rate của **retrieval một lượt với câu hỏi gốc** (15 câu), lệnh `python -m src.eval_retrieval`. Cột cuối là **Context Evidence Rate của agent** (context cuối gửi model).
- **Test:** 5 câu.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate (một lượt → agent) | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | Dense, top-3, filter từ câu hỏi, 4000 ký tự | phương pháp 1 (bắt buộc) | dev | 6/15 (40%) | 9/15 (60%) | 9/15 → **11/15** | retrieval ~45 ms; agent 2,85 s/câu, 1438 token prompt/câu, ~$0,0046 / 17 câu |
| Phương pháp 2 | BM25, top-3, 4000 | lexical bắt được số và thuật ngữ hiếm | dev | 11/15 (73%) | 12/15 (80%) | 12/15 → 12/15 | ~1 ms; agent 2,82 s, 1421 tok, ~$0,0045 |
| Phương pháp 3 | Hybrid RRF dense+BM25 (20+20, k=60), top-3, 4000 | gộp hai tín hiệu | dev | 7/15 (47%) | 9/15 (60%) | 9/15 (không chạy agent) | ~20 ms |
| Cải thiện 1 (phương pháp 4) | Hybrid + cross-encoder rerank (20 candidate), top-3, 4000 | rerank sửa lỗi xếp hạng giữa các đoạn gần nghĩa | dev | 10/15 (67%) | **14/15 (93%)** | 14/15 → 13/15 | ~800 ms/query (CPU); agent 3,01 s, 1383 tok, ~$0,0044 |
| Cải thiện 2 | Rerank, **top-5**, 6000 | evidence ở hạng 4–5 sẽ vào context | dev | 10/15 | 14/15 | 14/15 → 13/15 | agent 3,11 s, 1704 tok, ~$0,0053 |
| Cải thiện 3 | Rerank, top-3 + **neighbors ±1**, 6000 | evidence nằm ở chunk kề chunk được lấy | dev | 10/15 | 14/15 | 14/15 → **14/15** | agent 3,39 s, 1920 tok, ~$0,0060 |
| Cấu hình đã chọn | = Cải thiện 3 | chốt trên dev | test | 5/5 | 5/5 | 5/5 → 5/5 | ~530 ms retrieval; agent 4,0 s, 1702 tok, ~$0,0022 / 7 câu |
| Baseline | = Baseline | | test | 2/5 | 2/5 | 2/5 → 5/5 | ~23 ms; agent 3,1 s, 1182 tok, ~$0,0016 / 7 câu |

Thời gian là trung vị, đo trên CPU laptop, chưa tính thời gian tải model lần đầu (17–30 s). Chi phí ước tính theo giá gpt-4o-mini tham khảo là $0,15 / $0,60 cho mỗi triệu token input/output; số token thật nằm trong trace. Embedding, BM25 và reranker chạy local nên không tốn phí.

**Lỗi dev, nguyên nhân, cách thử, lý do chọn:**
1. **Agent từ chối khi chưa tìm** (lần chạy v0: q08, q10, q12, u02). Ở bước đầu model có sẵn cả tool `answer` và `abstain`, nên nó gọi `abstain` ngay, dù retrieval tìm thấy evidence của q08, q10, q12.
   - Cách sửa: bước đầu chỉ cung cấp tool `search`.
   - Kết quả: tool use 13 → 17/17, context 10 → 13/15, từ chối nhầm 5 → 2. u02 vẫn được từ chối đúng, nhưng giờ là sau khi đã tìm.
2. **q06: đúng trang, sai chunk.** Agent viết lại query thành "Kepler third law masses of the Sun and planet". Rerank lấy `p99:par800:1` (giới thiệu định luật 3 có khối lượng), còn đáp án nằm ở `p99:par800:2` ("How did Kepler miss this factor?…").
   - Top-5 vẫn không lấy được chunk này, nên **giả thuyết của cải thiện 2 bị bác**.
   - Neighbors ±1 kéo `p99:2` theo `p99:1`, agent trả lời đúng với citation [p99:1][p99:2], nên **giả thuyết của cải thiện 3 được xác nhận**.
3. **q15: trang gold không được lấy.** Chunk gold `p141:par800:1` bắt đầu bằng "These show that the Moon is moving away by 3.8 centimeters per year". Cả ba query của agent chỉ lấy được trang 134/136/143, và agent còn lặp lại cùng một query hai lần. Neighbors không giúp được vì không chunk nào của trang 141 được lấy. Lỗi này chưa sửa, xem phần giới hạn.
4. **Citation chỉ liệt kê, không chèn inline** (q01 trong smoke test). Model trả lời đúng nhưng không viết `[S1]` trong câu.
   - Cách sửa: nối nhãn vào cuối câu trả lời và ghi cờ `citations_appended` trong trace. Citation không có trong context vẫn bị loại.

**Lý do chọn:** áp dụng tiêu chí đã chốt trước khi chạy cải thiện 2 và 3, theo thứ tự:
- (1) Bắt buộc abstention 2/2, tool use 17/17 và 0 lỗi: tất cả cấu hình đều đạt.
- (2) Context Evidence Rate của agent: cải thiện 3 đạt 14/15, cao nhất.
- (3) Vì chỉ hơn rerank top-3 (13/15) đúng 1 câu, xét luật chi phí: rerank top-3 rẻ hơn 28% token, dưới ngưỡng 30%, nên giữ cải thiện 3.

**Kết luận:** hybrid RRF trọng số đều *kém* BM25, vì dense yếu kéo ranking xuống; RRF chỉ có ích khi đi kèm rerank. Agent giúp dense (một lượt 9/15 → agent 11/15) nhờ tự viết lại query, nhưng lại làm rerank mất q06 (một lượt 14/15 → agent 13/15).

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi** ([src/agent.py](../../src/agent.py), viết lại từ `starter/agentic.py`):
  - **Tool:** `search(query)`, `answer(answer, citations)`, `abstain(reason)`. Bước đầu chỉ có `search`.
  - **Số lượt tìm:** tối đa `max_searches=3`; sau đó tool `search` bị gỡ và model buộc phải trả lời hoặc từ chối. Ngoài ra có giới hạn 5 bước để tránh vòng lặp. Mức 3 được chọn vì trên dev, câu đã có evidence thường được trả lời sau 1 lượt; các câu cần 3 lượt (q15, u01) đều kết thúc bằng từ chối.
  - **Context:** gộp kết quả của mọi lượt tìm và bỏ trùng. Chunk được xếp xen kẽ theo rank (hạng 1 của mỗi lượt trước, rồi hạng 2…), để lượt tìm sau vẫn thêm được evidence, rồi cắt theo budget ký tự. Chunk hiển thị cho model dưới nhãn `[S1]`, `[S2]`…; trace đổi lại thành chunk ID thật.
  - **Citation:** phải có trong context cuối. Nếu không, lần chạy được ghi là abstain với lý do `invalid_citations`, và câu trả lời gốc vẫn được giữ lại để chấm.
  - **Lỗi provider:** 429, 5xx và lỗi mạng được thử lại tối đa 5 lần. 400/401/402/403/404 dừng cả lượt chạy ngay. Lỗi được ghi `status: "error"`, không thay bằng kết quả giả.
  - **Prompt:** system prompt ghi rõ nội dung tài liệu là dữ liệu, không phải chỉ dẫn.

| Cấu hình / split | Tool use | Số lượt gọi (search: số câu) | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Baseline dense / dev | 17/17 | 1: 14 · 2: 1 · 3: 2 | chưa chấm tay; tự động: 11/15 trả lời có citation gold | – | 12/12 | 2/2 | 3/15 (q06, q14, q15) |
| Đã chọn / dev | 17/17 | 1: 15 · 3: 2 | chưa chấm tay; tự động: 14/15 trả lời có citation gold | – | 14/14 | 2/2 | 1/15 (q15) |
| Baseline dense / test | 7/7 | 1: 7 | 0/1/4 → **1,8** | 5/5 | 5/5 | 2/2 | 0/5 |
| Đã chọn / test | 7/7 | 1: 7 | 0/0/5 → **2,0** | 6/6 | 5/5 | 2/2 | 0/5 |

Chấm context trên test, mức 0/1/2, cho 5 câu × 2 cấu hình: tất cả đạt 2 (đủ evidence và nguồn).

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q17 (test, đã chọn) | có đáp án, filter chương 3 | 1 search "closest and farthest points in the orbit of an object circling Earth" → p101:1 hạng 1 | "…perigee, … apogee [astronomy2e-ch03:p101:par800:1]" | **2**: chunk có nguyên văn "For the Moon or a satellite orbiting Earth… perigee and apogee" |
| u03 (test, đã chọn) | thiếu evidence | 1 search "total solar eclipse Vietnam date" → các trang về nhật thực (2017/2024 ở Mỹ) | từ chối: "context does not contain … Vietnam" | **abstention đúng**: không đưa ra ngày nào; corpus chỉ trỏ tới Appendix H, ngoài phạm vi index |
| q20 (test, baseline) | có đáp án | 1 search "lunar eclipses more often than solar eclipses" → p144:2 | đúng ý chính [p144:2], nhưng thêm "more people can witness… making them more frequent from any given location" | **1**: phần thêm không có trong nguồn và lập luận sai (tần suất *tại một nơi* không phụ thuộc số người xem). Cấu hình đã chọn trả lời đủ với [p144:2][p145:2] → 2 |
| q06 (dev, đã chọn) | có đáp án | 1 search → p99:1 hạng 1; neighbors thêm p99:2 | "…mass of the Sun is 1 and the mass of a typical planet is negligibly small… [p99:1], [p99:2]" | ví dụ cải thiện nhờ neighbors; baseline và rerank top-3 từ chối nhầm câu này |
| q15 (dev, đã chọn) | có đáp án, filter chương 4 | 3 search (một query bị lặp) → trang 134/136/143, không có 141 | từ chối | **lỗi**: từ chối nhầm vì retrieval không lấy được trang gold |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Trace agent: `results/agent/<split>-<method>[-tag]/traces.json`. Mỗi câu ghi từng search (query, filter, ID, nhãn, rank, score, trang, `neighbor_of`, thời gian retrieval, token LLM), câu trả lời gốc, citation, context cuối, `label_map`, token và chi phí.
  - Tóm tắt agent: `summary.json` trong cùng thư mục. So sánh các lần chạy: `results/agent/comparison-<split>.json`.
  - Nhãn test: `results/agent/test-*/reviews.json` (có lý do cho từng câu) và `agent_summary.json`, được tổng hợp bằng `starter/agent_metrics.py`.
  - Retrieval: `results/retrieval/<split>-<method>[-tag]/{traces,summary}.json`.
  - Lệnh chạy đầy đủ ở [README](../../README.md) và [test_results.md](../../results/test_results.md).
  - Tổng chi phí LLM của mọi lần chạy khoảng $0,03–0,04 theo giá tham khảo; mỗi lần chạy dev mất khoảng 1 phút.
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - **Trước/sau:** dev context 11/15 (baseline) → 14/15 (đã chọn); từ chối nhầm 3 → 1. Test grounded 1,8 → 2,0. Cái giá phải trả là token prompt tăng khoảng 34–44% và thêm khoảng 0,5–0,9 s/câu (reranker chạy CPU).
  - **Giới hạn:**
    - Test chỉ 7 câu, cả hai cấu hình gần điểm tối đa, nên chênh lệch 1 câu không có ý nghĩa thống kê.
    - Dev chỉ có 15 câu, và chênh lệch giữa các cấu hình rerank cũng chỉ 1 câu.
    - Grounded và citation trên dev chưa được chấm tay; dev chỉ dùng chỉ số tự động "citation trỏ tới chunk chứa evidence".
    - Nhãn test do Claude chấm nháp bằng cách đọc câu trả lời và chunk được trích; tôi đã đọc lại và xác nhận.
    - Evidence Hit chỉ nhận khớp nguyên văn trong một chunk, nên evidence tương đương ở trang khác bị đánh giá thấp (ví dụ p145 ở q20).
    - Các câu test lấy từ Day 6 và đã được phân tích lỗi ở Day 6.
    - Trích PDF còn nhiễu: header/footer, công thức bị mất (ví dụ "eccentricity … is only about ;").
    - gpt-4o-mini qua OpenRouter không cố định phiên bản backend, nên chạy lại có thể khác chút ít dù temperature 0.
    - q15 cho thấy agent chưa biết đổi hướng khi query lặp lại.

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện.
