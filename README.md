# Day 7 — Agentic RAG

> ## Bài làm — Nguyễn Khắc Khánh (2A202603472)
>
> **Corpus:** OpenStax *Astronomy 2e* (CC BY 4.0), ch.3–4, 56 trang, 265 chunk (index Day 6 `chunks_B`) · **Embedding:** `intfloat/multilingual-e5-small` 384d, local · **Vector store:** Chroma 1.5.9 của Day 6, cosine · **Agent:** `openai/gpt-4o-mini` qua OpenRouter, tool calling · **Retrieval:** dense, BM25, hybrid RRF, hybrid + cross-encoder rerank · **Cấu hình chọn:** rerank top-3 + chunk lân cận, budget 6000 ký tự · **Báo cáo:** [report/individual/2A202603472_NguyenKhacKhanh.md](report/individual/2A202603472_NguyenKhacKhanh.md)
>
> | | Baseline dense | Cấu hình chọn |
> |---|---|---|
> | Dev: context agent có evidence | 11/15 | **14/15** |
> | Test: grounded trung bình /2 | 1,8 | **2,0** |
> | Abstention đúng (dev / test) | 2/2 / 2/2 | 2/2 / 2/2 |
>
> ### Setup (Windows PowerShell, thư mục gốc repo)
>
> Cần repo Day 6 đã chạy `python -m src.run all` (có `starter/chroma_db`), đặt cạnh repo này hoặc khai báo đường dẫn bằng biến môi trường `DAY6_DIR` trỏ tới thư mục gốc repo Day 6.
>
> ```bash
> python -m venv .venv
> .venv\Scripts\Activate.ps1
> pip install -r requirements.txt
> Copy-Item .env.example .env    # điền OPENROUTER_API_KEY, OPENROUTER_MODEL=openai/gpt-4o-mini
> ```
>
> ### Chạy lại
>
> ```bash
> python -m src.export_day6                                   # Chroma Day 6 -> data/chunks.json, questions.json, cases.json
> python -m src.eval_retrieval --split dev --budget 4000      # 4 phương pháp retrieval (thêm --top-k 5 --budget 6000 --tag k5 | --budget 6000 --neighbors 1 --tag nb1)
> python -m src.agent --split dev --method dense              # agent; các cấu hình khác: --method bm25 | rerank [--top-k 5 --budget 6000 --tag k5] [--budget 6000 --neighbors 1 --tag nb1]
> python -m src.compare_agent --split dev                     # so sánh các lần chạy agent
> python -m src.agent --split test --method dense --top-k 3 --budget 4000 --tag baseline
> python -m src.agent --split test --method rerank --top-k 3 --budget 6000 --neighbors 1 --tag selected
> python starter/agent_metrics.py --cases data/cases.json --traces results/agent/test-rerank-selected/traces.json --reviews results/agent/test-rerank-selected/reviews.json --split test --output results/agent/test-rerank-selected/agent_summary.json
> $env:PYTHONUTF8=1; python -m unittest discover -s tests -v   # 22 test; cần UTF-8 vì một test của đề đọc file bằng encoding mặc định
> ```
>
> | Thành phần | Đường dẫn |
> |---|---|
> | Code | [src/](src/) (`export_day6`, `retrievers`, `eval_retrieval`, `agent`, `compare_agent`), test [tests/test_day7_agent.py](tests/test_day7_agent.py) |
> | Dữ liệu, benchmark | [data/chunks.json](data/chunks.json), [data/questions.json](data/questions.json), [data/unanswerable.json](data/unanswerable.json), [data/cases.json](data/cases.json) |
> | Tiêu chí chọn, phân tích dev, cấu hình chốt | [results/selection_criteria.md](results/selection_criteria.md), [results/dev_analysis.md](results/dev_analysis.md), [results/selected_config.json](results/selected_config.json) |
> | Kết quả test, nhãn | [results/test_results.md](results/test_results.md), `results/agent/test-*/reviews.json`, `agent_summary.json` |
> | Trace | `results/agent/*/traces.json` (tool call, context, câu trả lời, citation, token), `results/retrieval/*/traces.json` (top-10, score, hit) |
>
> `results/agent/dev-rerank-v0` là lần chạy trước khi sửa lỗi "từ chối khi chưa tìm"; `dev-rerank-smoke` là lần chạy thử 2 câu.

---

**Bài cá nhân · Python 3.11+ · tiếp nối Day 6**

Dùng embedding và vector store từ Day 6 để xây một agent biết tìm bằng chứng, trả lời có nguồn và nhận biết khi không đủ thông tin. Bạn tự chọn kiến trúc, số tool và cách tổ chức retrieval.

> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

## Bài cần làm

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

1. Kết nối dữ liệu, embedding và vector store Day 6.
2. Xây agent dùng model thật và ít nhất một khả năng truy xuất dữ liệu. Agent có thể dùng nhiều tool và nhiều lượt gọi.
3. Thử **ít nhất 3 phương pháp retrieval khác nhau**, trong đó có dense retrieval dùng vector store Day 6. Tự chọn các phương pháp còn lại; xem [gợi ý](GUIDE.md#2-thử-các-phương-pháp-retrieval).
4. Thử nhiều cấu hình, benchmark, phân tích lỗi và chọn cấu hình tốt trên **dev**; sau đó đánh giá trên **test**.

Benchmark tối thiểu có **24 câu**: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng; test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Xem [benchmark](BENCHMARK.md) và [rubric](RUBRIC.md).

## Thử thách nâng cao · +15 điểm

Index một tài liệu thực tế từ **100 trang** có bảng, ảnh và biểu đồ; xây agent truy xuất và trả lời các câu hỏi về nội dung đó. Hoàn thành [thử thách nâng cao](ADVANCED.md) và đạt **ít nhất 90% benchmark test** để nhận 15 điểm bonus. Phần cơ bản tối đa 85 điểm; toàn bài tối đa 100 điểm.

## Bắt đầu

1. Đọc [hướng dẫn](GUIDE.md), [rubric](RUBRIC.md) và [quy tắc](RULES.md).
2. Tạo repo cá nhân theo [cách nộp bài](SUBMISSION.md), chuẩn bị đầu ra Day 6 và model hỗ trợ tool calling.
3. Dùng hoặc thay thế [starter](starter/README.md); bạn cần kết nối model và vector store thật.
4. Chạy `python -m unittest discover -s tests -v`. Test mock chỉ kiểm tra code, không phải benchmark agent thật.

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

Nộp code, cách chạy, cấu hình, benchmark, trace và [báo cáo](report/individual/MSSV_HoTen.md). Thời gian ở [checkpoints](CHECKPOINTS.md) là gợi ý, có thể điều chỉnh theo lớp và phạm vi thử nghiệm.
