# Kết quả test (bước 6)

Chạy **sau khi chốt cấu hình trên dev** ([selected_config.json](selected_config.json)). Test gồm 5 câu có đáp án (q16–q20) + 2 câu thiếu evidence (u03, u04). Model `openai/gpt-4o-mini`, temperature 0.

> Nhãn trong `reviews.json` là **nhãn nháp** do Claude chấm bằng cách đọc câu trả lời và chunk được trích; người làm bài cần đọc lại và xác nhận trước khi nộp.

## Lệnh

```
python -m src.agent --split test --method dense --top-k 3 --budget 4000 --tag baseline
python -m src.agent --split test --method rerank --top-k 3 --budget 6000 --neighbors 1 --tag selected
python -m src.eval_retrieval --split test --methods dense --budget 4000 --tag baseline
python -m src.eval_retrieval --split test --methods rerank --budget 6000 --neighbors 1 --tag selected
python -m src.compare_agent --split test
python starter/agent_metrics.py --cases data/cases.json --traces results/agent/<run>/traces.json --reviews results/agent/<run>/reviews.json --split test --output results/agent/<run>/agent_summary.json
```

## Retrieval một lượt (query gốc)

| Cấu hình | Hit@1 | Evidence Hit@3 | MRR@10 | Context Evidence Rate | ms/query |
|---|---|---|---|---|---|
| Baseline dense, top-3, 4000 | 2/5 (40%) | 2/5 (40%) | 0.497 | 2/5 | ~23 |
| Chọn: rerank, top-3 + neighbors, 6000 | 5/5 (100%) | 5/5 (100%) | 1.000 | 5/5 | ~530 |

## Agent

| Cấu hình | Tool use | Lượt tìm | Context có evidence (chấm 0/1/2) | Grounded 0/1/2 (TB) | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm | Token prompt/câu | s/câu | $ ước tính (7 câu) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline dense | 7/7 | 1 cho mọi câu | 5/5 (0/0/5) | 0/1/4 → **1.8** | 5/5 | 5/5 | 2/2 | 0/5 | 1182 | 3.1 | 0.0016 |
| Chọn: rerank + neighbors | 7/7 | 1 cho mọi câu | 5/5 (0/0/5) | 0/0/5 → **2.0** | 6/6 | 5/5 | 2/2 | 0/5 | 1702 | 4.0 | 0.0022 |

Giá ước tính: $0.15 / $0.60 cho 1M token vào/ra (cần đối chiếu giá OpenRouter). Không có lỗi provider.

## Nhãn từng câu

| Query | Loại | Tool call → kết quả | Câu trả lời, citation | Baseline | Chọn |
|---|---|---|---|---|---|
| q16 | có đáp án (filter ch.4) | 1 search → p121:2 trong context | "about 3%" [p121:2] | 2 | 2 |
| q17 | có đáp án (filter ch.3) | 1 search → p101:1 | perigee / apogee [p101:1] | 2 | 2 |
| q18 | có đáp án | 1 search → p103:0 | e ≥ 0.8 + định luật 2 Kepler [p103:0] | 2 | 2 |
| q19 | có đáp án | 1 search → p142:2 | nghiêng ~5°, eclipse season [p142:2] | 2 | 2 |
| q20 | có đáp án | 1 search → p144:2 (+p145:2 ở cấu hình chọn) | nửa đêm Trái Đất vs vùng hẹp | **1**: thêm suy diễn sai "more people can witness… making them more frequent" | 2 [p144:2][p145:2] |
| u03 | thiếu evidence | 1 search "total solar eclipse Vietnam date" | từ chối | đúng | đúng |
| u04 | thiếu evidence | 1 search "mass of the International Space Station" | từ chối | đúng | đúng |

## Nhận xét

- Agent bù được retrieval yếu của dense: Hit@3 một lượt chỉ 2/5 nhưng context agent 5/5, vì model viết lại câu hỏi thành query ngắn, nhiều từ khóa (ví dụ q20: "lunar eclipses more often than solar eclipses").
- Cấu hình chọn hơn baseline ở chất lượng câu trả lời (2.0 so với 1.8, khác biệt duy nhất ở q20), tốn thêm ~44% token prompt và ~0.9 s/câu.
- Test chỉ 7 câu và cả hai cấu hình gần trần; chênh lệch 1 câu không đủ kết luận thống kê. Kết luận chính vẫn dựa trên dev (context 14/15 so với 11/15).
- Day 6 đã phân tích lỗi trên chính các câu test này (q16, q20) nhưng không dùng chúng để chỉnh cấu hình Day 7; vẫn cần ghi rõ trong báo cáo vì đây không phải test hoàn toàn "mù".
