# Tiêu chí chọn cấu hình (chốt trước khi chạy các cấu hình cải thiện, 2026-11-14)

Chỉ dùng **dev** (15 câu có đáp án, 2 câu thiếu evidence). Test chưa được chạy.

Áp dụng theo thứ tự, chỉ chuyển sang tiêu chí sau khi hòa:

1. **Điều kiện bắt buộc:** abstention đúng 2/2 trên câu thiếu evidence, tool use 17/17, không lỗi hệ thống.
2. **Agent Context Evidence Rate** cao nhất: số câu có đáp án mà context cuối gửi model chứa evidence gold / 15.
3. **Trả lời có căn cứ** cao nhất: số câu có đáp án được trả lời (không từ chối) *và* có ít nhất một citation là chunk chứa evidence gold / 15. Đây là chỉ số tự động thay thế tạm; điểm grounded 0/1/2 sẽ chấm tay trên test.
4. **Chi phí:** ít token prompt trung bình hơn, rồi thời gian trung vị thấp hơn. Nếu tiêu chí 2–3 chênh ≤ 1 câu (≈ 7%), ưu tiên cấu hình rẻ hơn ít nhất 30% token.

Cố định giữa các cấu hình: corpus `chunks_B` (265 chunk), model `openai/gpt-4o-mini`, temperature 0, `max_searches=3`, filter lấy từ câu hỏi.
