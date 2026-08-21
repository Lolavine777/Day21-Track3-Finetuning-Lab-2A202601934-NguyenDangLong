# Reflection - Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Sự sụt giảm nghiêm trọng của năng lực phổ quát (regression score tụt 28% từ 0.758 xuống 0.478) ngay cả khi target accuracy đạt tới 97.0%.
Nếu chỉ đánh giá bằng train loss hoặc target accuracy thông thường, tôi sẽ hoàn toàn không phát hiện ra mô hình đã bị quên thảm họa (Catastrophic Forgetting) đối với các tri thức phổ thông.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Thời gian nhiều nhất nằm ở giai đoạn sinh suy luận (generation/evaluation) và huấn luyện 3 run đối chứng ở NB4 (chiếm gần 48 phút trên T4 GPU).
Điều này đúng như dự đoán vì việc đánh giá 3 baseline và chấm chéo trên 4 nhóm chỉ số đòi hỏi sinh toàn bộ tập test nhiều lần một cách chặt chẽ.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước đây tôi tin rằng nâng rank $r$ càng cao thì mô hình càng mạnh và train loss càng thấp thì mô hình càng thông minh.
Thực nghiệm đã chứng minh run `attn_only` với rank $r=283$ ép train loss xuống 0.5378 (thấp hơn `correct` 0.6258) nhưng độ chính xác target thực tế lại kém hơn bản $r=16$ all-linear (0.965 vs 0.970).

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi sử dụng AI assistant để rà soát kiến trúc mã nguồn, hỗ trợ xây dựng giao diện demo Gradio và tự động hóa các kịch bản kiểm tra đối chứng.
Chỗ AI hay mắc lỗi nhất là thói quen mặc định `bf16=True` theo các bài hướng dẫn trên A100 mà không nhận diện phần cứng Turing T4 không hỗ trợ native bfloat16, dẫn đến lỗi unscale của GradScaler nếu không chủ động ép kiểu sang fp16/fp32.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Bước đầu tiên là đóng băng tập dữ liệu đánh giá đại diện và đo Baseline B với Prompt tối ưu kỹ lưỡng trước khi huấn luyện để chứng minh giá trị kinh tế của việc fine-tune; tiếp đó là giải mã ngược Loss Mask để đảm bảo 100% không tính loss vào phần câu hỏi của khách hàng.
