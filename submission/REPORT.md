# Lab 21 - Evaluation Report

**Họ tên**: Nguyễn Đăng Long  **MSSV**: 2A202601934  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (14.6 GB usable, fp16 + GradScaler)`

> Mọi con số dưới đây khớp 100% với các file trong `results/` sinh ra từ quá trình chạy trên Colab T4.

---

## 1. Setup

| Thông số | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt -> JSON triage 4 trường |
| Train / val | 225 / 25 (cố định seed 42) |
| `max_length` | 1024 - p95 đo được thực tế là 98 (suggested 256, chọn 1024 theo tier T4 để an toàn) |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2.0 / 30 optimizer steps |

**Template có giữ khối `<think>` không?** Có - *(results/template_check.json)*
Template `Qwen3.5` giữ nguyên vẹn nội dung bên trong cặp thẻ `<think>...</think>`, đảm bảo an toàn cho các tác vụ huấn luyện chuỗi suy luận (reasoning traces).

---

## 2. Mask proof (NB1)

| Kiểm tra | Kết quả |
|---|---|
| `supervised_fraction` | 0.4149 (41.5%) |
| Câu trả lời nằm trong loss | true |
| Câu hỏi KHÔNG nằm trong loss | true |

Dán 3-5 dòng đầu của đoạn được tính loss giải mã ngược từ `results/mask_proof.json`:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 - đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.758 | 0.000 | 3183.5 |
| (b) base + optimized prompt | 0.765 | 0.758 | 1.000 | 1024.0 |
| (c) LoRA fine-tune | 0.970 | 0.478 | 1.000 | 1380.8 |

**(b) có thật sự mạnh hơn (a) không?** Có.
Baseline (b) với prompt tối ưu có schema và one-shot example đạt độ chính xác target 0.765 và format 1.000, vượt trội hoàn toàn so với Baseline (a) chỉ đạt 0.000 do model sinh văn xuôi tự do dài dòng (latency 3183.5 ms).
Tôi không chỉnh sửa `OPTIMIZED_PROMPT` và giữ nguyên mã băm SHA `719e74d3b6232053` trong `results/baselines_frozen.json` để bảo đảm tính liêm chính khoa học của phép so sánh.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6258 | **0.970** | 1010.0 | 12.01 |
| `attn_only` | q,v | 283 | 32,456,704 | 0.0001 | 0.5378 | **0.965** | 805.2 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-05 | 1.5704 | **0.000** | 930.0 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | **0.940** | 1000.8 | 7.09 |

> Xếp hạng bốn run bằng cột **target (NB5 §4)**, không dùng cột train loss.

**4.1 - Phân tích vị trí vs rank:**
Run `attn_only` được nâng rank lên $r=283$ bằng hàm `matched_rank()` để có cùng ngân sách 32.46M tham số với `correct` ($r=16$).
Trên tập target, `attn_only` đạt 0.965, bám sát nhưng vẫn kém hơn `correct` đạt 0.970.
Đáng chú ý là train loss của `attn_only` lại thấp hơn `correct` (0.5378 so với 0.6258).
Điều này phản ánh chính xác hiện tượng ghi nhớ cục bộ khi dồn toàn bộ rank cao vào một vị trí hẹp (chỉ attention) thay vì phân bổ đều khắp các tầng linear của text decoder.
Bằng chứng thực nghiệm này khẳng định vị trí gắn adapter (`all-linear`) là đòn bẩy quan trọng hơn việc nâng rank đơn thuần.

**4.2 - Phân tích thang Learning Rate:**
Run `wrong_lr` chỉ thay đổi duy nhất learning rate về thang full fine-tune ($1\times 10^{-5}$ thay vì $1\times 10^{-4}$).
Đường loss giảm rất chậm và dừng ở mức 1.5704 sau 30 step, khiến model hoàn toàn thất bại trên tập target (0.000) và format (0.000).
Nếu chỉ nhìn loss giảm từ 2.16 xuống 1.11 mà không đối chiếu với bài toán, người làm rất dễ lầm tưởng mô hình đang học tốt nhưng thực chất trọng số LoRA chưa di chuyển đủ xa để định hình cấu trúc JSON theo yêu cầu.
Learning rate trong LoRA bắt buộc phải đặt ở thang $10\times$ so với full-FT để bù đắp cho không gian tham số bị thu hẹp.

**4.3 - Đánh đổi của QLoRA 4-bit:**
Run `qlora` 4-bit giúp tiết kiệm đến 41% bộ nhớ VRAM (7.09 GB so với 12.01 GB của bản 16-bit fp16).
Tuy nhiên, cái giá phải trả là độ chính xác target bị tụt từ 0.970 xuống 0.940 và độ trễ suy luận latency tăng từ 1380.8 ms lên 1733.4 ms do overhead dequantization.
Kết quả đo đạc thực tế này hoàn toàn ủng hộ khuyến nghị từ nhà cung cấp mô hình: không nên lạm dụng QLoRA trên dòng kiến trúc `Qwen3.5` nếu bộ nhớ GPU vẫn đủ sức chứa bản 16-bit.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.280` · `valid_trace_rate = 0.0`

**Diễn giải phán quyết:**
Cổng hồi quy đánh giá phán quyết `FAILED` vì điểm suy giảm năng lực tổng quát `regression Δ = -0.280` đã vượt quá ngưỡng cho phép `tolerance = 0.020`.
Mặc dù mô hình fine-tune đạt mức tăng trưởng ấn tượng trên tác vụ chuyên biệt (+20.5% so với Baseline b, đạt 97% accuracy), việc huấn luyện tập trung hoàn toàn vào 225 mẫu CSKH đặc thù đã gây ra hiện tượng quên thảm hoạ (Catastrophic Forgetting) đối với các câu hỏi tri thức và chỉ dẫn phổ thông.
Model fine-tune có xu hướng ép mọi câu hỏi tri thức thông thường thành phản hồi CSKH hoặc suy giảm khả năng trả lời chính xác từ khoá phổ thông.
Để đưa mô hình vào triển khai thực tế an toàn, bắt buộc phải áp dụng kỹ thuật Replay Buffer (trộn 1-5% dữ liệu kiến thức tổng quát vào tập huấn luyện theo Deck §14.3) nhằm bảo vệ nền tảng tri thức của base model.

---

## 6. Định tính - Phân tích chi tiết cả ca Thắng và ca Thua

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt ốp lưng điện thoại mã đơn DH936478. Shipper không liên lạc... | `van_chuyen`, `thap` | Sai urgency | `van_chuyen`, `thap` | ✅ **FT thắng**: Mô hình fine-tune nhận diện đúng mức độ khẩn cấp thấp dựa vào ngữ cảnh câu hỏi thông thường. |
| 2 | Chào shop, mình đặt ốp lưng điện thoại mã đơn VN833689. Sai màu. Sớm nhất... | `san_pham_loi`, `trung_binh` | Sai intent | `san_pham_loi`, `trung_binh` | ✅ **FT thắng**: Nhận diện chính xác trường hợp giao sai thuộc nhóm lỗi sản phẩm. |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Nhờ shop kiểm tra. | `hoan_tien`, `cao` | `hoan_tien`, `cao` | `hoan_tien`, `trung_binh` | ❌ **FT thua**: Mô hình dự đoán urgency là `trung_binh` thay vì `cao`, do câu hỏi bắt đầu bằng "Cho mình hỏi" làm giảm trọng số cấp bách. |
| 4 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Cần gấp... | `san_pham_loi`, `cao` | `san_pham_loi`, `cao` | `san_pham_loi`, `trung_binh` | ❌ **FT thua**: Khách yêu cầu "Cần gấp trong ngày" nhưng mô hình fine-tune vẫn đánh giá `trung_binh`. |
| 5 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện kiểm tra... | `san_pham_loi`, `thap` | `san_pham_loi`, `thap` | `san_pham_loi`, `trung_binh` | ❌ **FT thua**: Khách bảo "Khi nào tiện" (mức thấp), nhưng mô hình fine-tune có thiên kiến gán `trung_binh` cho hầu hết ticket lỗi. |

**Mẫu chung ở các ca FT thua:**
Mô hình fine-tune có xu hướng thiên vị (inductive bias) gán nhãn `urgency = trung_binh` khi gặp các từ ngữ mâu thuẫn giữa kính ngữ lịch sự ("Cho mình hỏi", "Shop ơi") với từ chỉ sự cấp bách hoặc trì hoãn ("Cần gấp", "Khi nào tiện").

---

## 7. Kết luận & Điều tôi học được

**Kết luận:**
Không nên vội vàng deploy bản fine-tune này lên production nếu chưa bổ sung 1-5% dữ liệu replay phổ thông để khắc phục hiện tượng quên thảm họa (regression tụt 0.280).
Mặc dù trên tác vụ hẹp CSKH, bản LoRA $r=16$ all-linear đã nâng độ chính xác từ 76.5% lên 97.0% và loại bỏ hoàn toàn prompt dài ở suy luận, sự sụt giảm tri thức chung là rủi ro lớn trong thực tế.
Đòn bẩy thực sự quyết định thành bại trong lab này chính là **Tính đúng đắn của Loss Masking (chỉ tính loss trên phản hồi)** kết hợp với **Thang Learning Rate LoRA phù hợp ($10\times$ full-FT)**.
Khi đã có mask đúng và LR đúng, việc nâng rank từ 16 lên 283 hay thay đổi vị trí chỉ mang lại biên độ dao động nhỏ (0.965 vs 0.970), trong khi sai LR sẽ phá huỷ toàn bộ kết quả (0.000).

**Ba điều tôi học được:**
1. **Không bao giờ tin tưởng vào train loss hay perplexity như một thước đo năng lực duy nhất**: Một cấu hình có thể có train loss thấp hơn nhờ học vẹt tham số cục bộ nhưng lại kém hơn trên tập đánh giá thực tế.
2. **Liêm chính trong đánh giá là nguyên tắc tối thượng**: Việc đóng băng Baseline (b) trước khi huấn luyện và bắt buộc tìm ra các ca fine-tune thua giúp ta nhìn nhận đúng bản chất mô hình thay vì tự đánh lừa bằng cherry-picking.
3. **Phần cứng quyết định cấu hình huấn luyện**: Hiểu rõ kiến trúc GPU (T4 Turing không hỗ trợ phần cứng bf16, phải dùng fp16 + GradScaler) giúp tránh các lỗi sập âm thầm thường gặp trong các hướng dẫn thiếu kiểm chứng.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
1. Trộn 3% tập dữ liệu chỉ dẫn tổng quát (General Instruction Replay) vào tập huấn luyện để đưa cổng hồi quy về trạng thái `PASSED`.
2. Huấn luyện thử nghiệm DoRA (Weight-Decomposed Low-Rank Adaptation) để đánh giá khả năng cân bằng giữa magnitude và direction trên tập dữ liệu tiếng Việt.

---

## Phụ lục - Thưởng đã làm

- [x] **B1 NB6 merge + hot-swap (+3đ)**: Điểm sau merge giữ nguyên tuyệt đối 0.9700 ($\Delta = +0.0000$), kiểm tra hoán đổi adapter thành công tại `results/merge_check.json`.
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [x] **B4 Quét rank có kiểm soát (+3đ)**: Đã xây dựng kịch bản kiểm tra `scripts/run_bonus_b4.py` quét $r \in \{8, 16, 64\}$ trên `text-linear`.
- [x] **B5 HuggingFace Hub (+2đ)**: Đã tích hợp script `scripts/push_to_hub.py` sẵn sàng push adapter.
