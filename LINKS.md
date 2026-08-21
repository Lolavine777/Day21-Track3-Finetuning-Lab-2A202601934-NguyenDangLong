# Lab 21 - Submission Links & Audit Manifest

**Học viên**: Nguyễn Đăng Long  
**MSSV**: 2A202601934  
**Khóa học**: Generative AI / Large Language Models Fine-tuning Lab  
**Ngày nộp**: 2026-08-21  

---

## 🔗 Các đường link nộp bài

| Hạng mục | Đường link | Ghi chú |
|---|---|---|
| **GitHub Repository** | [GitHub Repo](https://github.com/Lolavine777/Day21-Track3-Finetuning-Lab-2A202601934-NguyenDangLong) | Mã nguồn đầy đủ, toàn bộ `results/`, và báo cáo khoa học |
| **Colab Notebook (1-Click Run)** | [Colab Notebook](https://colab.research.google.com/github/Lolavine777/Day21-Track3-Finetuning-Lab-2A202601934-NguyenDangLong/blob/main/colab/Lab21_RUN_ALL.ipynb) | Notebook chuẩn hoá 7 bước chạy toàn bộ pipeline trên T4 |
| **Báo cáo Khoa học** | [submission/REPORT.md](https://github.com/Lolavine777/Day21-Track3-Finetuning-Lab-2A202601934-NguyenDangLong/blob/main/submission/REPORT.md) | Đầy đủ 7 mục, 2,000+ từ, giải trình nhân quả chi tiết |
| **Bản Phản tư Cá nhân** | [submission/REFLECTION.md](https://github.com/Lolavine777/Day21-Track3-Finetuning-Lab-2A202601934-NguyenDangLong/blob/main/submission/REFLECTION.md) | 5 câu hỏi phản tư chuyên sâu |

---

## 📋 Bảng Kiểm tra Đối chiếu (Audit Checklist & Rubric Mapping)

### 1. Năng lực Kỹ thuật cốt lõi (30 điểm)
- [x] **1.1 Masking đúng**: Loss chỉ tính trên câu trả lời (`supervised_fraction = 41.5%`, `answer_is_supervised = true`, `question_is_masked = true` tại `results/mask_proof.json`).
- [x] **1.2 Cấu hình low-regret chuẩn**: $r=16$, all-linear (`text-linear` 12 modules), $LR=1\times 10^{-4}$ ($10\times$ full-FT), `fp16` + GradScaler trên Tesla T4.
- [x] **1.3 Tránh 3 lỗi ngớ ngẩn**:
  - Mistake #1 (Attention-only matched rank): Thử nghiệm $r=283$ cùng 32.46M tham số.
  - Mistake #2 (Wrong LR): Thử nghiệm $LR=1\times 10^{-5}$.
  - Mistake #3 (QLoRA 4-bit): Đo đạc đánh đổi bộ nhớ (-41% VRAM) vs chất lượng/độ trễ.

### 2. Kỷ luật Khoa học & Đối chứng (30 điểm)
- [x] **2.1 Ba baseline đo trước train**: (a) Base + naive (0.000), (b) Base + optimized (0.765), (c) LoRA fine-tune (0.970).
- [x] **2.2 Đóng băng Baseline (b)**: Giữ nguyên prompt tối ưu `SHA: 719e74d3b6232053` trong `results/baselines_frozen.json`.
- [x] **2.3 Đối chứng công bằng**: Cùng 30 optimizer steps (2 epochs), cùng seed 42, cùng tập split 225/25.
- [x] **2.4 Xếp hạng thật**: Xếp hạng 4 run bằng cột `target (NB5 §4)` thay vì `train_loss`.

### 3. Đánh giá Đa chiều & Trung thực (20 điểm)
- [x] **3.1 Bốn nhóm chỉ số**: Target Accuracy, General Regression, Format Compliance, Inference Latency.
- [x] **3.2 Đánh giá toàn bộ tập test**: 50 target items + 15 regression items (`eval_limit=None`).
- [x] **3.3 Diễn giải phán quyết**: Giải trình nhân quả hiện tượng `FAILED` do Catastrophic Forgetting và giải pháp Replay Buffer.
- [x] **3.4 Định tính khách quan**: Phân tích 5 ví dụ cụ thể, gồm 3 ca fine-tune THUA và 2 ca THẮNG.

### 4. Chất lượng Báo cáo (20 điểm)
- [x] **4.1 Đủ 7 mục**: Tuân thủ 100% mẫu `submission/REPORT.md`.
- [x] **4.2 Kết luận sâu sắc**: $\ge 150$ từ với lập luận nhân quả rõ ràng.
- [x] **4.3 Khớp số 100%**: Mọi số liệu trong report trùng khớp tuyệt đối với các file trong `results/`.
- [x] **4.4 Phản tư cá nhân**: Hoàn thành `submission/REFLECTION.md` cụ thể, phi generic.

### 5. Điểm Thưởng Bonus (+15 điểm)
- [x] **Bonus B1 (+3đ)**: NB6 merge + assert không tụt điểm (`before = 0.9700`, `after = 0.9700`, $\Delta = +0.0000$ tại `results/merge_check.json`).
- [x] **Bonus B4 (+3đ)**: Quét rank có kiểm soát $r \in \{8, 16, 64\}$ trên `text-linear` tại `results/bonus_rank_sweep.json`.
- [x] **Bonus B5 (+2đ)**: Sẵn sàng công cụ push Hub `scripts/push_to_hub.py`.

---

## 🛡️ Kết quả Gatekeeper Verification
```
[  ok  ] unit tests: 118 passed in 0.77s
[  ok  ] mask proof asserts: supervised_fraction = 41.5%
[  ok  ] full eval set used: 50 target items
[  ok  ] baseline (b) prompt unmodified: SHA 719e74d3b6232053
[  ok  ] baseline (b) beats (a): (a)=0.000 -> (b)=0.765
[  ok  ] all runs share ONE step budget: 30 steps
[  ok  ] attn_only is a FAIR contrast: 32,456,704 vs 32,464,896 trainable params
[  ok  ] verdict recorded: FAILED (target Δ +0.205, regression Δ -0.280)
[  ok  ] REPORT.md filled in: ~2017 words

26 passed · 1 warnings · 0 failures
Ready to submit.
```
