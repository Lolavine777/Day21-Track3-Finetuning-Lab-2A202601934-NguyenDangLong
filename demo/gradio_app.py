#!/usr/bin/env python3
"""Interactive Web Demo for Lab 21 — Vietnamese CSKH Ticket Triage.

Supports:
- Single ticket triage with LoRA fine-tuned model
- Live 3-way comparison (Baseline A vs Baseline B vs LoRA Fine-tune)
- Real-time Multi-tenant Adapter Hot-swapping (Deck §18)
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from labkit import device, evaluate as ev, generate
from labkit.config import NAIVE_PROMPT, OPTIMIZED_PROMPT, get_tier

# Global state for loaded base model and adapters
_BASE_MODEL = None
_TOKENIZER = None
_LOADED_ADAPTERS = {}
_ACTIVE_ADAPTER = None


def get_model_and_tokenizer():
    global _BASE_MODEL, _TOKENIZER
    if _BASE_MODEL is None:
        tier = get_tier(os.environ.get("COMPUTE_TIER", "T4"))
        print(f"Loading base model {tier.model_id}...")
        _BASE_MODEL, _TOKENIZER = generate.load_base(tier)
        # Load correct adapter if exists
        correct_dir = ROOT / "adapters" / "correct"
        if (correct_dir / "adapter_model.safetensors").exists():
            _BASE_MODEL = PeftModel.from_pretrained(_BASE_MODEL, str(correct_dir), adapter_name="correct")
            _LOADED_ADAPTERS["correct"] = True
            print("Loaded LoRA adapter 'correct'.")
    return _BASE_MODEL, _TOKENIZER


def run_inference(ticket_text: str, mode: str):
    """Run single inference mode."""
    if not ticket_text.strip():
        return "Vui lòng nhập nội dung ticket.", "", 0.0, {}

    model, tok = get_model_and_tokenizer()
    t0 = time.perf_counter()

    if mode == "LoRA Fine-tuned (Naive Prompt)":
        if "correct" in _LOADED_ADAPTERS:
            model.set_adapter("correct")
            model.enable_adapters()
        preds, lat = generate.generate_batch(model, tok, [ticket_text], system=NAIVE_PROMPT, progress=False)
    elif mode == "Baseline B (Base + Optimized Prompt)":
        if hasattr(model, "disable_adapters"):
            with model.disable_adapters():
                preds, lat = generate.generate_batch(model, tok, [ticket_text], system=OPTIMIZED_PROMPT, progress=False)
        else:
            preds, lat = generate.generate_batch(model, tok, [ticket_text], system=OPTIMIZED_PROMPT, progress=False)
    elif mode == "Baseline A (Base + Naive Prompt)":
        if hasattr(model, "disable_adapters"):
            with model.disable_adapters():
                preds, lat = generate.generate_batch(model, tok, [ticket_text], system=NAIVE_PROMPT, progress=False)
        else:
            preds, lat = generate.generate_batch(model, tok, [ticket_text], system=NAIVE_PROMPT, progress=False)
    else:
        preds, lat = ["Unknown mode"], 0.0

    raw_output = preds[0]
    parsed = ev._parse_json_loose(raw_output) or {}
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    formatted_json = json.dumps(parsed, ensure_ascii=False, indent=2) if parsed else "Không trích xuất được JSON hợp lệ"
    
    intent_val = parsed.get("intent", "N/A")
    urgency_val = parsed.get("urgency", "N/A")
    product_val = parsed.get("product", "N/A")
    sentiment_val = parsed.get("sentiment", "N/A")

    summary_html = f"""
    <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px;">
        <span style="background: #2563eb; color: white; padding: 4px 12px; border-radius: 16px; font-weight: bold;">Intent: {intent_val}</span>
        <span style="background: #dc2626; color: white; padding: 4px 12px; border-radius: 16px; font-weight: bold;">Urgency: {urgency_val}</span>
        <span style="background: #059669; color: white; padding: 4px 12px; border-radius: 16px; font-weight: bold;">Product: {product_val}</span>
        <span style="background: #d97706; color: white; padding: 4px 12px; border-radius: 16px; font-weight: bold;">Sentiment: {sentiment_val}</span>
    </div>
    """
    return summary_html, formatted_json, raw_output, f"{elapsed_ms:.1f} ms"


def run_side_by_side(ticket_text: str):
    """Run all 3 baselines side-by-side for comparison."""
    if not ticket_text.strip():
        return "", "", "", "", "", "", ""

    h_a, j_a, r_a, lat_a = run_inference(ticket_text, "Baseline A (Base + Naive Prompt)")
    h_b, j_b, r_b, lat_b = run_inference(ticket_text, "Baseline B (Base + Optimized Prompt)")
    h_ft, j_ft, r_ft, lat_ft = run_inference(ticket_text, "LoRA Fine-tuned (Naive Prompt)")

    return j_a, lat_a, j_b, lat_b, j_ft, lat_ft, h_ft


# Sample tickets
SAMPLE_TICKETS = [
    ["Shop ơi, mình đặt bàn phím cơ mã đơn DH123456. Giao hàng chậm quá, đã 5 ngày chưa nhận được. Nhờ shop kiểm tra giùm."],
    ["Sản phẩm tai nghe bluetooth bị rè một bên tai trái, sạc không vào điện. Tôi muốn đổi cái mới ngay lập tức."],
    ["Tôi muốn hủy đơn hàng áo thun oversize và hoàn tiền vào ví ShopeePay cho tôi, do đặt nhầm size."],
    ["Cho mình hỏi chuột không dây Logitech bên shop có được bảo hành chính hãng 12 tháng không ạ?"],
    ["Shop phục vụ quá tệ, gửi nhầm màu son môi và đóng gói rách nát. Yêu cầu đổi trả gấp!"],
]


def build_ui():
    with gr.Blocks(title="Lab 21 — CSKH Ticket Triage Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🚀 Lab 21 — Customer Support Ticket Triage LLM Demo
            ### Fine-tuning LLMs with LoRA & Multi-Group Regression Verification
            **Base Model**: Qwen3.5-4B · **Task**: 4-field Vietnamese Ticket Triage (`intent`, `urgency`, `product`, `sentiment`)
            """
        )

        with gr.Tab("🎯 Single Model Triage"):
            with gr.Row():
                with gr.Column(scale=2):
                    input_text = gr.Textbox(
                        label="Nội dung Ticket CSKH (Vietnamese Ticket)",
                        placeholder="Nhập câu hỏi, khiếu nại hoặc yêu cầu của khách hàng...",
                        lines=4,
                    )
                    mode_radio = gr.Radio(
                        choices=[
                            "LoRA Fine-tuned (Naive Prompt)",
                            "Baseline B (Base + Optimized Prompt)",
                            "Baseline A (Base + Naive Prompt)",
                        ],
                        value="LoRA Fine-tuned (Naive Prompt)",
                        label="Chọn Model / Baseline",
                    )
                    submit_btn = gr.Button("🚀 Phân loại Ticket", variant="primary")
                    gr.Examples(examples=SAMPLE_TICKETS, inputs=[input_text])

                with gr.Column(scale=2):
                    badge_output = gr.HTML(label="Trường trích xuất")
                    json_output = gr.Code(label="JSON Result", language="json")
                    latency_output = gr.Textbox(label="Độ trễ (Latency)", interactive=False)
                    raw_output = gr.Textbox(label="Raw Model Output", lines=3, interactive=False)

            submit_btn.click(
                fn=run_inference,
                inputs=[input_text, mode_radio],
                outputs=[badge_output, json_output, raw_output, latency_output],
            )

        with gr.Tab("⚖️ 3-Way Live Benchmark (Baseline A vs B vs LoRA)"):
            gr.Markdown("### So sánh trực tiếp 3 mốc đánh giá trên cùng một ticket:")
            bench_input = gr.Textbox(
                label="Ticket CSKH cần so sánh",
                value=SAMPLE_TICKETS[0][0],
                lines=3,
            )
            bench_btn = gr.Button("⚡ So sánh 3 mốc (Run 3-Way Benchmark)", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🥉 Baseline (a) — Base + Naive Prompt")
                    out_a = gr.Code(label="JSON (a)", language="json")
                    lat_a = gr.Textbox(label="Latency (a)")
                with gr.Column():
                    gr.Markdown("#### 🥈 Baseline (b) — Base + Optimized Prompt")
                    out_b = gr.Code(label="JSON (b)", language="json")
                    lat_b = gr.Textbox(label="Latency (b)")
                with gr.Column():
                    gr.Markdown("#### 🥇 LoRA Fine-tune — (c)")
                    out_ft = gr.Code(label="JSON (c)", language="json")
                    lat_ft = gr.Textbox(label="Latency (c)")

            badge_bench = gr.HTML()
            bench_btn.click(
                fn=run_side_by_side,
                inputs=[bench_input],
                outputs=[out_a, lat_a, out_b, lat_b, out_ft, lat_ft, badge_bench],
            )

        with gr.Tab("🔄 Multi-Tenant Adapter Hot-Swap (Deck §18)"):
            gr.Markdown(
                """
                ### Hoán đổi LoRA Adapter không cần reload base model (Deck §18)
                Mô hình nền (Base Model) chỉ nạp **1 lần** vào VRAM. Mỗi tác vụ/khách hàng là một adapter vài MB.
                """
            )
            swap_ticket = gr.Textbox(
                label="Ticket",
                value="Tôi muốn đổi trả tai nghe bluetooth do bị hỏng mic.",
                lines=2,
            )
            adapter_choice = gr.Radio(
                choices=["correct (all-linear, r=16)", "attn_only (q,v matched rank)", "wrong_lr (LR 1x)"],
                value="correct (all-linear, r=16)",
                label="Chọn Active Adapter",
            )
            swap_btn = gr.Button("Hot-swap & Generate")
            swap_output = gr.Code(language="json", label="Output từ Adapter đã chọn")
            
            def run_swap(ticket, ad_choice):
                key = "correct" if "correct" in ad_choice else ("attn_only" if "attn" in ad_choice else "wrong_lr")
                model, tok = get_model_and_tokenizer()
                ad_dir = ROOT / "adapters" / key
                if ad_dir.exists():
                    if key not in _LOADED_ADAPTERS:
                        model.load_adapter(str(ad_dir), adapter_name=key)
                        _LOADED_ADAPTERS[key] = True
                    model.set_adapter(key)
                preds, _ = generate.generate_batch(model, tok, [ticket], system=NAIVE_PROMPT, progress=False)
                parsed = ev._parse_json_loose(preds[0]) or {}
                return json.dumps(parsed, ensure_ascii=False, indent=2)

            swap_btn.click(fn=run_swap, inputs=[swap_ticket, adapter_choice], outputs=[swap_output])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=True)
