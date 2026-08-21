#!/usr/bin/env python3
"""Bonus B4: Controlled rank sweep across r in {8, 16, 64}.

Fixed placement = text-linear, fixed LR = 1e-4, fixed step budget.
Measures target accuracy and parameter scaling.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import Dataset
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer

from labkit import data, evaluate as ev, generate, modeling, report, train
from labkit.config import LoraSpec, LORA_LR, get_tier, training_epochs


def main() -> int:
    tier = get_tier(os.environ.get("COMPUTE_TIER", "T4"))
    split_dir = ROOT / "data" / "split"
    if not split_dir.exists():
        print("Run NB1 first to generate data/split")
        return 1

    train_rows = [json.loads(l) for l in (split_dir / "train.jsonl").open(encoding="utf-8") if l.strip()]
    target_rows = [json.loads(l) for l in (ROOT / "data" / "eval_target.jsonl").open(encoding="utf-8") if l.strip()]

    eval_limit = int(os.environ.get("EVAL_LIMIT", "0"))
    if eval_limit:
        target_rows = target_rows[:eval_limit]

    print(f"=== Bonus B4: Controlled Rank Sweep on {tier.name} ({tier.model_id}) ===")
    
    ranks = [8, 16, 64]
    sweep_results = []

    for r in ranks:
        key = f"rank_{r}"
        spec = LoraSpec(
            key=key,
            r=r,
            alpha=2 * r,
            target="text-linear",
            lr=LORA_LR,
            load_in_4bit=False,
            label=f"all-linear · r={r} · LR 10x",
            teaches=f"Controlled rank test at r={r}",
        )
        print(f"\n--- Training {key} (r={r}, alpha={2*r}) ---")
        
        model, tok = generate.load_base(tier)
        train_ds = Dataset.from_list(
            data.to_training_dataset(tok, train_rows, max_length=tier.max_length,
                                     mask_mode=os.environ.get("MASK_MODE", "assistant-only"))
        )
        targets = modeling.resolve_target_modules(model, "text-linear")
        trainable = modeling.count_lora_params(model, targets, r)
        max_steps = train.planned_steps(len(train_ds), tier, training_epochs())

        out_dir = ROOT / "adapters" / key
        want_sft = train.sft_config_kwargs(tier, spec, str(out_dir), max_steps=max_steps)
        sft_kwargs, _ = train.filter_kwargs(SFTConfig, want_sft, label=f"SFTConfig[{key}]")
        lora_kwargs, _ = train.filter_kwargs(
            LoraConfig, train.lora_config_kwargs(spec, targets), label=f"LoraConfig[{key}]"
        )

        trainer = SFTTrainer(
            model=model,
            args=SFTConfig(**sft_kwargs),
            train_dataset=train_ds,
            processing_class=tok,
            peft_config=LoraConfig(**lora_kwargs),
        )
        train.align_trainable_precision(trainer.model)

        t0 = time.perf_counter()
        res = trainer.train()
        train_sec = time.perf_counter() - t0
        trainer.model.save_pretrained(out_dir)

        # Eval on target
        preds, lat = generate.generate_batch(
            trainer.model, tok, [row["input"] for row in target_rows],
            system=generate.NAIVE_PROMPT, label=f"{key}/target"
        )
        tgt_acc = sum(ev.triage_field_accuracy(p, row["label"]) for p, row in zip(preds, target_rows)) / len(target_rows)
        fmt_acc = sum(ev.has_required_keys(p, ev.TRIAGE_KEYS) for p in preds) / len(preds)

        del trainer, model
        generate.free_memory()

        entry = {
            "rank": r,
            "trainable_params": trainable,
            "train_loss": round(res.training_loss, 4),
            "target_acc": round(tgt_acc, 4),
            "format_acc": round(fmt_acc, 4),
            "latency_ms": round(lat, 1),
            "train_sec": round(train_sec, 1),
        }
        sweep_results.append(entry)
        print(f"Result for r={r}: target={tgt_acc:.4f}, loss={res.training_loss:.4f}, params={trainable:,}")

    report.write_json(sweep_results, "bonus_rank_sweep.json", results_dir=ROOT / "results")
    print("\n=== Rank Sweep Summary Table ===")
    print(report.markdown_table(sweep_results, ["rank", "trainable_params", "train_loss", "target_acc", "format_acc", "latency_ms", "train_sec"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
