#!/usr/bin/env python3
"""Bonus B5: Push trained LoRA adapter to Hugging Face Hub.

Usage:
    python scripts/push_to_hub.py --repo-id <username>/lab21-qwen35-triage-vi [--token <hf_token>]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from peft import PeftModel
from transformers import AutoTokenizer
from labkit.config import get_tier


def main() -> int:
    parser = argparse.ArgumentParser(description="Push LoRA adapter to HuggingFace Hub")
    parser.add_argument("--repo-id", required=True, help="HF Hub repository ID (e.g. username/lab21-qwen35-triage-vi)")
    parser.add_argument("--adapter-dir", default="adapters/correct", help="Path to adapter dir (default: adapters/correct)")
    parser.add_argument("--token", default=None, help="HF Token (or read from HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Make repo private (default: public)")
    args = parser.parse_args()

    adapter_path = ROOT / args.adapter_dir
    if not (adapter_path / "adapter_model.safetensors").exists():
        print(f"Error: Adapter not found at {adapter_path}. Run NB3 first.")
        return 1

    token = args.token or os.environ.get("HF_TOKEN")
    tier = get_tier(os.environ.get("COMPUTE_TIER", "T4"))

    print(f"Loading base tokenizer from {tier.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(tier.model_id, trust_remote_code=True)

    print(f"Pushing adapter from {adapter_path} to HuggingFace Hub: {args.repo_id}...")
    
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)

    # Upload folder
    api.upload_folder(
        folder_path=str(adapter_path),
        repo_id=args.repo_id,
        repo_type="model",
    )

    # Push tokenizer as well
    tokenizer.push_to_hub(args.repo_id, token=token)

    print(f"\nSuccessfully pushed adapter to: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
