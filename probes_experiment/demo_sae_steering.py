#!/usr/bin/env python3
"""
Demo: SAE feature steering on 10 examples.

Loads Mistral-7B, constructs a delta vector from top uncertainty SAE features
found by sae_feature_analysis.py, and shows baseline vs steered answers.

Usage:
    python probes_experiment/demo_sae_steering.py
    python probes_experiment/demo_sae_steering.py --alpha 3.0 --n_examples 10
    python probes_experiment/demo_sae_steering.py --device mps   # Apple Silicon
"""

import argparse
import os
import sys
import json
import time

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from sae_lens import SAE
from sae_muc.hooks import register_sae_latent_hooks, clear_sae_latent_hooks


def log_step(msg: str):
    """Print a timestamped status line."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ── Top uncertainty features from sae_feature_analysis.py ──────────────────
# These are features that are MORE active for uncertain answers (direction=uncertain),
# confirmed by multiple methods (t-stat, mean_diff, frequency).
UNCERTAINTY_FEATURES = {
    # Layer 15: strongest uncertainty signals
    15: {
        "sae_id": "blocks.16.hook_resid_pre",
        "features": [
            (17678, 8.45),   # t-stat=8.45, also top hedge score
            (62652, 8.50),   # t-stat=8.50
            (63959, 7.58),   # t-stat=7.58, hedge=0.207
            (17229, 7.51),   # t-stat=7.51
            (63890, 7.32),   # t-stat=7.32, hedge=0.187
        ],
    },
    # Layer 23: very strong signals (t>10)
    23: {
        "sae_id": "blocks.24.hook_resid_pre",
        "features": [
            (54799, 12.43),  # t-stat=12.43, hedge=0.225
            (50003, 11.80),  # t-stat=11.80, in 3-method intersection
            (28051, 11.02),  # t-stat=11.02, in 3-method intersection
            (18525, 10.45),  # t-stat=10.45, in 3-method intersection
            (22939, 9.70),   # t-stat=9.70
            (34257, 9.33),   # in 3-method intersection
            (3260,  8.80),   # in 3-method intersection
        ],
    },
}


def build_delta_vectors(layer_to_sae: dict[int, SAE]) -> dict[int, torch.Tensor]:
    """Build sparse delta vectors: +1 at each uncertainty feature index."""
    deltas = {}
    for hf_layer, sae in layer_to_sae.items():
        if hf_layer not in UNCERTAINTY_FEATURES:
            continue
        d_sae = sae.cfg.d_sae
        delta = torch.zeros(d_sae)
        for feat_idx, t_stat in UNCERTAINTY_FEATURES[hf_layer]["features"]:
            # Weight by normalized t-stat so stronger features contribute more
            delta[feat_idx] = t_stat / 10.0  # scale to ~1.0
        deltas[hf_layer] = delta
    return deltas


def load_questions(n: int) -> list[dict]:
    """Load n questions from test set, mixing correct and incorrect."""
    csv_path = os.path.join(ROOT, "datasets__nq_open (6)", "Mistral-7B-Instruct-v0.3", "test.csv")
    df = pd.read_csv(csv_path)

    # Pick a mix: some with low VU (model is confident), some with high VU
    df_sorted = df.sort_values("verbal_uncertainty")
    # Take half from confident (low VU) and half from uncertain (high VU)
    n_half = n // 2
    confident_qs = df_sorted.head(n_half)
    uncertain_qs = df_sorted.tail(n - n_half)
    selected = pd.concat([confident_qs, uncertain_qs]).reset_index(drop=True)

    questions = []
    for _, row in selected.iterrows():
        questions.append({
            "question": row["question"],
            "answer": row["answer"],
            "vu_before": float(row["verbal_uncertainty"]),
            "se": float(row["sentence_semantic_entropy"]),
        })
    return questions


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 50,
                    label: str = "") -> str:
    """Generate with model.generate() (fast KV-cache) and print result."""
    messages = [
        {"role": "user", "content": f"Question: {question}\nAnswer: "},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    print(f"  {label} generating ...", end="", flush=True)
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    elapsed = time.time() - t0
    n_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    answer = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f" done ({n_tokens} tok, {elapsed:.1f}s, {n_tokens/elapsed:.1f} tok/s)", flush=True)
    return answer.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="Steering strength (higher = more uncertainty)")
    parser.add_argument("--n_examples", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, mps, cuda")
    parser.add_argument("--model_name", type=str,
                        default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    # ── Resolve device ──
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    log_step(f"Device: {device}, dtype: {args.dtype}")

    # ── Step 1/5: Load tokenizer ──
    log_step(f"[1/5] Loading tokenizer: {args.model_name} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log_step("[1/5] Tokenizer ready ✓")

    # ── Step 2/5: Load model (~14GB, takes a while first time) ──
    log_step(f"[2/5] Loading model weights (~14GB in {args.dtype}) ...")
    log_step("       First run downloads from HuggingFace; subsequent runs use cache.")
    t0 = time.time()
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=dtype, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=dtype
        ).to(device)
    model.eval()
    log_step(f"[2/5] Model loaded in {time.time()-t0:.1f}s ✓")

    # ── Step 3/5: Load SAEs ──
    log_step(f"[3/5] Loading SAEs for layers {list(UNCERTAINTY_FEATURES.keys())} ...")
    layer_to_sae: dict[int, SAE] = {}
    for hf_layer, info in tqdm(UNCERTAINTY_FEATURES.items(), desc="SAE layers", unit="sae"):
        sae_id = info["sae_id"]
        log_step(f"       SAE layer {hf_layer} ({sae_id}) ...")
        t0 = time.time()
        sae = SAE.from_pretrained("mistral-7b-res-wg", sae_id, device=device, dtype=args.dtype)
        layer_to_sae[hf_layer] = sae
        log_step(f"       SAE layer {hf_layer} loaded in {time.time()-t0:.1f}s ✓")
    log_step("[3/5] All SAEs ready ✓")

    # ── Step 4/5: Build delta vectors + load questions ──
    log_step("[4/5] Building delta vectors from uncertainty features ...")
    layer_to_delta = build_delta_vectors(layer_to_sae)
    for l, d in layer_to_delta.items():
        n_nonzero = (d != 0).sum().item()
        log_step(f"       Layer {l}: {n_nonzero} active features in delta vector")

    questions = load_questions(args.n_examples)
    log_step(f"[4/5] Selected {len(questions)} questions ({len(questions)//2} confident + {len(questions) - len(questions)//2} uncertain) ✓")

    # ── Step 5/5: Generate answers ──
    print()
    print("=" * 80)
    log_step(f"[5/5] GENERATING BASELINE vs STEERED (alpha={args.alpha})")
    print(f"       Layers: {list(UNCERTAINTY_FEATURES.keys())}")
    print("=" * 80)

    results = []
    pbar = tqdm(questions, desc="Generating", unit="q")
    for i, q in enumerate(pbar):
        pbar.set_postfix_str(q["question"][:40] + "...")

        print(f"\n{'─' * 80}")
        print(f"Q{i+1}/{len(questions)}: {q['question']}")
        print(f"     Golden: {q['answer']}")
        print(f"     VU={q['vu_before']:.3f}  SE={q['se']:.3f}")

        # Baseline (no hooks)
        clear_sae_latent_hooks(model)
        t0 = time.time()
        baseline = generate_answer(model, tokenizer, q["question"], label="BASELINE")
        t_base = time.time() - t0
        print(f"  BASELINE ({t_base:.1f}s):  {baseline}")

        # Steered (boost uncertainty features)
        hook_layers = list(layer_to_delta.keys())
        register_sae_latent_hooks(model, layer_to_sae, layer_to_delta, hook_layers, args.alpha)
        t0 = time.time()
        steered = generate_answer(model, tokenizer, q["question"], label="STEERED ")
        t_steer = time.time() - t0
        clear_sae_latent_hooks(model)
        print(f"  STEERED  ({t_steer:.1f}s):  {steered}")

        results.append({
            "question": q["question"],
            "golden": q["answer"],
            "vu_before": q["vu_before"],
            "se": q["se"],
            "baseline": baseline,
            "steered": steered,
            "alpha": args.alpha,
        })

    # ── Save results ──
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "demo_sae_steering.json")
    with open(out_path, "w") as f:
        json.dump({
            "alpha": args.alpha,
            "layers": list(UNCERTAINTY_FEATURES.keys()),
            "features_per_layer": {
                str(l): [idx for idx, _ in info["features"]]
                for l, info in UNCERTAINTY_FEATURES.items()
            },
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
