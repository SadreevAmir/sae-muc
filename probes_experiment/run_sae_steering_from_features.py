#!/usr/bin/env python3
"""
Run LLM generation with SAE steering using features from robust analysis output.

Output JSONL schema (one line per question):
  alpha, question, most_likely_answer, responses
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sae_lens import SAE
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sae_muc.generation import generate_lines_for_batch
from sae_muc.hooks import clear_sae_latent_hooks, register_sae_latent_hooks
from sae_muc.layer_map import hf_layers_for_release
from sae_muc.prompts_mini import make_sentence_user_content

MAX_SE = 2.302585092994045

# Edit only this block for quick button-run setup.
RUN_CONFIG = {
    "repo_root": str(ROOT),
    "model_name": "Mistral-7B-Instruct-v0.3",
    "full_model_name": "",
    "release": "mistral-7b-res-wg",
    "robust_stats_json": "sae_muc/artifacts/sae_feature_analysis_merged_stats.json",
    "test_csv": "",
    "n_rows": 4,
    "alpha_max": 50,
    "alpha_quantization_bins": 4,
    "gen_batch_size": 4,
    "top_features_per_layer": 32,
    "feature_value": 1.0,
    "sae_dtype": "float32",
    "device_map": "auto",
    "output_jsonl": "/Users/amir/Downloads/out.jsonl",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SAE steering from robust feature stats JSON")
    p.add_argument("--repo_root", type=str, default=RUN_CONFIG["repo_root"])
    p.add_argument("--model_name", type=str, default=RUN_CONFIG["model_name"])
    p.add_argument("--full_model_name", type=str, default=RUN_CONFIG["full_model_name"])
    p.add_argument("--release", type=str, default=RUN_CONFIG["release"])
    p.add_argument(
        "--robust_stats_json",
        type=str,
        default=RUN_CONFIG["robust_stats_json"],
    )
    p.add_argument("--test_csv", type=str, default=RUN_CONFIG["test_csv"])
    p.add_argument("--n_rows", type=int, default=RUN_CONFIG["n_rows"])
    p.add_argument("--alpha_max", type=float, default=RUN_CONFIG["alpha_max"])
    p.add_argument(
        "--alpha_quantization_bins",
        type=int,
        default=RUN_CONFIG["alpha_quantization_bins"],
        help="Quantization bins. alpha_step = alpha_max / bins.",
    )
    p.add_argument("--gen_batch_size", type=int, default=RUN_CONFIG["gen_batch_size"])
    p.add_argument(
        "--top_features_per_layer",
        type=int,
        default=RUN_CONFIG["top_features_per_layer"],
        help="How many robust uncertain-up features to keep per layer.",
    )
    p.add_argument(
        "--feature_value",
        type=float,
        default=RUN_CONFIG["feature_value"],
        help="Single value assigned to each selected feature in delta.",
    )
    p.add_argument("--sae_dtype", type=str, default=RUN_CONFIG["sae_dtype"], choices=("float32", "float16", "bfloat16"))
    p.add_argument("--device_map", type=str, default=RUN_CONFIG["device_map"])
    p.add_argument("--output_jsonl", type=str, default=RUN_CONFIG["output_jsonl"])
    return p.parse_args()


def resolve_full_model_name(model_name: str, full_model_name: str) -> str:
    if full_model_name:
        return full_model_name
    if "Llama" in model_name:
        return f"meta-llama/{model_name}"
    if "Qwen" in model_name:
        return f"Qwen/{model_name}"
    if "Mistral" in model_name:
        return f"mistralai/{model_name}"
    return model_name


def resolve_stats_path(root: Path, stats_arg: str) -> Path:
    p = Path(stats_arg)
    if p.is_file():
        return p
    p2 = root / stats_arg
    if p2.is_file():
        return p2
    raise FileNotFoundError(f"Robust stats JSON not found: {stats_arg}")


def resolve_test_csv(root: Path, model_name: str, explicit: str) -> Path:
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        p2 = root / explicit
        if p2.is_file():
            return p2
        raise FileNotFoundError(f"test.csv not found: {explicit}")

    candidates = [
        root / "datasets__nq_open" / model_name / "test.csv",
        root / "datasets__nq_open" / f"{model_name}_sentence" / "test.csv",
        root / "datasets__nq_open" / "sampled" / "test.csv",
        root / "datasets" / "nq_open" / model_name / "test.csv",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError("Could not locate test.csv in standard locations.")


def load_robust_features(stats_path: Path, release: str, top_k: int) -> tuple[dict[int, str], dict[int, torch.Tensor]]:
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    expected_layers = {l for l, _ in hf_layers_for_release(release)}
    layer_to_sae_id: dict[int, str] = {}
    layer_to_delta: dict[int, torch.Tensor] = {}

    for item in stats.get("layers", []):
        if "layer" not in item or "sae_id" not in item:
            continue
        layer = int(item["layer"])
        if layer not in expected_layers:
            continue
        feature_idx = item.get("top_uncertainty_feature_idx", [])
        if not feature_idx:
            continue
        layer_to_sae_id[layer] = str(item["sae_id"])
        layer_to_delta[layer] = torch.tensor(feature_idx[:top_k], dtype=torch.long)

    if not layer_to_delta:
        raise RuntimeError("No layers/features found in robust stats JSON for selected release.")
    return layer_to_sae_id, layer_to_delta


def load_saes_and_build_delta(
    release: str,
    layer_to_sae_id: dict[int, str],
    layer_to_feature_idx: dict[int, torch.Tensor],
    sae_dtype: str,
    feature_value: float,
) -> tuple[dict[int, SAE], dict[int, torch.Tensor]]:
    layer_to_sae: dict[int, SAE] = {}
    layer_to_delta: dict[int, torch.Tensor] = {}

    for layer, sae_id in layer_to_sae_id.items():
        sae = SAE.from_pretrained(release=release, sae_id=sae_id, device="cpu", dtype=sae_dtype)
        d = torch.zeros(sae.cfg.d_sae, dtype=torch.float32)
        idx = layer_to_feature_idx[layer]
        d[idx] = float(feature_value)
        d = d / (d.norm() + 1e-8)
        layer_to_sae[layer] = sae
        layer_to_delta[layer] = d
    return layer_to_sae, layer_to_delta


def build_messages(questions: list[str]) -> list[list[dict]]:
    return [[{"role": "user", "content": make_sentence_user_content(q)}] for q in questions]


def compute_alphas(vu: np.ndarray, se: np.ndarray, alpha_max: float, bins: int) -> tuple[np.ndarray, float]:
    raw = (se / MAX_SE - vu) * alpha_max
    raw = np.clip(raw, 0.0, alpha_max)
    if bins <= 0:
        raise ValueError("--alpha_quantization_bins must be >= 1")
    alpha_step = alpha_max / float(bins)
    if alpha_step <= 0:
        return np.zeros_like(raw), 0.0
    q = np.clip(np.round(raw / alpha_step) * alpha_step, 0.0, alpha_max)
    q = np.round(q, 6)
    return q, alpha_step


def run_generation(
    model,
    tokenizer,
    questions: list[str],
    messages: list[list[dict]],
    alphas: np.ndarray,
    hook_layers: list[int],
    layer_to_sae: dict[int, SAE],
    layer_to_delta: dict[int, torch.Tensor],
    output_jsonl: Path,
    batch_size: int,
) -> None:
    n = len(questions)
    by_alpha: dict[float, list[int]] = {}
    for i, a in enumerate(alphas.tolist()):
        by_alpha.setdefault(float(a), []).append(i)

    results: list[dict | None] = [None] * n

    for alpha in sorted(by_alpha.keys()):
        clear_sae_latent_hooks(model)
        if alpha > 0:
            register_sae_latent_hooks(model, layer_to_sae, layer_to_delta, hook_layers, alpha)
        idxs = by_alpha[alpha]

        for j in tqdm(range(0, len(idxs), batch_size), desc=f"alpha={alpha:.4f}"):
            chunk_idx = idxs[j : j + batch_size]
            batch_q = [questions[k] for k in chunk_idx]
            batch_m = [messages[k] for k in chunk_idx]
            lines = generate_lines_for_batch(model, tokenizer, batch_q, batch_m, alpha)
            for local_i, global_i in enumerate(chunk_idx):
                results[global_i] = lines[local_i]

    clear_sae_latent_hooks(model)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in results:
            if item is None:
                raise RuntimeError("Internal error: missing generated row.")
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    root = Path(args.repo_root).resolve()
    output_jsonl = Path(args.output_jsonl)
    if int(args.n_rows) == 0:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(output_jsonl, "w", encoding="utf-8"):
            pass
        print(f"alpha_max={args.alpha_max}")
        print(f"alpha_quantization_bins={args.alpha_quantization_bins}")
        print("rows=0")
        print(f"output={output_jsonl}")
        print("No rows selected (n_rows=0). Wrote empty JSONL and exited.")
        return

    stats_path = resolve_stats_path(root, args.robust_stats_json)
    test_csv = resolve_test_csv(root, args.model_name, args.test_csv)
    full_model_name = resolve_full_model_name(args.model_name, args.full_model_name)

    df = pd.read_csv(test_csv)
    if "question" not in df.columns or "verbal_uncertainty" not in df.columns or "sentence_semantic_entropy" not in df.columns:
        raise ValueError("test.csv must contain columns: question, verbal_uncertainty, sentence_semantic_entropy")
    n = min(max(0, int(args.n_rows)), len(df))
    df = df.iloc[:n].copy()

    questions = df["question"].astype(str).tolist()
    vu = df["verbal_uncertainty"].to_numpy(dtype=np.float64)
    se = df["sentence_semantic_entropy"].to_numpy(dtype=np.float64)
    alphas, alpha_step = compute_alphas(vu, se, float(args.alpha_max), int(args.alpha_quantization_bins))
    unique_groups = len(np.unique(alphas))

    layer_to_sae_id, layer_to_feature_idx = load_robust_features(
        stats_path=stats_path,
        release=args.release,
        top_k=int(args.top_features_per_layer),
    )
    layer_to_sae, layer_to_delta = load_saes_and_build_delta(
        release=args.release,
        layer_to_sae_id=layer_to_sae_id,
        layer_to_feature_idx=layer_to_feature_idx,
        sae_dtype=args.sae_dtype,
        feature_value=args.feature_value,
    )
    hook_layers = sorted(layer_to_sae.keys())
    messages = build_messages(questions)

    print(f"alpha_max={args.alpha_max}")
    print(f"alpha_quantization_bins={args.alpha_quantization_bins}")
    print(f"alpha_step={alpha_step}")
    print(f"unique_quantized_alpha_groups={unique_groups}")
    print(f"hook_layers={hook_layers}")
    print(f"rows={len(questions)}")
    print(f"output={output_jsonl}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        full_model_name,
        torch_dtype=torch.float16,
        device_map=args.device_map,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(full_model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    run_generation(
        model=model,
        tokenizer=tokenizer,
        questions=questions,
        messages=messages,
        alphas=alphas,
        hook_layers=hook_layers,
        layer_to_sae=layer_to_sae,
        layer_to_delta=layer_to_delta,
        output_jsonl=output_jsonl,
        batch_size=max(1, int(args.gen_batch_size)),
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
