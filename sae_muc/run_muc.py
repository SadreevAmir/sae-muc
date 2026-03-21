#!/usr/bin/env python3
"""
MUC with SAE latent-bump hooks. Standalone: only needs sae_muc/, transformers, sae-lens, pandas, etc.

Data layout (same as Meta repo): under --repo_root
  datasets/{dataset}/{model_name}/{split}.csv
  detection/LR_outputs/{dataset}/{model_name}/{split}_verbal_uncertainty_sentence_semantic_entropy.json

Run from parent of this package directory, e.g.:
  python -m sae_muc.run_muc --repo_root . --model_name Mistral-7B-Instruct-v0.3 ...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import jsonlines
from sae_lens import SAE

_PKG_DIR = Path(__file__).resolve().parent
_REPO_PARENT = _PKG_DIR.parent
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

from sae_muc.generation import generate_all_responses
from sae_muc.hooks import clear_sae_latent_hooks, register_sae_latent_hooks
from sae_muc.layers_util import process_layers_to_process
from sae_muc.prompts_mini import UNCERTAINTY_SYSTEM, make_sentence_user_content


def resolve_repo_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    cand = _REPO_PARENT
    if (cand / "datasets").is_dir():
        return cand
    return Path.cwd().resolve()


def load_detection_res(root: Path, dataset: str, model_name: str, split: str) -> list:
    path = (
        root
        / "detection"
        / "LR_outputs"
        / dataset
        / model_name
        / f"{split}_verbal_uncertainty_sentence_semantic_entropy.json"
    )
    with open(path) as f:
        return __import__("json").load(f)["y_pred"]


def load_intervention(path: Path) -> tuple[str, dict[int, dict]]:
    data = torch.load(path, map_location="cpu")
    release = data["release"]
    layers: dict[int, dict] = {}
    for k, v in data["layers"].items():
        layers[int(k)] = v
    return release, layers


def load_saes_for_layers(release: str, layers: dict[int, dict], sae_dtype: str) -> dict[int, SAE]:
    out: dict[int, SAE] = {}
    for hf_layer, meta in layers.items():
        sae_id = meta["sae_id"]
        print(f"Loading SAE {release} / {sae_id} …")
        sae = SAE.from_pretrained(release, sae_id, device="cpu", dtype=sae_dtype)
        out[hf_layer] = sae
    return out


def get_answers_sae(
    questions: list,
    alphas: np.ndarray,
    detection_res: list,
    out_file: str,
    process_layers: list[int],
    model,
    tokenizer,
    prompt_type: str,
    layer_to_sae: dict[int, SAE],
    layer_to_delta: dict[int, torch.Tensor],
) -> None:
    print("will save to", out_file)

    if os.path.exists(out_file):
        with jsonlines.open(out_file, "r") as reader:
            history_len = len(list(reader))
    else:
        history_len = 0
    print("history_len", history_len)
    assert len(questions) == len(alphas) == len(detection_res)

    for i, (question, alpha, dr) in tqdm(
        enumerate(zip(questions, alphas, detection_res)),
        total=len(questions),
    ):
        if i < history_len:
            continue

        if alpha == 0 or dr == 0:
            line = {
                "alpha": 0,
                "question": question,
                "most_likely_answer": "",
                "responses": [],
            }
            with jsonlines.open(out_file, "a") as writer:
                writer.write(line)
        else:
            clear_sae_latent_hooks(model)
            register_sae_latent_hooks(
                model,
                layer_to_sae,
                layer_to_delta,
                process_layers,
                float(alpha),
            )
            if prompt_type == "uncertainty":
                messages = [
                    {"role": "system", "content": UNCERTAINTY_SYSTEM},
                    {"role": "user", "content": f"Question: {question}\nAnswer: "},
                ]
            elif prompt_type == "sentence":
                messages = [
                    {"role": "user", "content": make_sentence_user_content(question)},
                ]
            else:
                raise ValueError(prompt_type)
            generate_all_responses(
                model, tokenizer, [question], [messages], alpha, out_file, batch_size=1
            )

        if i % 100 == 0:
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_root",
        type=str,
        default=None,
        help="Корень с подкаталогами datasets/ и detection/ (по умолчанию: родитель sae_muc, если есть datasets/).",
    )
    parser.add_argument("--dataset", type=str, default="nq_open")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--iti_method", type=int, default=2)
    parser.add_argument("--str_process_layers", type=str, default="range(15,32)")
    parser.add_argument("--max_alpha", type=float, default=1.0)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, required=True)
    parser.add_argument(
        "--intervention_path",
        type=str,
        default="sae_muc/artifacts/mistral_intervention.pt",
        help="Относительно repo_root, если не абсолютный путь.",
    )
    parser.add_argument("--sae_dtype", type=str, default="float32")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Каталог для jsonl; по умолчанию sae_muc/outputs/... под repo_root.",
    )
    args = parser.parse_args()

    root = resolve_repo_root(args.repo_root)
    dataset = args.dataset
    split = args.split
    model_name = args.model_name
    prompt_type = args.prompt_type

    if "Llama" in model_name:
        full_model_name = f"meta-llama/{model_name}"
    elif "Qwen" in model_name:
        full_model_name = f"Qwen/{model_name}"
    elif "Mistral" in model_name:
        full_model_name = f"mistralai/{model_name}"
    else:
        full_model_name = model_name

    process_layers = process_layers_to_process(args.str_process_layers)
    results_df = pd.read_csv(root / "datasets" / dataset / model_name / f"{split}.csv")
    vu_scores_llm = results_df["verbal_uncertainty"].to_numpy()
    su_scores = results_df["sentence_semantic_entropy"].to_numpy()
    questions = results_df["question"].tolist()

    MAX_SE = 2.302585092994045
    MAX_ALPHA = args.max_alpha
    detection_res = load_detection_res(root, dataset, model_name, split)

    if args.iti_method != 2:
        raise NotImplementedError("Only iti_method=2.")

    alphas = (su_scores / MAX_SE - vu_scores_llm) * MAX_ALPHA
    alphas = np.clip(alphas, 0, MAX_ALPHA)
    alphas = np.round(alphas, 4)

    jsonl_name = (
        f"with_vufi_{args.iti_method}_{args.str_process_layers}_{args.max_alpha}.jsonl"
    )
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = (
            root
            / "sae_muc"
            / "outputs"
            / dataset
            / model_name
            / prompt_type
            / split
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = str(out_dir / jsonl_name)

    inter_path = Path(args.intervention_path)
    if not inter_path.is_file():
        inter_path = root / args.intervention_path
    release, layer_meta = load_intervention(inter_path)
    layer_to_delta = {l: layer_meta[l]["delta"] for l in layer_meta}
    layer_to_sae = load_saes_for_layers(release, layer_meta, args.sae_dtype)

    print("repo_root", root)
    print("process_layers", process_layers)
    print("SAE hooks on layers", [l for l in process_layers if l in layer_to_sae])
    skipped = [l for l in process_layers if l not in layer_to_sae]
    if skipped:
        print("No SAE config for HF layers (skipped):", skipped)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        full_model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(full_model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    get_answers_sae(
        questions,
        alphas,
        detection_res,
        out_file,
        process_layers,
        model,
        tokenizer,
        prompt_type,
        layer_to_sae,
        layer_to_delta,
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
