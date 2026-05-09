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

from sae_muc.generation import generate_lines_for_batch
from sae_muc.hooks import (
    clear_sae_latent_hooks,
    register_sae_latent_hooks,
    register_sae_clamp_hooks,
)
from sae_muc.layer_map import hf_layers_for_release
from sae_muc.layers_util import parse_layers_str, process_layers_to_process
from sae_muc.prompts_mini import (
    UNCERTAINTY_SYSTEM,
    make_plain_user_content,
    make_sentence_user_content,
)
from sae_muc.vuf_hooks import clear_vuf_residual_hooks, register_vuf_residual_hooks

# Steering methods that use SAE (need SAE loaded)
_SAE_STEERING_METHODS = {"sae", "sae_emd", "sae_projected_vuf", "sae_clamp"}


def _chat_messages_for_question(question: str, prompt_type: str) -> list[dict]:
    if prompt_type == "uncertainty":
        return [
            {"role": "system", "content": UNCERTAINTY_SYSTEM},
            {"role": "user", "content": f"Question: {question}\nAnswer: "},
        ]
    if prompt_type == "plain":
        return [{"role": "user", "content": make_plain_user_content(question)}]
    if prompt_type == "sentence":
        return [{"role": "user", "content": make_sentence_user_content(question)}]
    raise ValueError(prompt_type)


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
    data = torch.load(path, map_location="cpu", weights_only=False)
    release = data["release"]
    layers: dict[int, dict] = {}
    for k, v in data["layers"].items():
        layers[int(k)] = v
    return release, layers


def load_intervention_v2(path: Path, method: str) -> tuple[str, dict[int, dict]]:
    """
    Load v2 intervention config and extract the relevant method's data.

    Returns (release, layer_meta) where layer_meta[hf_layer] contains:
      - "sae_id": str
      - For emd/projected_vuf: "delta": Tensor[d_sae]
      - For clamp: "clamp_config": {unc_indices, unc_targets, cert_indices}
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    release = data["release"]
    raw_layers = data["layers"]

    layers: dict[int, dict] = {}
    for k, v in raw_layers.items():
        hf_layer = int(k)
        entry: dict = {"sae_id": v["sae_id"]}

        if method == "sae_emd":
            entry["delta"] = v["method_emd"]["delta"]
        elif method == "sae_projected_vuf":
            pvuf = v.get("method_projected_vuf")
            if pvuf is None:
                print(f"  Warning: layer {hf_layer} has no projected_vuf data (no hedge?), skipping")
                continue
            entry["delta"] = pvuf["delta"]
        elif method == "sae_clamp":
            clamp = v["method_clamp"]
            unc_idx = clamp["uncertainty_features"]
            cert_idx = clamp["certainty_features"]
            target_vals = clamp["target_uncertain_values"]
            entry["clamp_config"] = {
                "unc_indices": torch.tensor(unc_idx, dtype=torch.long),
                "unc_targets": torch.tensor(
                    [target_vals[i] for i in unc_idx], dtype=torch.float32
                ),
                "cert_indices": torch.tensor(cert_idx, dtype=torch.long),
            }
            # Also store delta for compatibility (not used by clamp hooks)
            entry["delta"] = torch.zeros(1)
        else:
            raise ValueError(f"Unknown v2 method: {method}")

        layers[hf_layer] = entry

    return release, layers


def resolve_vuf_residual_layers(
    process_layers: list[int],
    vuf_layers_str: str | None,
    intervention_path_arg: str | None,
    root: Path,
    align_release: str,
) -> list[int]:
    """
    Слои, на которые вешается классический residual VUF.

    Если задан ``vuf_layers_str`` — только они (явная настройка).
    Иначе — пересечение ``process_layers`` с множеством слоёв, где есть SAE-конфиг:
    сначала из ``intervention_path`` (если файл есть), иначе из ``hf_layers_for_release``.
    """
    if vuf_layers_str:
        out = parse_layers_str(vuf_layers_str)
        if not out:
            raise ValueError("--vuf_layers задан, но список слоёв пуст.")
        print("residual VUF: явный --vuf_layers:", out)
        return out

    inter_path: Path | None = None
    if intervention_path_arg:
        p = Path(intervention_path_arg)
        inter_path = p if p.is_file() else (root / intervention_path_arg)
        if not inter_path.is_file():
            inter_path = None

    if inter_path is not None:
        _, meta = load_intervention(inter_path)
        sae_layers = set(meta.keys())
        source = f"intervention {inter_path}"
    else:
        sae_layers = {h for h, _ in hf_layers_for_release(align_release)}
        source = f"release {align_release!r} (intervention.pt не найден)"

    aligned = sorted(set(process_layers) & sae_layers)
    if not aligned:
        raise ValueError(
            "Пересечение --str_process_layers с SAE-слоями пусто "
            f"({source}). Задайте --vuf_layers явно (например те же слои, что в intervention) "
            "или расширьте --str_process_layers / --vuf_align_release."
        )
    print(
        "residual VUF: слои (совпадают с подмножеством SAE после пересечения с str_process_layers):",
        aligned,
        f"[источник SAE-слоёв: {source}]",
    )
    return aligned


def load_saes_for_layers(release: str, layers: dict[int, dict], sae_dtype: str) -> dict[int, SAE]:
    out: dict[int, SAE] = {}
    for hf_layer, meta in layers.items():
        sae_id = meta["sae_id"]
        print(f"Loading SAE {release} / {sae_id} …")
        sae = SAE.from_pretrained(release, sae_id, device="cpu", dtype=sae_dtype)
        out[hf_layer] = sae
    return out


def get_answers_muc(
    questions: list,
    alphas: np.ndarray,
    detection_res: list,
    out_file: str,
    hook_layers: list[int],
    model,
    tokenizer,
    prompt_type: str,
    steering: str,
    layer_to_sae: dict[int, SAE] | None,
    layer_to_delta: dict[int, torch.Tensor] | None,
    hedge_2d: torch.Tensor | None,
    layer_to_clamp: dict[int, dict] | None = None,
    gen_batch_size: int = 16,
    apply_during_generation: bool = True,
    greedy_only: bool = False,
) -> None:
    print("will save to", out_file)

    if os.path.exists(out_file):
        with jsonlines.open(out_file, "r") as reader:
            history_len = len(list(reader))
    else:
        history_len = 0
    print("history_len", history_len)
    assert len(questions) == len(alphas) == len(detection_res)

    n = len(questions)
    bsz = max(1, int(gen_batch_size))
    print("gen_batch_size (подряд одинаковые α и detection):", bsz)

    def _skip_generation(alpha_f: float, dr) -> bool:
        return float(alpha_f) == 0.0 or dr == 0

    def _register_hooks(alpha_f: float) -> None:
        clear_vuf_residual_hooks(model)
        clear_sae_latent_hooks(model)

        if steering in ("sae", "sae_emd", "sae_projected_vuf"):
            if layer_to_sae is None or layer_to_delta is None:
                raise RuntimeError(f"{steering} steering requires layer_to_sae and layer_to_delta.")
            register_sae_latent_hooks(
                model,
                layer_to_sae,
                layer_to_delta,
                hook_layers,
                float(alpha_f),
                apply_during_generation=apply_during_generation,
            )
        elif steering == "sae_clamp":
            if layer_to_sae is None or layer_to_clamp is None:
                raise RuntimeError("sae_clamp steering requires layer_to_sae and layer_to_clamp.")
            register_sae_clamp_hooks(
                model,
                layer_to_sae,
                layer_to_clamp,
                hook_layers,
                float(alpha_f),
                apply_during_generation=apply_during_generation,
            )
        elif steering == "residual":
            if hedge_2d is None:
                raise RuntimeError("Residual (article) steering requires hedge_2d.")
            register_vuf_residual_hooks(
                model,
                hedge_2d,
                hook_layers,
                float(alpha_f),
                apply_during_generation=apply_during_generation,
            )
        else:
            raise ValueError(f"Unknown steering: {steering}")

    i = history_len
    with tqdm(total=n, initial=i, desc="questions") as pbar:
        while i < n:
            a0 = float(alphas[i])
            dr0 = detection_res[i]

            if _skip_generation(a0, dr0):
                clear_sae_latent_hooks(model)
                clear_vuf_residual_hooks(model)
                with jsonlines.open(out_file, "a") as writer:
                    while i < n:
                        ai = float(alphas[i])
                        dri = detection_res[i]
                        if not _skip_generation(ai, dri):
                            break
                        writer.write(
                            {
                                "alpha": 0,
                                "question": questions[i],
                                "most_likely_answer": "",
                                "responses": [],
                            }
                        )
                        i += 1
                        pbar.update(1)
                if i % 100 == 0:
                    torch.cuda.empty_cache()
                continue

            run_start = i
            while i < n:
                if float(alphas[i]) != a0 or detection_res[i] != dr0:
                    break
                if _skip_generation(float(alphas[i]), detection_res[i]):
                    break
                i += 1
            run_end = i
            i = run_start

            _register_hooks(a0)
            while i < run_end:
                chunk_end = min(i + bsz, run_end)
                batch_q = questions[i:chunk_end]
                batch_m = [_chat_messages_for_question(q, prompt_type) for q in batch_q]
                lines = generate_lines_for_batch(
                    model, tokenizer, batch_q, batch_m, float(a0),
                    greedy_only=greedy_only,
                )
                with jsonlines.open(out_file, "a") as writer:
                    for line in lines:
                        writer.write(line)
                pbar.update(chunk_end - i)
                i = chunk_end
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
    parser.add_argument(
        "--alpha_step",
        type=float,
        default=0.0,
        help="Если > 0: α после clip округляются до кратных этому шагу (напр. 5 при max_alpha=20). "
        "Если 0: как раньше — np.round(..., 4).",
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, required=True)
    parser.add_argument(
        "--intervention_path",
        type=str,
        default="sae_muc/artifacts/mistral_intervention.pt",
        help="Для --steering sae: конфиг с δ по слоям (от build_intervention_config).",
    )
    parser.add_argument(
        "--hedge_path",
        type=str,
        default=None,
        help="Для --steering residual: Hs_hedge_universal.pt [n_layers, d_model].",
    )
    parser.add_argument(
        "--steering",
        type=str,
        choices=("sae", "sae_emd", "sae_projected_vuf", "sae_clamp", "residual"),
        default="sae",
        help=(
            "sae: v1 latent bump (hedge projection); "
            "sae_emd: v2 EMD with consensus features; "
            "sae_projected_vuf: v2 SAE-projected VUF; "
            "sae_clamp: v2 feature clamping; "
            "residual: h←h+α·r̂ (raw VUF from paper)."
        ),
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Dry run: только первые N строк датасета (после загрузки CSV).",
    )
    parser.add_argument(
        "--gen_batch_size",
        type=int,
        default=16,
        help="Сколько примеров с одинаковыми α и detection обрабатывать одним вызовом generate (GPU).",
    )
    parser.add_argument(
        "--vuf_layers",
        type=str,
        default=None,
        help=(
            "Только для --steering residual: на какие HF-слои вешать h←h+α·r̂. "
            "Формат как у str_process_layers (range(15,32)) или список: 15,23. "
            "Если не задано: пересечение str_process_layers с слоями из intervention.pt "
            "(если файл есть) иначе с --vuf_align_release — как у SAE для честного сравнения."
        ),
    )
    parser.add_argument(
        "--vuf_align_release",
        type=str,
        default="mistral-7b-res-wg",
        help=(
            "Релиз SAELens для авто-списка HF-слоёв, когда --vuf_layers не задан "
            "и intervention.pt не найден (см. sae_muc.layer_map)."
        ),
    )
    parser.add_argument("--sae_dtype", type=str, default="float32")
    parser.add_argument(
        "--apply_during_generation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply steering hooks during autoregressive generation steps "
            "(seq_len=1), not only during prefill. Default: True."
        ),
    )
    parser.add_argument(
        "--greedy_only",
        action="store_true",
        default=False,
        help=(
            "Only generate the greedy (most_likely) answer. "
            "Skips the 10 sampled responses — much faster."
        ),
    )
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
    if args.steering == "residual":
        hook_layers = resolve_vuf_residual_layers(
            process_layers,
            args.vuf_layers,
            args.intervention_path,
            root,
            args.vuf_align_release,
        )
    else:
        hook_layers = process_layers

    is_v2_method = args.steering in ("sae_emd", "sae_projected_vuf", "sae_clamp")

    results_df = pd.read_csv(root / "datasets" / dataset / model_name / f"{split}.csv")
    if args.max_questions is not None:
        n = max(0, int(args.max_questions))
        results_df = results_df.iloc[:n].copy()
        print("max_questions (dry run):", n, "rows")
    vu_scores_llm = results_df["verbal_uncertainty"].to_numpy()
    su_scores = results_df["sentence_semantic_entropy"].to_numpy()
    questions = results_df["question"].tolist()

    MAX_SE = 2.302585092994045
    MAX_ALPHA = args.max_alpha
    detection_res = load_detection_res(root, dataset, model_name, split)
    nq = len(questions)
    if len(detection_res) < nq:
        raise ValueError(
            f"detection y_pred length {len(detection_res)} < number of questions {nq}"
        )
    detection_res = detection_res[:nq]

    if args.iti_method != 2:
        raise NotImplementedError("Only iti_method=2.")

    alphas = (su_scores / MAX_SE - vu_scores_llm) * MAX_ALPHA
    alphas = np.clip(alphas, 0, MAX_ALPHA)
    step = float(args.alpha_step)
    if step > 0:
        alphas = np.clip(np.round(alphas / step) * step, 0, MAX_ALPHA)
    else:
        alphas = np.round(alphas, 4)

    base_name = f"with_vufi_{args.iti_method}_{args.str_process_layers}_{args.max_alpha}"
    if args.prompt_type != "uncertainty":
        base_name += f"_{args.prompt_type}"
    if args.steering != "sae":
        base_name += f"_{args.steering}"
    if args.steering == "residual":
        base_name += "_L" + "-".join(str(l) for l in hook_layers)
    if args.max_questions is not None:
        base_name += f"_first{args.max_questions}"
    jsonl_name = base_name + ".jsonl"
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

    layer_to_sae: dict[int, SAE] | None = None
    layer_to_delta: dict[int, torch.Tensor] | None = None
    layer_to_clamp: dict[int, dict] | None = None
    hedge_2d: torch.Tensor | None = None

    if args.steering in _SAE_STEERING_METHODS:
        inter_path = Path(args.intervention_path)
        if not inter_path.is_file():
            inter_path = root / args.intervention_path

        if is_v2_method:
            # v2 config from build_intervention_config_v2
            release, layer_meta = load_intervention_v2(inter_path, args.steering)
            if args.steering == "sae_clamp":
                layer_to_clamp = {l: layer_meta[l]["clamp_config"] for l in layer_meta}
                # delta not needed for clamp but load_saes_for_layers needs sae_id
                layer_to_delta = {}
            else:
                layer_to_delta = {l: layer_meta[l]["delta"] for l in layer_meta}
        else:
            # v1 config from build_intervention_config (legacy)
            release, layer_meta = load_intervention(inter_path)
            layer_to_delta = {l: layer_meta[l]["delta"] for l in layer_meta}

        layer_to_sae = load_saes_for_layers(release, layer_meta, args.sae_dtype)
        print("repo_root", root)
        print("steering", args.steering)
        print("process_layers", process_layers)
        print("SAE hooks on layers", [l for l in process_layers if l in layer_to_sae])
        skipped = [l for l in process_layers if l not in layer_to_sae]
        if skipped:
            print("No SAE config for HF layers (skipped):", skipped)
    elif args.steering == "residual":
        hp = Path(args.hedge_path) if args.hedge_path else None
        if hp is None or not hp.is_file():
            hp2 = root / args.hedge_path if args.hedge_path else None
            if hp2 and hp2.is_file():
                hp = hp2
        if hp is None or not hp.is_file():
            raise FileNotFoundError(
                "For --steering residual, provide a valid --hedge_path "
                "(e.g. calibration/.../Hs_hedge_universal.pt)."
            )
        hedge_2d = torch.load(hp, map_location="cpu", weights_only=False)
        if hedge_2d.ndim != 2:
            raise ValueError(f"Hs_hedge expected [n_layers, d_model], got {tuple(hedge_2d.shape)}")
        print("repo_root", root)
        print("residual hook_layers:", hook_layers)
        print("residual steering hedge from", hp)
    else:
        raise ValueError(f"Unknown steering: {args.steering}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        full_model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(full_model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    get_answers_muc(
        questions,
        alphas,
        detection_res,
        out_file,
        hook_layers,
        model,
        tokenizer,
        prompt_type,
        args.steering,
        layer_to_sae,
        layer_to_delta,
        hedge_2d,
        layer_to_clamp=layer_to_clamp,
        gen_batch_size=args.gen_batch_size,
        apply_during_generation=args.apply_during_generation,
        greedy_only=args.greedy_only,
    )
    os.environ["RUN_MUC_LAST_JSONL"] = str(Path(out_file).resolve())
    print("RUN_MUC_LAST_JSONL", os.environ["RUN_MUC_LAST_JSONL"], flush=True)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
