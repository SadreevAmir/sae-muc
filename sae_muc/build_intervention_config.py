#!/usr/bin/env python3
"""
Build per-layer latent intervention vectors (delta) by projecting Hs_hedge onto SAE decoder directions.

Example:
  python sae_muc/build_intervention_config.py \\
    --hedge_path calibration/outputs/merged/Mistral-7B-Instruct-v0.3/uncertainty/Hs_hedge_universal.pt \\
    --out_path sae_muc/artifacts/mistral_intervention.pt \\
    --release mistral-7b-res-wg \\
    --top_k 64 \\
    --sae_dtype float32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from sae_lens import SAE

_PKG_DIR = Path(__file__).resolve().parent
_REPO_PARENT = _PKG_DIR.parent
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

from sae_muc.layer_map import hf_layers_for_release


def _delta_from_hedge_row(
    h_row: torch.Tensor,
    sae: SAE,
    top_k: int,
    device: str,
) -> torch.Tensor:
    """Sparse-normalized delta in feature space from one residual direction."""
    h = h_row.to(device=device, dtype=torch.float32)
    h = h / (h.norm() + 1e-8)
    w_dec = sae.W_dec.detach().to(device=device, dtype=torch.float32)
    # [d_sae]
    scores = (h @ w_dec.T) / (w_dec.norm(dim=1) + 1e-8)
    k = min(top_k, scores.numel())
    top = torch.topk(scores.abs(), k=k)
    delta = torch.zeros_like(scores)
    delta[top.indices] = scores[top.indices]
    delta = delta / (delta.norm() + 1e-8)
    return delta.cpu()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hedge_path", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--release", type=str, default="mistral-7b-res-wg")
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--sae_device", type=str, default="cpu")
    p.add_argument("--sae_dtype", type=str, default="float32", choices=("float32", "float16", "bfloat16"))
    args = p.parse_args()

    hedge = torch.load(args.hedge_path, map_location="cpu")
    if hedge.ndim != 2:
        raise ValueError(f"Expected Hs_hedge [n_layers, d_model], got {tuple(hedge.shape)}")

    n_layers, d_model = hedge.shape
    mapping = hf_layers_for_release(args.release)

    layers_out: dict[int, dict] = {}
    for hf_layer, sae_id in mapping:
        if hf_layer >= n_layers:
            continue
        print(f"Layer HF={hf_layer} sae_id={sae_id} …")
        sae = SAE.from_pretrained(args.release, sae_id, device=args.sae_device, dtype=args.sae_dtype)
        if sae.cfg.d_in != d_model:
            raise ValueError(
                f"d_in mismatch: SAE expects {sae.cfg.d_in}, hedge has d_model={d_model}"
            )
        h_row = hedge[hf_layer]
        delta = _delta_from_hedge_row(h_row, sae, args.top_k, args.sae_device)
        layers_out[hf_layer] = {"sae_id": sae_id, "delta": delta}

    if not layers_out:
        raise RuntimeError("No layers written; check hedge shape vs layer map.")

    out = {
        "release": args.release,
        "hedge_path": str(Path(args.hedge_path).resolve()),
        "top_k": args.top_k,
        "layers": layers_out,
    }
    outp = Path(args.out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, outp)
    print(f"Saved {len(layers_out)} layer entries to {outp}")


if __name__ == "__main__":
    main()
