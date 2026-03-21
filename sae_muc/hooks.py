"""
Forward hooks: latent bump in SAE feature space + residual error term (SAE-native steering).
"""

from __future__ import annotations

from typing import Any

import torch
from sae_lens import SAE


def _apply_sae_latent_bump(
    x: torch.Tensor,
    sae: SAE[Any],
    delta: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    x: [..., d_in] — same dtype as model (often float16).
    delta: [d_sae] — intervention direction in feature space (L2-normalized recommended).
    Returns tensor same shape and dtype as x.
    """
    orig_dtype = x.dtype
    orig_shape = x.shape
    flat = x.reshape(-1, orig_shape[-1])
    dev = flat.device

    sae_dev = next(sae.parameters()).device
    sae_dtype = sae.dtype
    flat_work = flat.to(device=sae_dev, dtype=sae_dtype)

    d = delta.to(device=sae_dev, dtype=sae_dtype)
    if d.ndim != 1:
        raise ValueError("delta must be 1-D [d_sae]")
    a = torch.tensor(alpha, device=sae_dev, dtype=sae_dtype)

    with torch.no_grad():
        f = sae.encode(flat_work)
        recon = sae.decode(f)
        err = flat_work - recon
        f2 = f + a * d.unsqueeze(0)
        # constant_norm_rescale: first decode() deletes x_norm_coeff on the SAE; second decode needs it.
        if getattr(sae.cfg, "normalize_activations", None) == "constant_norm_rescale":
            sae.encode(flat_work)
        recon2 = sae.decode(f2)
        out = recon2 + err

    out = out.to(dtype=orig_dtype, device=dev).view(orig_shape)
    return out


def register_sae_latent_hooks(
    model: torch.nn.Module,
    layer_to_sae: dict[int, SAE[Any]],
    layer_to_delta: dict[int, torch.Tensor],
    process_layers: list[int],
    alpha: float,
) -> None:
    """
    Register hooks on HF `model.model.layers[l]` outputs for layers in
    process_layers that have entries in layer_to_sae / layer_to_delta.
    """
    if not hasattr(model, "_sae_muc_hooks"):
        model._sae_muc_hooks = {}

    for l in list(model._sae_muc_hooks.keys()):
        model._sae_muc_hooks[l].remove()
    model._sae_muc_hooks.clear()

    for l in process_layers:
        if l not in layer_to_sae or l not in layer_to_delta:
            continue
        sae = layer_to_sae[l]
        delta = layer_to_delta[l]

        # device_map="auto" даёт hf_device_map с диапазонами (model.layers.0-15), не по одному ключу на слой
        layer_mod = model.model.layers[l]
        device = next(layer_mod.parameters()).device
        sae.to(device=device, dtype=sae.dtype)

        def make_hook(sae_ref=sae, delta_ref=delta, alpha_val=alpha):
            def hook_fn(module, inputs, outputs):
                # transformers: чаще tuple (h, …); в новых версиях слой может вернуть один Tensor
                if torch.is_tensor(outputs):
                    h = outputs
                    if h.shape[1] <= 1:
                        return outputs
                    h2 = _apply_sae_latent_bump(h, sae_ref, delta_ref, alpha_val)
                    return h2
                if not isinstance(outputs, tuple):
                    raise TypeError(
                        f"Unexpected decoder layer output type {type(outputs)}; "
                        "expected Tensor or tuple."
                    )
                h = outputs[0]
                if h.shape[1] <= 1:
                    return outputs
                h2 = _apply_sae_latent_bump(h, sae_ref, delta_ref, alpha_val)
                return (h2,) + outputs[1:]

            return hook_fn

        handle = model.model.layers[l].register_forward_hook(make_hook())
        model._sae_muc_hooks[l] = handle


def clear_sae_latent_hooks(model: torch.nn.Module) -> None:
    if not hasattr(model, "_sae_muc_hooks"):
        return
    for l, h in list(model._sae_muc_hooks.items()):
        h.remove()
    model._sae_muc_hooks.clear()
