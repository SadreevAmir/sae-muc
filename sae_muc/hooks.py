"""
Forward hooks for SAE-based steering with 3 methods:

  1. emd (latent bump):      f' = f + α·δ,  h' = decode(f') + (h - decode(f))
  2. projected_vuf:          same as emd but δ is SAE-projected VUF
  3. clamp:                  raise uncertainty features to target, suppress certainty features

All methods preserve the SAE reconstruction error term: h' = decode(f') + err.
"""

from __future__ import annotations

from typing import Any

import torch
from sae_lens import SAE


# --------------------------------------------------------------------------- #
#  Common SAE encode/decode helper                                             #
# --------------------------------------------------------------------------- #

def _sae_encode_decode_err(
    flat_work: torch.Tensor,
    sae: SAE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (features, reconstruction, error) for flat input."""
    f = sae.encode(flat_work)
    recon = sae.decode(f)
    err = flat_work - recon
    return f, recon, err


def _sae_decode_with_norm_fix(
    sae: SAE,
    f_new: torch.Tensor,
    flat_work: torch.Tensor,
) -> torch.Tensor:
    """Decode f_new, handling constant_norm_rescale SAEs that need re-encoding."""
    if getattr(sae.cfg, "normalize_activations", None) == "constant_norm_rescale":
        sae.encode(flat_work)
    return sae.decode(f_new)


def _check_sae_device(sae: SAE) -> tuple[torch.device, torch.dtype]:
    p0 = next(sae.parameters())
    if p0.is_meta or str(p0.device) == "meta":
        raise RuntimeError(
            "SAE weights on 'meta' device (no data). Restart runtime and reload SAE."
        )
    return p0.device, p0.dtype


def _prepare_flat(x: torch.Tensor, sae: SAE):
    """Flatten x, move to SAE device/dtype. Returns flat_work, orig_dtype, orig_shape, dev."""
    orig_dtype = x.dtype
    orig_shape = x.shape
    flat = x.reshape(-1, orig_shape[-1])
    dev = flat.device
    sae_dev, sae_dtype = _check_sae_device(sae)
    flat_work = flat.to(device=sae_dev, dtype=sae_dtype)
    return flat_work, orig_dtype, orig_shape, dev


def _finalize(out: torch.Tensor, orig_dtype, orig_shape, dev) -> torch.Tensor:
    if getattr(out, "is_meta", False):
        raise RuntimeError("SAE returned meta tensor. Restart runtime and reload SAE.")
    return out.to(dtype=orig_dtype, device=dev).view(orig_shape)


# --------------------------------------------------------------------------- #
#  Method 1 & 2: Latent bump (EMD / Projected VUF)                            #
# --------------------------------------------------------------------------- #

def _apply_sae_latent_bump(
    x: torch.Tensor,
    sae: SAE[Any],
    delta: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    Encode-Modify-Decode with error term.
    x: [..., d_in], delta: [d_sae].  h' = decode(f + α·δ) + (h - decode(f))
    """
    flat_work, orig_dtype, orig_shape, dev = _prepare_flat(x, sae)
    sae_dev, sae_dtype = _check_sae_device(sae)

    d = delta.to(device=sae_dev, dtype=sae_dtype)
    if d.ndim != 1:
        raise ValueError("delta must be 1-D [d_sae]")
    a = torch.tensor(alpha, device=sae_dev, dtype=sae_dtype)

    with torch.no_grad():
        f, recon, err = _sae_encode_decode_err(flat_work, sae)
        f2 = f + a * d.unsqueeze(0)
        recon2 = _sae_decode_with_norm_fix(sae, f2, flat_work)
        out = recon2 + err

    return _finalize(out, orig_dtype, orig_shape, dev)


# --------------------------------------------------------------------------- #
#  Method 3: Feature clamping                                                  #
# --------------------------------------------------------------------------- #

def _apply_sae_clamp(
    x: torch.Tensor,
    sae: SAE[Any],
    clamp_config: dict,
    alpha: float,
) -> torch.Tensor:
    """
    Feature clamping with error term preservation.

    For uncertainty features: f'[i] = f[i] + α * max(0, target[i] - f[i])
      (raise toward target activation of uncertain samples)
    For certainty features:   f'[i] = f[i] * (1 - α)
      (soft suppression toward zero)

    h' = decode(f') + (h - decode(f))

    clamp_config keys:
      unc_indices:  LongTensor of uncertainty feature indices
      unc_targets:  FloatTensor of target values (mean activation on uncertain samples)
      cert_indices: LongTensor of certainty feature indices
    """
    flat_work, orig_dtype, orig_shape, dev = _prepare_flat(x, sae)
    sae_dev, sae_dtype = _check_sae_device(sae)

    unc_idx = clamp_config["unc_indices"].to(device=sae_dev)
    unc_tgt = clamp_config["unc_targets"].to(device=sae_dev, dtype=sae_dtype)
    cert_idx = clamp_config["cert_indices"].to(device=sae_dev)
    a = min(max(float(alpha), 0.0), 1.0)  # clamp alpha to [0, 1] for clamping

    with torch.no_grad():
        f, recon, err = _sae_encode_decode_err(flat_work, sae)
        f_new = f.clone()

        # Uncertainty features: push up toward target
        if unc_idx.numel() > 0:
            current_unc = f_new[:, unc_idx]
            target_expanded = unc_tgt.unsqueeze(0).expand_as(current_unc)
            gap = torch.clamp(target_expanded - current_unc, min=0.0)
            f_new[:, unc_idx] = current_unc + a * gap

        # Certainty features: suppress toward zero
        if cert_idx.numel() > 0:
            f_new[:, cert_idx] = f_new[:, cert_idx] * (1.0 - a)

        recon2 = _sae_decode_with_norm_fix(sae, f_new, flat_work)
        out = recon2 + err

    return _finalize(out, orig_dtype, orig_shape, dev)


# --------------------------------------------------------------------------- #
#  Hook registration (unified for all methods)                                 #
# --------------------------------------------------------------------------- #

def register_sae_latent_hooks(
    model: torch.nn.Module,
    layer_to_sae: dict[int, SAE[Any]],
    layer_to_delta: dict[int, torch.Tensor],
    process_layers: list[int],
    alpha: float,
    apply_during_generation: bool = True,
) -> None:
    """
    Register EMD / projected_vuf hooks (methods 1 & 2).
    delta per layer is a [d_sae] tensor (direction in feature space).

    apply_during_generation: if True, the hook fires on every forward pass
        (including single-token autoregressive steps). If False, it only
        fires during prefill (seq_len > 1).
    """
    _clear_and_init(model)

    for l in process_layers:
        if l not in layer_to_sae or l not in layer_to_delta:
            continue
        sae = layer_to_sae[l]
        delta = layer_to_delta[l]
        _ensure_sae_on_device(sae, model.model.layers[l])

        def make_hook(sae_ref=sae, delta_ref=delta, alpha_val=alpha,
                      gen=apply_during_generation):
            def hook_fn(module, inputs, outputs):
                return _apply_hook_to_outputs(
                    outputs,
                    lambda h: _apply_sae_latent_bump(h, sae_ref, delta_ref, alpha_val),
                    apply_during_generation=gen,
                )
            return hook_fn

        handle = model.model.layers[l].register_forward_hook(make_hook())
        model._sae_muc_hooks[l] = handle


def register_sae_clamp_hooks(
    model: torch.nn.Module,
    layer_to_sae: dict[int, SAE[Any]],
    layer_to_clamp: dict[int, dict],
    process_layers: list[int],
    alpha: float,
    apply_during_generation: bool = True,
) -> None:
    """
    Register feature clamping hooks (method 3).
    clamp_config per layer has unc_indices, unc_targets, cert_indices.

    apply_during_generation: if True, the hook fires on every forward pass
        (including single-token autoregressive steps). If False, it only
        fires during prefill (seq_len > 1).
    """
    _clear_and_init(model)

    for l in process_layers:
        if l not in layer_to_sae or l not in layer_to_clamp:
            continue
        sae = layer_to_sae[l]
        clamp_cfg = layer_to_clamp[l]
        _ensure_sae_on_device(sae, model.model.layers[l])

        def make_hook(sae_ref=sae, cfg_ref=clamp_cfg, alpha_val=alpha,
                      gen=apply_during_generation):
            def hook_fn(module, inputs, outputs):
                return _apply_hook_to_outputs(
                    outputs,
                    lambda h: _apply_sae_clamp(h, sae_ref, cfg_ref, alpha_val),
                    apply_during_generation=gen,
                )
            return hook_fn

        handle = model.model.layers[l].register_forward_hook(make_hook())
        model._sae_muc_hooks[l] = handle


# --------------------------------------------------------------------------- #
#  Internal helpers                                                            #
# --------------------------------------------------------------------------- #

def _clear_and_init(model: torch.nn.Module) -> None:
    if not hasattr(model, "_sae_muc_hooks"):
        model._sae_muc_hooks = {}
    for l in list(model._sae_muc_hooks.keys()):
        model._sae_muc_hooks[l].remove()
    model._sae_muc_hooks.clear()


def _ensure_sae_on_device(sae: SAE, layer_mod: torch.nn.Module) -> None:
    device = next(layer_mod.parameters()).device
    if next(sae.parameters()).device != device:
        sae.to(device=device)


def _apply_hook_to_outputs(outputs, transform_fn, *, apply_during_generation: bool = True):
    """Apply transform_fn to the hidden state tensor in layer outputs.

    When apply_during_generation is False, single-token steps (seq_len <= 1,
    i.e. autoregressive decoding) are skipped — the hook only fires during
    prefill.  When True (default), the hook fires on every forward pass.
    """
    if torch.is_tensor(outputs):
        if not apply_during_generation and outputs.shape[1] <= 1:
            return outputs
        return transform_fn(outputs)
    if not isinstance(outputs, tuple):
        raise TypeError(
            f"Unexpected decoder layer output type {type(outputs)}; "
            "expected Tensor or tuple."
        )
    h = outputs[0]
    if not apply_during_generation and h.shape[1] <= 1:
        return outputs
    h2 = transform_fn(h)
    return (h2,) + outputs[1:]


def clear_sae_latent_hooks(model: torch.nn.Module) -> None:
    if not hasattr(model, "_sae_muc_hooks"):
        return
    for l, h in list(model._sae_muc_hooks.items()):
        h.remove()
    model._sae_muc_hooks.clear()
