"""
Raw residual VUF steering (Meta `causal.py` / `semantic_control.py`): h ← h + α * r̂,
без SAE. `Hs_hedge` форма [n_layers, d_model] как в `Hs_hedge_universal.pt`.
"""

from __future__ import annotations

import torch


def clear_vuf_residual_hooks(model: torch.nn.Module) -> None:
    if not hasattr(model, "_vuf_muc_hooks"):
        return
    for _l, h in list(model._vuf_muc_hooks.items()):
        h.remove()
    model._vuf_muc_hooks.clear()


def register_vuf_residual_hooks(
    model: torch.nn.Module,
    hedge_2d: torch.Tensor,
    process_layers: list[int],
    alpha: float,
    apply_during_generation: bool = True,
) -> None:
    """
    hedge_2d: [n_layers, d_model] на CPU; для слоя l берётся строка hedge_2d[l], L2-нормируется.

    apply_during_generation: if True, the hook fires on every forward pass
        (including single-token autoregressive steps). If False, it only
        fires during prefill (seq_len > 1).
    """
    if hedge_2d.ndim != 2:
        raise ValueError(f"Expected Hs_hedge [n_layers, d_model], got {tuple(hedge_2d.shape)}")

    clear_vuf_residual_hooks(model)
    model._vuf_muc_hooks = {}

    for l in process_layers:
        if l < 0 or l >= hedge_2d.shape[0]:
            continue
        h = hedge_2d[l].float()
        h = h / (h.norm() + 1e-8)
        layer_mod = model.model.layers[l]
        dev = next(layer_mod.parameters()).device
        dt = next(layer_mod.parameters()).dtype
        h_vec = h.to(device=dev, dtype=dt)
        alpha_f = float(alpha)

        def make_hook(h_ref, a_f: float, gen=apply_during_generation):
            def hook_fn(module, inputs, outputs):
                if torch.is_tensor(outputs):
                    h_out = outputs
                    if gen or h_out.shape[1] > 1:
                        h_out = h_out + h_ref * a_f
                    return h_out
                if not isinstance(outputs, tuple):
                    raise TypeError(f"Unexpected layer output {type(outputs)}")
                o0 = outputs[0]
                if gen or o0.shape[1] > 1:
                    o0 = o0 + h_ref * a_f
                return (o0,) + outputs[1:]

            return hook_fn

        handle = layer_mod.register_forward_hook(make_hook(h_vec, alpha_f))
        model._vuf_muc_hooks[l] = handle
