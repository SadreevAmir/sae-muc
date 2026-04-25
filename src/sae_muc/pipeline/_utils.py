"""Shared helpers used by multiple pipeline stages.

Hosts pure functions that several stages need (`_pool` for hidden-state
pooling, `_resolve_layer` for picking a single layer from the available
VUF directions). Keeping them here breaks the cycle between
`intervene` ↔ `sae_features` and avoids duplicating logic between
`intervene` and `vuf`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _pool(hs: "torch.Tensor", pooling: str, q_len: int, seq_len: int) -> "torch.Tensor":
    """hs shape: [seq_len, d_model]. Returns [d_model]."""
    q_len = max(1, min(q_len, seq_len))
    if pooling == "last_token_q":
        return hs[q_len - 1]
    if pooling == "last_token_a":
        return hs[-1]
    if pooling == "mean_q":
        return hs[:q_len].mean(dim=0)
    if pooling == "mean_a":
        if q_len >= seq_len:
            # Degenerate: no answer tokens. Fall back to the last question token.
            return hs[q_len - 1]
        return hs[q_len:].mean(dim=0)
    raise ValueError(f"Unknown pooling: {pooling!r}")


def _resolve_layer(layer_cfg: int | str, available: list[int]) -> int:
    """Single-layer convenience wrapper around `_resolve_layers`.

    Raises if the resolved spec covers more than one layer — callers that
    accept multi-layer specs must call `_resolve_layers` directly.
    """
    layers = _resolve_layers(layer_cfg, available)
    if len(layers) != 1:
        raise ValueError(
            f"intervene.layer={layer_cfg!r} resolves to {len(layers)} layers; "
            f"this caller expects exactly one (use _resolve_layers)."
        )
    return layers[0]


def _resolve_layers(
    layer_cfg, available: list[int], *, model_name: str | None = None
) -> list[int]:
    """Resolve `cfg.stages.intervene.layer` (or vuf/sae_features layer spec) to a list.

    Accepts:
      - int                   → [int]  (must be in `available`)
      - list[int]             → ordered, deduplicated subset of `available`
      - "auto"                → [middle of `available`]
      - "paper_range"         → App E.1 range for `model_name`, intersected
                                with `available`
    """
    from sae_muc.pipeline.paper_layer_ranges import paper_layer_range

    if isinstance(layer_cfg, list):
        if not layer_cfg:
            raise ValueError("intervene.layer=[] is empty; specify at least one layer.")
        seen: set[int] = set()
        out: list[int] = []
        for raw in layer_cfg:
            layer = int(raw)
            if layer in seen:
                continue
            if layer not in available:
                raise ValueError(
                    f"intervene.layer={layer} has no VUF direction; "
                    f"available layers: {available}"
                )
            seen.add(layer)
            out.append(layer)
        return out
    if layer_cfg == "auto":
        if not available:
            raise ValueError("intervene.layer='auto' but no VUF directions are available.")
        return [available[len(available) // 2]]
    if layer_cfg == "paper_range":
        if model_name is None:
            raise ValueError(
                "intervene.layer='paper_range' requires the model name to look "
                "up the App E.1 range; pass model_name=cfg.model.name."
            )
        paper_layers = paper_layer_range(model_name)
        intersected = [l for l in paper_layers if l in available]
        if not intersected:
            raise ValueError(
                f"intervene.layer='paper_range' resolved to {paper_layers} for "
                f"{model_name!r}, but none of those are in available={available}."
            )
        return intersected
    layer = int(layer_cfg)
    if layer not in available:
        raise ValueError(
            f"intervene.layer={layer} has no VUF direction; available layers: {available}"
        )
    return [layer]
