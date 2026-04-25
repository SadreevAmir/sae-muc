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
    if layer_cfg == "auto":
        if not available:
            raise ValueError("intervene.layer='auto' but no VUF directions are available.")
        return available[len(available) // 2]
    layer = int(layer_cfg)
    if layer not in available:
        raise ValueError(
            f"intervene.layer={layer} has no VUF direction; available layers: {available}"
        )
    return layer
