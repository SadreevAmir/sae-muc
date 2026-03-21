"""
Map TransformerLens SAE hook ids (resid_pre) to HuggingFace decoder layer indices.

TL `blocks.L.hook_resid_pre` is the residual stream at the input of block L,
i.e. the same tensor as the output of HF `model.layers[L-1]`.
"""

from __future__ import annotations

# (hf_layer_index_after_block, sae_id) for mistral-7b-res-wg release
MISTRAL_7B_RES_WG_LAYERS: list[tuple[int, str]] = [
    (7, "blocks.8.hook_resid_pre"),
    (15, "blocks.16.hook_resid_pre"),
    (23, "blocks.24.hook_resid_pre"),
]


def hf_layers_for_release(release: str) -> list[tuple[int, str]]:
    if release == "mistral-7b-res-wg":
        return list(MISTRAL_7B_RES_WG_LAYERS)
    raise ValueError(f"Unknown release {release!r}; extend hf_layers_for_release()")
