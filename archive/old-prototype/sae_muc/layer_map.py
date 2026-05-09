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

# Llama Scope residual SAEs (SAELens release llama_scope_lxr_32x, fnlp/Llama3_1-8B-Base-LXR-32x).
# Модель в yaml: meta-llama/Llama-3.1-8B; для чата используем Instruct — тот же d_model.
LLAMA_SCOPE_LXR_RES_LAYERS: list[tuple[int, str]] = [
    (15, "l15r_32x"),
    (23, "l23r_32x"),
]


def hf_layers_for_release(release: str) -> list[tuple[int, str]]:
    if release == "mistral-7b-res-wg":
        return list(MISTRAL_7B_RES_WG_LAYERS)
    if release == "llama_scope_lxr_32x":
        return list(LLAMA_SCOPE_LXR_RES_LAYERS)
    raise ValueError(f"Unknown release {release!r}; extend hf_layers_for_release()")


def neuronpedia_residual_slug(release: str, hf_layer: int) -> tuple[str, str] | None:
    """
    (model_id, sae_id) для ссылок вида https://www.neuronpedia.org/{model}/{sae}/{index}.
    Заполнено для релизов, где это есть в SAELens pretrained_saes.yaml.
    """
    if release == "llama_scope_lxr_32x":
        return ("llama3.1-8b", f"{hf_layer}-llamascope-res-131k")
    return None
