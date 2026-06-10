"""Paper per-model constants: App E.1 layer ranges + App G.1 max_alpha.

`paper_layer_range` is used when `cfg.stages.intervene.layer == "paper_range"`:
each substring match returns the inclusive VUF-intervention layer range.
`paper_max_alpha` is used when `cfg.stages.intervene.alpha_max == "paper"`:
the per-model MUC steering ceiling from App G.1. Both match by case-insensitive
substring on the model name — extend the tables as needed.
"""

from __future__ import annotations


_PAPER_RANGES: tuple[tuple[str, tuple[int, int]], ...] = (
    # Llama-3.1-8B-Instruct: layers 15-31 (App E.1, Tab.5).
    ("llama", (15, 31)),
    # Mistral-7B-Instruct: layers 15-31.
    ("mistral", (15, 31)),
    # Qwen2.5-7B-Instruct: layers 16-27.
    ("qwen", (16, 27)),
)

# App G.1 p.29 (verbatim): "We set max_α = 1.0 for Llama-3.1-8B, 0.4 for
# Mistral-7B, 3.0 for Qwen2.5-7B, and 4.0 for Llama-3.1-70B across three
# datasets." Order matters: the 70B entry is tried before the generic "llama"
# so a 70B model resolves to 4.0 rather than the 8B's 1.0.
_PAPER_MAX_ALPHA: tuple[tuple[str, float], ...] = (
    ("70b", 4.0),
    ("mistral", 0.4),
    ("qwen", 3.0),
    ("llama", 1.0),
)


def paper_layer_range(model_name: str) -> list[int]:
    """Return the inclusive list of layers for `model_name` per paper App E.1.

    Match is by case-insensitive substring on the model name (e.g.
    "meta-llama/Llama-3.1-8B-Instruct" matches "llama"). Raises
    ValueError if no entry matches — the caller must either name the
    layer(s) explicitly or extend the table.
    """
    name = model_name.lower()
    for needle, (start, end) in _PAPER_RANGES:
        if needle in name:
            return list(range(start, end + 1))
    raise ValueError(
        f"intervene.layer='paper_range' but no App E.1 entry matches "
        f"model name {model_name!r}; extend paper_layer_ranges._PAPER_RANGES "
        f"or use an explicit layer / list[int]."
    )


def paper_max_alpha(model_name: str) -> float:
    """Return the per-model MUC steering ceiling for `model_name` per App G.1.

    Match is by case-insensitive substring (the 70B entry is checked first so
    a 70B Llama resolves to 4.0, not the 8B's 1.0). Raises ValueError if no
    entry matches — the caller must set an explicit float alpha_max instead.
    """
    name = model_name.lower()
    for needle, value in _PAPER_MAX_ALPHA:
        if needle in name:
            return value
    raise ValueError(
        f"intervene.alpha_max='paper' but no App G.1 entry matches model name "
        f"{model_name!r}; extend paper_layer_ranges._PAPER_MAX_ALPHA or set an "
        f"explicit float alpha_max."
    )
