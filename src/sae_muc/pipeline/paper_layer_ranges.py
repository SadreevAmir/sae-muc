"""Paper App E.1 layer ranges per model family.

Used when `cfg.stages.intervene.layer == "paper_range"`: each substring
match returns the inclusive range used by the paper for the linear-VUF
intervention. Substring match (case-insensitive) — extend as needed.
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
