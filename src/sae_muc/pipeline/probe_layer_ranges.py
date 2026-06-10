"""Paper App F.1 probe layer ranges (per dataset, per uncertainty).

App F.1 p.28 (verbatim): the uncertainty probes are "linear models trained on
the hidden states of LLMs ... sourced from multiple layers within the LLM. We
have selected the following layers based on the performance for each
uncertainty:
  • VU: Layers 5 to 20 for TriviaQA, 10 to 20 for NQ-Open, and 5 to 20 for PopQA.
  • SU: Layers 10 to 20 for TriviaQA, 10 to 20 for NQ-Open, and 5 to 25 for PopQA."

Used by:
  * the Probe-Predicted detector path (two regressor probes, paper §4.1 /
    Table 2) — each regressor reads its own uncertainty's range;
  * the hidden-state classifier probe (lr_hidden / combined) when
    `detect.detector_layer == "paper_range"` — uses the union of the two
    ranges for the dataset.
Ranges are inclusive. Keyed by dataset name (case-insensitive); the paper
reports them on Llama-3.1-8B and reuses them across models.
"""

from __future__ import annotations

_PROBE_RANGES: dict[tuple[str, str], tuple[int, int]] = {
    ("triviaqa", "vu"): (5, 20),
    ("nq_open", "vu"): (10, 20),
    ("popqa", "vu"): (5, 20),
    ("triviaqa", "su"): (10, 20),
    ("nq_open", "su"): (10, 20),
    ("popqa", "su"): (5, 25),
}


def probe_layer_range(dataset: str, uncertainty: str) -> list[int]:
    """Inclusive App F.1 layer list for (`dataset`, `uncertainty` ∈ {vu, su})."""
    key = (dataset.lower(), uncertainty.lower())
    if key not in _PROBE_RANGES:
        raise ValueError(
            f"no App F.1 probe range for dataset={dataset!r} uncertainty={uncertainty!r}; "
            f"known datasets: triviaqa / nq_open / popqa, uncertainties: vu / su. "
            f"Extend probe_layer_ranges._PROBE_RANGES or set an explicit detector_layer."
        )
    start, end = _PROBE_RANGES[key]
    return list(range(start, end + 1))


def probe_layer_union(dataset: str) -> list[int]:
    """Sorted union of the VU and SU App F.1 ranges for `dataset`.

    The classifier (hallucination) probe is not per-uncertainty, so when
    `detector_layer == "paper_range"` it sources from both ranges.
    """
    vu = probe_layer_range(dataset, "vu")
    su = probe_layer_range(dataset, "su")
    return sorted(set(vu) | set(su))
