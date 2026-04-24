"""semantic_entropy: cluster sampled answers by bidirectional entailment, compute SE.

Follows Kuhn et al. 2023: two answers a and b are considered semantically
equivalent iff a entails b AND b entails a. Clusters grow greedily in
arrival order; semantic entropy is the natural-log entropy of the cluster-
size distribution per question.

Only the N high-T sampled generations are clustered; the greedy answer is
left out of SE to stay faithful to the paper's Appendix C protocol.
"""

from __future__ import annotations

import logging
import math
from collections import Counter

import pandas as pd

from sae_muc.models.nli import NLIBackend
from sae_muc.pipeline.context import PipelineContext

log = logging.getLogger(__name__)

INPUT = "generations.parquet"
OUTPUT = "semantic_entropy.parquet"


def _cluster_by_entailment(answers: list[str], nli: NLIBackend) -> list[int]:
    """Greedy bidirectional-entailment clustering. Returns cluster index per answer."""
    cluster_of: list[int] = []
    reps: list[str] = []
    for ans in answers:
        if reps:
            forward = nli.entails([(ans, r) for r in reps])
            backward = nli.entails([(r, ans) for r in reps])
            matched = next(
                (i for i, (a, b) in enumerate(zip(forward, backward, strict=True)) if a and b),
                None,
            )
        else:
            matched = None
        if matched is not None:
            cluster_of.append(matched)
        else:
            cluster_of.append(len(reps))
            reps.append(ans)
    return cluster_of


def _entropy(cluster_indices: list[int]) -> float:
    """Natural-log entropy of the cluster-size distribution."""
    total = len(cluster_indices)
    if total == 0:
        return 0.0
    counts = Counter(cluster_indices)
    return -sum((c / total) * math.log(c / total) for c in counts.values())


def cluster_generations(
    ctx: PipelineContext,
    gens: pd.DataFrame,
    output_name: str,
) -> list[str]:
    """Cluster sampled answers in `gens` by bidirectional entailment; save SE."""
    sampled = gens[gens["kind"] == "sample"]
    n_questions = sampled["sample_id"].nunique()
    log.info(
        "clustering %d sampled answers across %d questions via NLI (%s) -> %s",
        len(sampled), n_questions, ctx.cfg.nli.model, output_name,
    )

    rows: list[dict] = []
    for sample_id, group in sampled.groupby("sample_id"):
        answers = group.sort_values("gen_idx")["text"].tolist()
        cluster_ids = _cluster_by_entailment(answers, ctx.nli)
        rows.append(
            {
                "sample_id": sample_id,
                "semantic_entropy": _entropy(cluster_ids),
                "n_clusters": len(set(cluster_ids)),
                "n_samples": len(answers),
            }
        )
    ctx.store.save_parquet(output_name, pd.DataFrame(rows))
    return [output_name]


def run(ctx: PipelineContext) -> list[str]:
    gens = ctx.store.load_parquet(INPUT)
    return cluster_generations(ctx, gens, OUTPUT)
