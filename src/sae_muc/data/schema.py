"""Shared dataset schema."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Sample(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_id: str
    question: str
    gold_answers: list[str]
    # "main" feeds VUF/SAE/detector fitting AND evaluation; "heldout" is a
    # disjoint set carried through generation/judging but excluded from every
    # fit, used only for a contamination-free evaluation of the intervention.
    split: str = "main"
