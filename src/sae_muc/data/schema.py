"""Shared dataset schema."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Sample(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_id: str
    question: str
    gold_answers: list[str]
