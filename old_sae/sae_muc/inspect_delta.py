"""
Разбор разреженного delta и ссылки на дашборды Neuronpedia (размеченные autointerp).

Neuronpedia URL: https://www.neuronpedia.org/[MODEL_ID]/[SAE_ID]/[FEATURE_INDEX]
API JSON:        https://www.neuronpedia.org/api/feature/[MODEL_ID]/[SAE_ID]/[FEATURE_INDEX]

Для релиза `llama_scope_lxr_32x` slug’и совпадают с полем `neuronpedia` в SAELens yaml; для других релизов id на сайте может отличаться от SAELens.
Его нужно взять из URL выбранного SAE на neuronpedia.org.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

import torch


def sparse_delta_entries(delta: torch.Tensor) -> list[tuple[int, float]]:
    """Индекс и значение для ненулевых координат delta [d_sae], по убыванию |value|."""
    d = delta.detach().float().cpu().flatten()
    nz = d.nonzero(as_tuple=True)[0]
    if nz.numel() == 0:
        return []
    vals = d[nz]
    pairs = list(zip(nz.tolist(), vals.tolist()))
    pairs.sort(key=lambda x: -abs(x[1]))
    return pairs


def print_sparse_delta_report(hf_layer: int, delta: torch.Tensor, limit: int = 40) -> None:
    pairs = sparse_delta_entries(delta)
    print(f"--- HF layer {hf_layer}: ненулевых латентов {len(pairs)} (топ-{min(limit, len(pairs))}) ---")
    for i, (idx, val) in enumerate(pairs[:limit]):
        print(f"  lat_{idx:6d}  coeff={val:+.6f}  |coeff|={abs(val):.6f}")


def neuronpedia_feature_urls(
    model_id: str,
    sae_id: str,
    delta: torch.Tensor,
    limit: int = 15,
) -> list[str]:
    """Готовые ссылки на страницы признаков с объяснениями (если они есть на NP)."""
    pairs = sparse_delta_entries(delta)
    base = f"https://www.neuronpedia.org/{model_id}/{sae_id}"
    return [f"{base}/{idx}" for idx, _ in pairs[:limit]]


def fetch_neuronpedia_feature_json(
    model_id: str,
    sae_id: str,
    feature_index: int,
    timeout_sec: float = 20.0,
) -> dict[str, Any] | None:
    """GET /api/feature/... — в ответе часто есть explanations / autointerp."""
    url = f"https://www.neuronpedia.org/api/feature/{model_id}/{sae_id}/{feature_index}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "sae-muc-inspect/1.0"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw)
    except Exception:
        return None


def print_top_explanations(
    model_id: str,
    sae_id: str,
    delta: torch.Tensor,
    limit: int = 8,
) -> None:
    """Печатает кратко первое explanation из API, если оно есть (нужен сетевой доступ)."""
    pairs = sparse_delta_entries(delta)
    for idx, val in pairs[:limit]:
        data = fetch_neuronpedia_feature_json(model_id, sae_id, idx)
        expl = ""
        if isinstance(data, dict):
            ex = data.get("explanations") or data.get("explanation")
            if isinstance(ex, list) and ex:
                expl = str(ex[0].get("text", ex[0]) if isinstance(ex[0], dict) else ex[0])[:200]
            elif isinstance(ex, str):
                expl = ex[:200]
        line = f"  [{idx}] coeff={val:+.4f}"
        if expl:
            line += f"  →  {expl}"
        print(line)
