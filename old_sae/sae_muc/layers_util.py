"""Parse --str_process_layers (same logic as src.utils.process_layers_to_process)."""

import re

_RANGE_RE = re.compile(r"range\(\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d+)\s*)?\)")


def _safe_parse_range(s: str) -> list[int] | None:
    m = _RANGE_RE.search(s)
    if not m:
        return None
    start, stop = int(m.group(1)), int(m.group(2))
    step = int(m.group(3)) if m.group(3) else 1
    return list(range(start, stop, step))


def parse_layers_str(s: str) -> list[int]:
    """
    Список HF-индексов слоёв из строки:
    - ``range(15,32)`` (как в --str_process_layers)
    - ``15,23`` или ``15, 23, 7``
    """
    t = (s or "").strip()
    if not t:
        return []
    parsed = _safe_parse_range(t)
    if parsed is not None:
        return sorted(set(parsed))
    parts = [p.strip() for p in t.split(",") if p.strip()]
    return sorted({int(p) for p in parts})


def process_layers_to_process(layers_to_process):
    if not layers_to_process:
        return []
    if isinstance(layers_to_process, str):
        return parse_layers_str(layers_to_process)
    if isinstance(layers_to_process, (list, tuple)):
        if len(layers_to_process) == 1 and isinstance(layers_to_process[0], str):
            return parse_layers_str(layers_to_process[0])
        return sorted([int(x) for x in layers_to_process])
    return []
