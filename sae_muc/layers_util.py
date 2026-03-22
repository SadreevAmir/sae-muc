"""Parse --str_process_layers (same logic as src.utils.process_layers_to_process)."""


def parse_layers_str(s: str) -> list[int]:
    """
    Список HF-индексов слоёв из строки:
    - ``range(15,32)`` (как в --str_process_layers)
    - ``15,23`` или ``15, 23, 7``
    """
    t = (s or "").strip()
    if not t:
        return []
    if "range" in t:
        return sorted(set(eval(t)))  # noqa: S307
    parts = [p.strip() for p in t.split(",") if p.strip()]
    return sorted({int(p) for p in parts})


def process_layers_to_process(layers_to_process):
    if not layers_to_process:
        return []
    if isinstance(layers_to_process, str) and "range" in layers_to_process:
        return sorted(list(eval(layers_to_process)))  # noqa: S307
    if len(layers_to_process) == 1 and "range" in layers_to_process[0]:
        return sorted(list(eval(layers_to_process[0])))  # noqa: S307
    return sorted([int(x) for x in layers_to_process])
