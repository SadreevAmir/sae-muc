"""Parse --str_process_layers (same logic as src.utils.process_layers_to_process)."""


def process_layers_to_process(layers_to_process):
    if not layers_to_process:
        return []
    if isinstance(layers_to_process, str) and "range" in layers_to_process:
        return sorted(list(eval(layers_to_process)))  # noqa: S307
    if len(layers_to_process) == 1 and "range" in layers_to_process[0]:
        return sorted(list(eval(layers_to_process[0])))  # noqa: S307
    return sorted([int(x) for x in layers_to_process])
