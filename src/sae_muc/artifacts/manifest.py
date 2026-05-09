"""Stage-level manifests.

One JSON file per pipeline stage lives under `<run_dir>/_manifests/<stage>.json`.
A stage is considered complete (and skipped on re-run) iff its manifest
exists AND all declared output paths exist. There is no per-sample cache:
if a stage fails mid-way, it is re-run from scratch.

Users bypass the cache with --force-stage / --force-all on the CLI.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StageManifest:
    def __init__(self, run_dir: Path, stage: str) -> None:
        self.run_dir = Path(run_dir)
        self.stage = stage
        self.path = self.run_dir / "_manifests" / f"{stage}.json"

    def exists(self) -> bool:
        return self.path.exists()

    def read(self) -> dict[str, Any]:
        return json.loads(self.path.read_text())

    def write(self, outputs: list[str], extra: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "stage": self.stage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "outputs": outputs,
        }
        if extra:
            payload.update(extra)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2))

    def should_skip(self) -> bool:
        """Skip if the manifest exists AND every declared output is still present."""
        if not self.exists():
            return False
        try:
            data = self.read()
        except (OSError, ValueError):
            return False
        for rel in data.get("outputs", []):
            if not (self.run_dir / rel).exists():
                return False
        return True
