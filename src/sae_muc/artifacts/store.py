"""ArtifactStore — owns a single run's directory and knows how to read/write named outputs.

Artefacts are always referenced by a name relative to the run directory
(e.g. `"generations.parquet"`, `"hidden_states/layer_15.safetensors"`), so
stages never build paths themselves.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from sae_muc.config import ExperimentConfig


class ArtifactStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "_manifests").mkdir(exist_ok=True)

    def path(self, name: str) -> Path:
        return self.run_dir / name

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    # --- parquet ---------------------------------------------------------
    def save_parquet(self, name: str, df: pd.DataFrame) -> Path:
        p = self.path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
        return p

    def load_parquet(self, name: str) -> pd.DataFrame:
        import pandas as pd

        return pd.read_parquet(self.path(name))

    # --- safetensors -----------------------------------------------------
    def save_safetensors(self, name: str, tensors: dict[str, Any]) -> Path:
        from safetensors.torch import save_file

        p = self.path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(p))
        return p

    def load_safetensors(self, name: str) -> dict[str, Any]:
        from safetensors.torch import load_file

        return load_file(str(self.path(name)))

    # --- json ------------------------------------------------------------
    def save_json(self, name: str, obj: Any) -> Path:
        p = self.path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False))
        return p

    def load_json(self, name: str) -> Any:
        return json.loads(self.path(name).read_text())


def make_run(cfg: ExperimentConfig, run_id: str | None = None) -> tuple[str, ArtifactStore]:
    """Resolve run_id and prepare the run directory; freeze config on first write."""
    if run_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_id = cfg.build_run_id(timestamp)
    run_dir = cfg.data_root / "runs" / run_id
    store = ArtifactStore(run_dir)
    if not store.exists("config.resolved.json"):
        store.save_json("config.resolved.json", cfg.model_dump(mode="json"))
    return run_id, store
