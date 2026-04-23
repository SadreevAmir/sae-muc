"""
Periodic copy of selected paths from Colab /content (or any local tree) to Google Drive.

Designed for use from a notebook thread; see colab_sae_muc.ipynb.
"""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path


def _copy_pair(src: Path, dst: Path) -> None:
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        return
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return
    for path in src.rglob("*"):
        if path.is_file():
            rel = path.relative_to(src)
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)


def sync_pairs_now(pairs: list[tuple[str | Path, str | Path]]) -> None:
    for s, d in pairs:
        _copy_pair(Path(s), Path(d))


class PeriodicDriveBackup:
    """
    Run sync_pairs_now immediately, then every interval_sec until stop().

    Usage:
        backup = PeriodicDriveBackup([("/content/out.jsonl", "/content/drive/...")], interval_sec=600)
        backup.start()
        ...
        backup.stop()
    """

    def __init__(
        self,
        pairs: list[tuple[str | Path, str | Path]],
        interval_sec: float = 600.0,
    ) -> None:
        self.pairs = pairs
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_sec):
            try:
                sync_pairs_now(self.pairs)
            except Exception as e:  # noqa: BLE001 — keep thread alive in Colab
                print(f"[PeriodicDriveBackup] sync error: {e}")

    def start(self) -> None:
        sync_pairs_now(self.pairs)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
