"""Logging configuration for sae-muc.

Goals:
  * Stage banners from `pipeline.runner` stand out visually.
  * Third-party INFO noise (httpx, transformers download progress,
    openai retry chatter) is suppressed to WARNING by default.
  * Stderr output is colourised when attached to a TTY; plain on pipes.

Call `configure()` once at program startup (the Typer `@app.callback`
in `cli.py` does this).
"""

from __future__ import annotations

import logging
import os
import sys

_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "openai._base_client",
    "transformers",
    "transformers.modeling_utils",
    "transformers.configuration_utils",
    "transformers.tokenization_utils_base",
    "huggingface_hub",
    "filelock",
    "datasets",
    "datasets.builder",
)

_BANNER_PREFIXES = ("==>", "[ok]", "[skip]", "[fail]")


class _Formatter(logging.Formatter):
    """Compact formatter with banner highlighting.

    Regular line:  HH:MM:SS  LEVEL    name                msg
    Banner line :  colorised plain message (no timestamp/name clutter).
    """

    _RESET = "\x1b[0m"
    _DIM = "\x1b[2m"
    _CYAN = "\x1b[36m"
    _GREEN = "\x1b[32m"
    _YELLOW = "\x1b[33m"
    _RED = "\x1b[31m"

    def __init__(self, *, color: bool) -> None:
        super().__init__()
        self._color = color

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        name = record.name
        if name.startswith("sae_muc."):
            name = name[len("sae_muc."):]

        # Banner lines: strip the ts/level/name frame.
        if any(msg.startswith(p) for p in _BANNER_PREFIXES):
            if not self._color:
                return msg
            colour = {
                "==>": self._CYAN,
                "[ok]": self._GREEN,
                "[skip]": self._DIM,
                "[fail]": self._RED,
            }.get(msg.split(" ", 1)[0], "")
            return f"{colour}{msg}{self._RESET}"

        ts = self.formatTime(record, "%H:%M:%S")
        level = record.levelname
        level_col = {
            "WARNING": self._YELLOW,
            "ERROR": self._RED,
            "CRITICAL": self._RED,
        }.get(level, "")
        if self._color:
            return (
                f"{self._DIM}{ts}{self._RESET}  "
                f"{level_col}{level:<7}{self._RESET}  "
                f"{self._DIM}{name:<28}{self._RESET}  {msg}"
            )
        return f"{ts}  {level:<7}  {name:<28}  {msg}"


def configure(level: str | int = "INFO") -> None:
    """Install a single stderr handler; quiet the usual third-party loggers."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stderr)
    use_color = sys.stderr.isatty() and os.environ.get("NO_COLOR", "") == ""
    handler.setFormatter(_Formatter(color=use_color))
    root.addHandler(handler)
    root.setLevel(level)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def add_file_handler(path) -> logging.FileHandler:
    """Add a no-color FileHandler to the root logger and return it.

    Caller owns the handler — remove with `logging.getLogger().removeHandler(h)`
    and `h.close()` in a finally-block. Append-mode so resumes via --run-id
    accumulate the full run history in one file.
    """
    fh = logging.FileHandler(str(path), mode="a", encoding="utf-8")
    fh.setFormatter(_Formatter(color=False))
    logging.getLogger().addHandler(fh)
    return fh
