from __future__ import annotations

import logging

from sae_muc.logging_setup import add_file_handler, configure


def test_add_file_handler_writes_log_lines(tmp_path):
    """File handler captures log records emitted while it's installed."""
    configure("INFO")  # baseline: stderr handler only
    log_path = tmp_path / "run.log"
    fh = add_file_handler(log_path)
    try:
        logging.getLogger("sae_muc.test").info("hello from sae_muc.test")
        logging.getLogger("sae_muc.pipeline.generate").info("==> generate")
    finally:
        logging.getLogger().removeHandler(fh)
        fh.close()
    text = log_path.read_text()
    assert "hello from sae_muc.test" in text
    assert "==> generate" in text


def test_add_file_handler_appends_on_resume(tmp_path):
    """Re-opening the same path keeps prior content (resume semantics)."""
    configure("INFO")
    log_path = tmp_path / "run.log"

    fh1 = add_file_handler(log_path)
    logging.getLogger("sae_muc.cli").info("first run")
    logging.getLogger().removeHandler(fh1)
    fh1.close()

    fh2 = add_file_handler(log_path)
    logging.getLogger("sae_muc.cli").info("resumed run")
    logging.getLogger().removeHandler(fh2)
    fh2.close()

    text = log_path.read_text()
    assert "first run" in text
    assert "resumed run" in text


def test_add_file_handler_no_color_codes(tmp_path):
    """File log must be plain text — no ANSI escape sequences."""
    configure("INFO")
    log_path = tmp_path / "run.log"
    fh = add_file_handler(log_path)
    try:
        logging.getLogger("sae_muc.pipeline.runner").info("==> stage")
        logging.getLogger("sae_muc.pipeline.runner").info("[ok] stage · 1.2s · output")
    finally:
        logging.getLogger().removeHandler(fh)
        fh.close()
    text = log_path.read_text()
    assert "\x1b[" not in text  # no ANSI escape introducer
