"""Command-line entry point for the sae-muc pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from dotenv import load_dotenv

from sae_muc.logging_setup import add_file_handler as _add_file_handler
from sae_muc.logging_setup import configure as _configure_logging

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def _bootstrap() -> None:
    """Configure logging and load .env before any command runs."""
    _configure_logging("INFO")
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(env_path)


@app.command()
def version() -> None:
    """Print the package version."""
    from sae_muc import __version__

    typer.echo(__version__)


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", exists=True, help="Path to experiment YAML."),
    stage: str = typer.Option(
        "all", "--stage", "-s", help="Stage name or 'all' for the full pipeline."
    ),
    run_id: str | None = typer.Option(
        None, "--run-id", help="Reuse an existing run_id (resume into an existing directory)."
    ),
    force_all: bool = typer.Option(
        False, "--force-all", help="Ignore stage manifests; recompute every stage."
    ),
) -> None:
    """Run one or all pipeline stages for the given experiment config."""
    from sae_muc.config import load_experiment_config
    from sae_muc.pipeline import STAGES, build_context, run_all, run_stage

    cfg = load_experiment_config(config)
    rid, ctx = build_context(cfg, run_id=run_id)
    typer.echo(f"run_id: {rid}")
    typer.echo(f"run_dir: {ctx.store.run_dir}")

    # File log lives in the run dir so artefacts and logs sit together in
    # shared storage. Append-mode: resumes via --run-id accumulate history.
    # tail -f /mnt/ssd/sae-muc/runs/<run_id>/run.log from any ssh session.
    file_handler = _add_file_handler(ctx.store.run_dir / "run.log")
    log = logging.getLogger("sae_muc.cli")
    log.info("run_id: %s", rid)
    log.info("run_dir: %s", ctx.store.run_dir)
    log.info("config: %s", config)

    try:
        if stage == "all":
            run_all(ctx, force_all=force_all)
        else:
            if stage not in STAGES:
                raise typer.BadParameter(
                    f"Unknown stage {stage!r}. Known: {sorted(STAGES)}"
                )
            run_stage(ctx, stage, force=force_all)
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


if __name__ == "__main__":
    app()
