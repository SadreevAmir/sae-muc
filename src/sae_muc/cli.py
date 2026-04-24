"""Command-line entry point for the sae-muc pipeline."""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

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

    if stage == "all":
        run_all(ctx, force_all=force_all)
    else:
        if stage not in STAGES:
            raise typer.BadParameter(
                f"Unknown stage {stage!r}. Known: {sorted(STAGES)}"
            )
        run_stage(ctx, stage, force=force_all)


if __name__ == "__main__":
    app()
