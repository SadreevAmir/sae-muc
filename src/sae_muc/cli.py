"""Command-line entry point for the sae-muc pipeline.

Real stages are added as the pipeline is implemented. For now this is a
minimal stub so that `sae-muc --help` works and scaffolding imports resolve.
"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def _bootstrap() -> None:
    """Load .env from the current working directory before any command runs."""
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(env_path)


@app.command()
def version() -> None:
    """Print the package version."""
    from sae_muc import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
