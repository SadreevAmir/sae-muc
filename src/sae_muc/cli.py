"""Command-line entry point for the sae-muc pipeline.

Real stages are added as the pipeline is implemented. For now this is a
minimal stub so that `sae-muc --help` works and scaffolding imports resolve.
"""

from __future__ import annotations

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def version() -> None:
    """Print the package version."""
    from sae_muc import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
