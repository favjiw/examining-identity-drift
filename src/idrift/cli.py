"""CLI (PRD FR-6). Tahap 4: run, estimate."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from idrift.config import load_config
from idrift.runner import estimate_calls, run_batch

app = typer.Typer(help="idrift: identity drift replication runner")
console = Console()


@app.command()
def version() -> None:
    """Print package version."""
    from idrift import __version__

    typer.echo(__version__)


@app.command(name="check-config")
def check_config(config: str = typer.Option(..., "--config", help="Path YAML config")) -> None:
    """Validate config file parses."""
    cfg = load_config(config)
    typer.echo(f"OK: batch={cfg.batch_name} runs={cfg.n_runs}")


@app.command()
def estimate(config: str = typer.Option(..., "--config", help="Path YAML config")) -> None:
    """Show API call estimates."""
    cfg = load_config(config)
    est = estimate_calls(cfg)
    console.print(f"[bold]Estimasi panggilan API[/bold] — batch [cyan]{cfg.batch_name}[/cyan]")
    console.print(f"  n_runs:              {est['n_runs']}")
    console.print(f"  conversation:        {est['conversation']} (= n_runs × 36 × 2)")
    console.print(f"  questionnaire:       {est['questionnaire']} (= n_runs × snapshots × n_q × repeats × agents)")
    console.print(f"    - snapshots:       {cfg.questionnaire.snapshots}")
    console.print(f"    - n_questionnaires:{est['n_questionnaires']}  (15 jika include=all)")
    console.print(f"    - repeats:         {est['repeats']}")
    console.print(f"    - measure_agents:  {est['measure_agents']}")
    console.print(f"  [bold]total:             {est['total']}[/bold]")


@app.command()
def run(
    config: str = typer.Option(..., "--config", help="Path YAML config"),
    quick: bool = typer.Option(False, "--quick", help="1 run smoke test"),
    resume: Optional[str] = typer.Option(None, "--resume", help="Dir .partial.json untuk resume"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Hapus logs batch lama"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Lewati konfirmasi batch besar"),
) -> None:
    """Run batch + auto-save log (FR-3)."""
    cfg = load_config(config)
    if quick:
        # Jangan mutasi file; effective n_runs=1 via run_batch quick flag
        pass

    est = estimate_calls(cfg)
    effective_total = 36 * 2 if quick else est["total"]
    if effective_total > 2000 and not yes:
        console.print(f"[yellow]Estimasi {effective_total} panggilan API (>2000).[/yellow]")
        confirm = typer.confirm("Lanjut?", default=False)
        if not confirm:
            console.print("Dibatalkan.")
            raise typer.Abort()

    n_runs = 1 if quick else cfg.n_runs
    console.print(f"[bold]Run batch[/bold] [cyan]{cfg.batch_name}[/cyan] — {n_runs} run(s)")

    # Progress per run
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        tasks: dict[int, int] = {}

        def on_progress(run_idx: int, theme_id: int, total: int) -> None:
            if run_idx not in tasks:
                tasks[run_idx] = progress.add_task(f"run {run_idx + 1}/{n_runs}", total=total)
            progress.update(tasks[run_idx], completed=theme_id)

        run_ids = run_batch(cfg, quick=quick, resume_dir=resume, overwrite=overwrite, on_progress=on_progress)

    console.print(f"[green]Selesai.[/green] {len(run_ids)} run → logs/{cfg.batch_name}/")
    for rid in run_ids:
        console.print(f"  - {rid}")


if __name__ == "__main__":
    app()
