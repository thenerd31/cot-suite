"""Command-line entrypoint: `cotsuite score` and `cotsuite eval`."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from cotsuite.core.trajectory import Trajectory

app = typer.Typer(help="cot-suite: CoT monitorability + faithfulness evaluation")
console = Console()


@app.command()
def score(
    trajectory_path: Annotated[Path, typer.Argument(help="JSONL file of Trajectory objects")],
    metrics: Annotated[
        str,
        typer.Option("--metrics", "-m", help="Comma-separated metric names"),
    ] = "legibility,coverage",
    autorater: Annotated[
        str,
        typer.Option("--autorater", "-a"),
    ] = "google/gemini-2.5-pro",
    runs: Annotated[int, typer.Option("--runs", "-n")] = 5,
) -> None:
    """Score one or more trajectories from a JSONL file."""
    from cotsuite import metrics as _  # noqa: F401 — ensure registrations
    from cotsuite import score_trajectory

    if not trajectory_path.exists():
        console.print(f"[red]file not found:[/red] {trajectory_path}", style="bold")
        raise typer.Exit(code=2)

    metric_list = [m.strip() for m in metrics.split(",") if m.strip()]
    table = Table(title=f"cot-suite score — {trajectory_path.name}")
    table.add_column("sample")
    for m in metric_list:
        table.add_column(m)

    with trajectory_path.open() as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            traj = Trajectory.model_validate_json(line)
            result = score_trajectory(
                traj,
                metrics=metric_list,
                autorater=autorater,
                runs=runs,
            )
            row = [str(i)]
            for m in metric_list:
                v = result.metrics.get(m)
                row.append(
                    f"{v.value:.2f} ± {v.stddev:.2f}" if v is not None else "—",
                )
            table.add_row(*row)

    console.print(table)


@app.command(name="eval")
def eval_(
    dataset: Annotated[str, typer.Argument(help="Dataset name, e.g. faithbench, gpqa_diamond")],
    model: Annotated[str, typer.Option("--model")] = "anthropic/claude-opus-4-5",
    tests: Annotated[str, typer.Option("--tests")] = "legibility,coverage",
    subset: Annotated[str, typer.Option("--subset")] = "full",
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("results"),
    models: Annotated[
        str | None,
        typer.Option("--models", help="Comma-separated list overrides --model"),
    ] = None,
) -> None:
    """Run a dataset-wide benchmark pass (stub — implemented by Month 3)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": dataset,
        "model": models.split(",") if models else [model],
        "tests": tests.split(","),
        "subset": subset,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    console.print(f"[yellow]eval stub[/yellow]: config written to {output_dir}/config.json")
    console.print("[dim]full benchmark runner lands in v0.4 (Month 3).[/dim]")


@app.command()
def prompts() -> None:
    """List bundled autorater prompt versions with their SHA256 hashes."""
    from importlib import resources

    from cotsuite.autoraters.legibility_coverage import LegibilityCoveragePrompt

    pkg = resources.files("cotsuite.autoraters.prompts")
    table = Table(title="Bundled autorater prompts")
    table.add_column("version")
    table.add_column("sha256 (first 12)")
    for entry in pkg.iterdir():
        if entry.name.endswith(".txt"):
            version = entry.name[:-4]
            prompt = LegibilityCoveragePrompt.load(version)
            table.add_row(version, prompt.sha256[:12])
    console.print(table)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
