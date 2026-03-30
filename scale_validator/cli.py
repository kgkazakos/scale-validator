"""Click CLI for scale-validator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import click

if TYPE_CHECKING:
    from scale_validator.models import ScaleValidationResult


@click.group()
@click.version_option(package_name="scale-validator")
def cli() -> None:
    """Scale validation — Cronbach's alpha, item analysis, and factor retention."""


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--items", "-i", required=True, help="Comma-separated item column names.")
@click.option("--scale-name", default="scale", show_default=True)
@click.option("--llm-backend", type=click.Choice(["openai", "anthropic", "gemini"]), default=None)
@click.option("--output", "-o", default=None, help="Write JSON report to this path.")
@click.option("--quiet", "-q", is_flag=True)
def validate(
    input_file: str,
    items: str,
    scale_name: str,
    llm_backend: Optional[str],
    output: Optional[str],
    quiet: bool,
) -> None:
    """Validate a Likert scale from a CSV file."""
    import pandas as pd

    from scale_validator.engine import validate_scale

    df = pd.read_csv(input_file)
    item_list = [i.strip() for i in items.split(",")]

    result = validate_scale(df, items=item_list, scale_name=scale_name, llm_backend=llm_backend)

    if not quiet:
        _print_result(result)

    if output:
        Path(output).write_text(json.dumps(result.to_dict(), indent=2))
        if not quiet:
            click.echo(f"✓ Report written to {output}")


def _print_result(result: "ScaleValidationResult") -> None:
    try:
        from rich import box
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()
        verdict_colours = {
            "excellent": "green", "good": "green", "adequate": "yellow",
            "questionable": "yellow", "poor": "red", "unacceptable": "red",
        }
        vc = verdict_colours.get(result.overall_verdict, "white")

        console.print()
        console.print(Panel(
            f"α = [bold]{result.cronbach_alpha:.3f}[/] ({result.alpha_interpretation})  "
            f"| Verdict: [{vc}]{result.overall_verdict.upper()}[/]  "
            f"| {result.n_items} items · {result.n_respondents} respondents  "
            f"| Mean IIC = {result.inter_item_correlation_mean:.3f}",
            title=f"[bold]scale-validator[/] · {result.scale_name}",
            expand=False,
        ))

        t = Table(box=box.SIMPLE_HEAVY, header_style="bold")
        t.add_column("Item")
        t.add_column("Mean")
        t.add_column("SD")
        t.add_column("Item-Total r")
        t.add_column("α if deleted")
        t.add_column("Flag")

        flag_colours = {"ok": "green", "borderline": "yellow", "weak": "red"}
        for s in result.item_stats:
            fc = flag_colours.get(s.flag, "white")
            t.add_row(
                s.item,
                f"{s.mean:.2f}",
                f"{s.std:.2f}",
                f"{s.item_total_correlation:.3f}",
                f"{s.alpha_if_deleted:.3f}",
                f"[{fc}]{s.flag}[/]",
            )
        console.print(t)

        if result.factor_retention:
            fr = result.factor_retention
            console.print(
                f"  Factor retention → Kaiser: [bold]{fr.n_factors_kaiser}[/] factor(s)  "
                f"| 95% variance: [bold]{fr.n_factors_95pct}[/] factor(s)"
            )

        if result.weak_items:
            console.print(f"  [yellow]Items to review: {', '.join(result.weak_items)}[/]")

        if result.narrative:
            console.print(Panel(result.narrative, title="Narrative", expand=False))
        console.print()

    except ImportError:
        click.echo(f"\nscale-validator: {result.scale_name}")
        click.echo(f"  α = {result.cronbach_alpha:.3f} ({result.alpha_interpretation}) — {result.overall_verdict}")
        for s in result.item_stats:
            click.echo(f"  {s.item}: r={s.item_total_correlation:.3f} [{s.flag}]")
