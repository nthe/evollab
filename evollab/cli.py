import asyncio
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import click
import halo

from . import commands, models, prompts


def in_asyncio_run(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


spinner_settings: dict[str, Any] = {
    "text": "Loading...",
    "color": "yellow",
    "spinner": "toggle10",
}


@click.group()
@click.option(
    "-m",
    "--model",
    help="Large language model to use",
    default="openai/gpt-4o-mini",
)
@click.option(
    "-f",
    "--output_format",
    help="Expected output format",
    default="text",
)
@click.option(
    "-t",
    "--temperature",
    help="Diversity of generated output",
    default=0.3,
)
@click.option(
    "--top_p",
    help="Probability of less probable words in output",
    default=0.95,
)
@click.option(
    "--seed",
    help="Reuse of seed helps with consistency of output",
    default=47,
)
@click.option(
    "--n",
    help="Number of generations to produce",
    default=1,
)
@click.option(
    "--silent",
    help="Display spinner during generation process",
    is_flag=True,
    default=False,
)
@click.pass_context
def cli(ctx, model, output_format, temperature, top_p, seed, n, silent):
    ctx.ensure_object(dict)
    ctx.obj["silent"] = silent
    ctx.obj["args"] = models.LLMArgs(
        model=model,
        output_format=output_format,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        n=n,
    )


async def run_simple_task(ctx: click.Context, task: Callable, text: str, *args) -> Any:
    llm_args = ctx.obj.get("args", models.LLMArgs.default())
    silent = ctx.obj.get("silent", True)
    with halo.Halo(**spinner_settings, enabled=not silent):
        result = await task(text, *args, args=llm_args)
    return result


def parse_text_arg(text: str | None = None) -> str:
    if text is None:
        stdin = click.get_text_stream("stdin")
        if not stdin.isatty():
            text = stdin.read()
    if text is None:
        raise click.MissingParameter(
            message="Expected as argument or at stdin.",
            param_hint="'TEXT'",
            param_type="argument",
        )
    return text


@cli.command()
@click.pass_context
@click.argument("text", required=False)
@in_asyncio_run
async def augment(ctx: click.Context, text: str | None = None) -> None:
    """Augment, by filling missing info or entities, to provided text."""
    result = await run_simple_task(
        ctx,
        commands.augment,
        parse_text_arg(text),
    )
    click.echo(result)


@cli.command()
@click.pass_context
@click.argument("text", required=False)
@click.option("-c", "--classes", multiple=True)
@in_asyncio_run
async def classify(
    ctx: click.Context,
    text: str | None = None,
    classes: tuple[str] | None = None,
) -> None:
    """Classify a provided text."""
    result = await run_simple_task(
        ctx,
        commands.classify,
        parse_text_arg(text),
        classes,
    )
    click.echo(result)


@cli.command()
@click.pass_context
@click.argument("text", required=False)
@in_asyncio_run
async def derive(ctx: click.Context, text: str | None = None) -> None:
    """Derive an instruction from a provided text."""
    result = await run_simple_task(
        ctx,
        commands.derive,
        parse_text_arg(text),
    )
    click.echo(result)


@cli.command()
@click.pass_context
@click.argument("text", required=False)
@in_asyncio_run
async def answer(ctx: click.Context, text: str | None) -> None:
    """Answer a question from a provided text."""
    result = await run_simple_task(
        ctx,
        commands.answer,
        parse_text_arg(text),
    )
    click.echo(result)


@cli.command()
@click.pass_context
@click.argument("text", required=False)
@in_asyncio_run
async def evolve(ctx: click.Context, text: str | None = None):
    """Evolve an instruction using a method."""
    result = await run_simple_task(
        ctx,
        commands.evolve,
        parse_text_arg(text),
    )
    for step in result.steps:
        click.echo(step)


if __name__ == "__main__":
    cli()
