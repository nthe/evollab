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


# def load_instructions(dataset_path) -> list[str]:
#     # Path("./evollab/alpaca_eval.json")
#     dataset = json.loads(dataset_path.read_text())
#     dataset = [d for d in dataset if d["dataset"] == "oasst"]
#     dataset = [d["instruction"] for d in dataset]
#     logger.info(f"loaded {len(dataset)} instructions")
#     return dataset
#     # from datasets import load_dataset


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


async def run_simple_task(ctx: click.Context, task: Callable, text: str) -> None:
    args = ctx.obj.get("args", models.LLMArgs.default())
    silent = ctx.obj.get("silent", True)
    with halo.Halo(**spinner_settings, enabled=not silent):
        result = await task(text, args=args)
    print(result)


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
    await run_simple_task(
        ctx,
        commands.augment,
        parse_text_arg(text),
    )


@cli.command()
@click.pass_context
@click.argument("text", required=False)
@in_asyncio_run
async def derive(ctx: click.Context, text: str | None = None) -> None:
    """Derive an instruction from a provided text."""
    await run_simple_task(
        ctx,
        commands.derive,
        parse_text_arg(text),
    )


@cli.command()
@click.pass_context
@click.argument("text", required=False)
@in_asyncio_run
async def answer(ctx: click.Context, text: str | None) -> None:
    """Answer a question from a provided text."""
    await run_simple_task(
        ctx,
        commands.answer,
        parse_text_arg(text),
    )


@cli.command()
@click.pass_context
@click.argument("text", required=False)
@click.option("--method", default="")
@click.option("--steps", default=1)
@in_asyncio_run
async def evolve(
    ctx: click.Context,
    text: str | None = None,
    method: str = "",
    steps: int = 1,
):
    """Evolve an instruction using a method."""
    if not method:
        method = prompts.initial_method

    elif Path(method).exists() and Path(method).is_file():
        method = Path(method).read_text().strip()

    silent = ctx.obj.get("silent", True)
    with halo.Halo(**spinner_settings, enabled=not silent):
        trajectory = await commands.evolve(
            method=models.Method(method),
            instruction=parse_text_arg(text),
            steps=steps,
        )

    for step in trajectory.steps:
        print(step)


if __name__ == "__main__":
    cli()
