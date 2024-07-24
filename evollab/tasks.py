import asyncio
import re
from random import sample

import click

from . import commands, prompts
from .models import (
    EvolReport,
    Feedback,
    LLMArgs,
    Method,
    Trajectory,
)


total_evol_steps: int = 3
total_optm_steps: int = 3
development_set_size: int = 10
mini_batch_size: int = 5


def extract_steps(text: str) -> list[dict[str, str]]:
    """Extract steps from a text.

    Args:
        text (str): Text to extract steps from.

    Returns:
        list[dict[str, str]]: List of steps.
    """
    # TODO: implementation
    return []


def extract_steps_headers(text: str) -> list[str]:
    """Extract steps headers from a text.

    Args:
        text (str): Text to extract steps headers from.

    Returns:
        list[str]: List of steps headers."""
    return re.findall(r"^Step\s*\d*\s\#[\w\s]*\#", text, re.MULTILINE)


async def concurrently(func, *args):
    """Run multiple async functions concurrently.

    Args:
        func: Async function to run.
        args: Arguments to pass to the async function.

    Returns:
        list: List of results from the async functions.
    """
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(func(*a)) for a in zip(*args)]
    return [t.result() for t in tasks]


async def analyze(trajectory: Trajectory) -> Feedback:
    """Analyze the evolution trajectory of an instruction.

    Args:
        trajectory (Trajectory): Evolution trajectory of an instruction.

    Returns:
        Feedback: List of feedbacks from the analysis.
    """
    stages = "\n\n".join(
        f"Stage {i}: {e.strip()}" for i, e in enumerate(trajectory.evolution)
    )

    async for analysis in commands.autochain(
        model="anthropic/claude-3.5-sonnet",
        messages=prompts.analyze.format(trajectory=stages),
        output_format="json",
        temperature=0.6,
        top_p=0.95,
        seed=47,
        n=1,
    ):
        # return first feedback (`n` is 1 anyway)
        return Feedback([a["constraint"] for a in analysis])

    # no analysis == no feedback
    return Feedback([])


async def optimize(method: Method, feedback: Feedback) -> Method:
    """Optimize a method based on feedback.

    Args:
        method (Method): Method to optimize.
        feedback (Feedback): List of feedbacks from the analysis.

    Returns:
        Method: Optimized method.
    """
    feedback_str = "\n\n".join(feedback)
    rendered_method = method.format(instruction="")

    async for optm_method in commands.autochain(
        model="anthropic/claude-3.5-sonnet",
        messages=prompts.optimize.format(
            feedback=feedback_str,
            method=rendered_method,
        ),
        output_format="text",
        temperature=0.6,
        top_p=0.95,
        seed=47,
        n=1,
    ):
        return Method(optm_method)

    # no optimized method? return the same method
    return method


def evaluate_answer(answer: str) -> bool:
    """Evaluate if an answer is valid.

    Args:
        answer (str): Answer to evaluate.

    Returns:
        bool: True if the answer is valid, False otherwise."""
    a: str = answer.strip().lower()

    # check if is a question
    if a.startswith(("understood", "thank you", "sure")) and a.endswith("?"):
        return False

    # check if asks more information
    if "please provide" in a:
        return False

    return True


async def evaluate_method(
    method: Method,
    instructions: list[str],
    args: LLMArgs,
) -> float:
    """Evaluate a method over a set of instructions.

    Args:
        method (Method): Method to evaluate.
        instructions (list[str]): Instructions to evaluate.

    Returns:
        float: Error rate of the method over the instructions.
    """
    click.echo(f"evaluating {method} over {len(instructions)} instructions")
    num_failures: int = 0
    for i, instr in enumerate(instructions):
        click.echo(f"evaluating over instruction {i + 1}/{len(instructions)}")
        trajectory = await commands.evolve(method, instr, steps=1)
        for evol_instr in trajectory.evolution:
            response = await commands.answer(evol_instr, args)
            # answer = await llm.evol.ainvoke(evol_instr)
            is_valid = evaluate_answer(str(response))
            if not is_valid:
                num_failures += 1

    error: float = num_failures / len(instructions)
    return error


async def evolve_batch(
    method: Method,
    instructions: list[str],
) -> tuple[list[Method], list[EvolReport]]:
    """Evolve a method over a batch of instructions.

    Args:
        method (Method): Initial method to evolve.
        instructions (list[str]): Instructions to evolve over.

    Returns:
        tuple[list[Method], list[EvolReport]]: List of evolved methods and their reports.
    """
    click.echo(f"evolving {method} over {len(instructions)} instructions")
    feedbacks = Feedback([])
    reports: list[EvolReport] = []
    for i, instr in enumerate(instructions):
        click.echo(f"evolving over instruction {i + 1}/{len(instructions)}")
        trajectory = await commands.evolve(method, instr, steps=total_evol_steps)
        click.echo(f"analyzing over instruction {i + 1}/{len(instructions)}")
        feedback = await analyze(trajectory)
        report = EvolReport(trajectory, feedback)
        reports.append(report)
        feedbacks.extend(feedback)

    if not feedbacks:
        return [method], reports

    new_methods: list[Method] = []
    for i in range(total_optm_steps):
        click.echo(f"optmizing method {i + 1}/{total_optm_steps}")
        new_method = await optimize(method, feedbacks)
        new_methods.append(new_method)

    return new_methods, reports


async def evolve_method(instructions: list[str]) -> Method:
    """Evolve a dataset of instructions.

    Args:
        instructions (list[str]): Instructions to evolve.

    Returns:
        Method: Best method evolved over the instructions.
    """
    # split development set and mini batches from src instructions
    dev_set: list[str] = sample(
        instructions,
        k=development_set_size,
    )

    mini_batches: list[list[str]] = [
        dev_set[i : i + mini_batch_size]
        for i in range(0, len(dev_set), mini_batch_size)
    ]

    init_method: Method = Method(prompts.initial_method)
    evol_methods: list[Method] = [init_method]
    reports_table: dict[Method, list[EvolReport]] = {}

    # evolve alternative methods over mini batches
    methods_and_reports = await concurrently(
        evolve_batch,
        [init_method] * len(mini_batches),
        mini_batches,
    )
    for new_methods, reports in methods_and_reports:
        for new_method in new_methods:
            if not new_method.is_equal_to(init_method):
                evol_methods.append(new_method)
                reports_table[new_method] = reports

    # calc errors over development set
    errors: list[float] = await concurrently(
        evaluate_method,
        evol_methods,
        [dev_set] * len(evol_methods),
    )

    # smallest score is the best
    best_method: Method = evol_methods[errors.index(min(errors))]

    click.echo(f"errors: {errors}")

    # for i, method in enumerate(evol_methods, start=1):
    #     with open(f"./{i}-method.txt", "w") as f:
    #         f.write(str(method.data))

    # with open("./best-method.txt", "w") as f:
    #     f.write(str(best_method.data))

    return best_method
