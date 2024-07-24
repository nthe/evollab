import json
import re
from typing import Any, AsyncGenerator, Iterable

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from . import prompts
from .models import (
    LLMArgs,
    Method,
    OutputFormat,
    Template,
    Trajectory,
)


async def autochain(
    messages: Iterable[ChatCompletionMessageParam],
    *,
    model: str,
    output_format: OutputFormat = "text",
    **model_kwargs,
) -> AsyncGenerator[Any, None]:
    result = await AsyncOpenAI().chat.completions.create(
        messages=messages,
        model=model,
        **model_kwargs,
    )
    for choice in result.choices:
        content = choice.message.content
        if output_format == "json":
            # TODO: implement proper json parsing
            yield json.loads(content)
        yield content


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


async def evolve(
    instruction: str,
    steps: int = 1,
    method: Method = Method(prompts.initial_method),
    args: LLMArgs = LLMArgs.default(),
) -> Trajectory:
    """Evolve instruction multiple times over a method.

    Args:
        method (Method): Initial method to evolve the instruction.
        instruction (str): Instruction to evolve.
        steps (int, optional): Number of evolution steps. Defaults to 1.

    Returns:
        Trajectory: Evolution trajectory of the instruction.

    Raises:
        ValueError: If unexpected steps headers are present.
    """
    trajectory = Trajectory(
        method=method,
        instruction=instruction,
    )

    # seq evolution of the instruction
    # the `n` parameter doesn't apply here
    for _ in range(steps):
        async for instr in autochain(
            messages=prompts.evolve.format(
                instruction=instruction,
                method=method,
            ),
            **args.__dict__,
        ):
            # post-process instruction if steps headers are present
            if steps_headers := extract_steps_headers(instr):
                raise ValueError(f"Unexpected steps headers: {steps_headers}")

            # TODO: keep only finally rewritten instruction

            trajectory.add(instr)

    return trajectory


async def augment(text: str, args: LLMArgs) -> str:
    """Augment a text.

    Args:
        text (str): Text to augment.
        args (LLMArgs): Language model arguments

    Returns:
        str: Augmented text.
    """
    async for answer in autochain(
        messages=prompts.augment.format(text=text),
        **args.__dict__,
    ):
        return answer
    return ""


async def derive(text: str, args: LLMArgs) -> str:
    """Derive an instruction from a text.

    Args:
        text (str): Text to derive an instruction from.
        args (LLMArgs): Language model arguments

    Returns:
        str: Derived instruction.
    """
    async for answer in autochain(
        messages=prompts.derive.format(text=text),
        **args.__dict__,
    ):
        return answer
    return ""


async def answer(text: str, args: LLMArgs) -> str:
    """Answer a question from a text.

    Args:
        text (str): Text to answer a question from.
        args (LLMArgs): Language model arguments

    Returns:
        str: Answer to the question.
    """
    async for answer in autochain(
        messages=Template(user=text).format(text=text),
        **args.__dict__,
    ):
        return answer
    return ""


async def classify(text: str, classes: list[str], args: LLMArgs) -> list[str]:
    """Classify a text.

    Args:
        text (str): Text to classify.
        classes (list[str]): List of classes to classify the text.
        args (LLMArgs): Language model arguments

    Returns:
        list[str]: List of classes the text belongs to.
    """
    async for answer in autochain(
        messages=prompts.classify.format(
            text=text,
            classes="\n".join(classes),
        ),
        **args.__dict__,
    ):
        return answer
    return [""]
