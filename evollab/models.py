from __future__ import annotations

from collections import UserString
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Literal, NewType

from openai.types.chat import ChatCompletionMessageParam


class Method(UserString):
    """Method to evolve an instruction over."""

    def __str__(self) -> str:
        snippet = self.data[:25]
        if len(self.data) > 25:
            snippet += "..."
        return f"Method({len(self.data)}, {snippet.strip()})"

    def is_equal_to(self, b: Method) -> bool:
        """Check if two methods are equal."""
        return self.data.strip().lower() == b.data.strip().lower()


Feedback = NewType("Feedback", list[str])
"""Feedback from the analysis of an evolution trajectory."""


@dataclass
class Trajectory:
    """Evolution trajectory of an instruction over a method."""

    method: Method
    instruction: str
    steps: list[str] = field(default_factory=list)

    @property
    def evolution(self) -> list[str]:
        """Evolution trajectory of the instruction."""
        return [self.instruction] + self.steps

    def add(self, step: str) -> None:
        """Add a step to the evolution trajectory."""
        self.steps.append(step)


@dataclass
class EvolReport:
    """Evolution report of an instruction."""

    trajectory: Trajectory
    feedback: Feedback


default_template_system_prompt: str = (
    "You are a helpful assistant. "
    "Current date is {current_date} "
    "(no need to mention it)."
)


@dataclass
class Template:
    """Template for a chat completion."""

    user: str
    system: str = ""
    current_date: str | None = None

    def format(self, **kwargs: Any) -> Iterable[ChatCompletionMessageParam]:
        """Format the template with the given arguments."""
        messages = []
        if not self.system:
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            self.system = default_template_system_prompt.format(
                current_date=self.current_date or now
            )
        messages.append({"role": "system", "content": self.system.format(**kwargs)})
        messages.append({"role": "user", "content": self.user.format(**kwargs)})
        return messages


OutputFormat = Literal["json", "text"]


@dataclass
class LLMArgs:
    model: str
    output_format: OutputFormat = "text"
    temperature: float = 0.3
    top_p: float = 0.95
    seed: int = 47
    n: int = 1

    @classmethod
    def default(cls) -> "LLMArgs":
        return LLMArgs(model="openai/gpt-4o-mini")
