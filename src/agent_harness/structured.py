"""Structured final answers: JSON extraction, pydantic validation, and repair prompts."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

M = TypeVar("M", bound=BaseModel)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class StructuredOutputError(ValueError):
    pass


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating code fences and surrounding prose."""
    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)]
    candidates.append(text)
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate.strip())
        except json.JSONDecodeError:
            continue
    raise StructuredOutputError("final answer is not valid JSON")


def parse_structured(text: str, model: type[M]) -> M:
    data = extract_json(text)
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise StructuredOutputError(f"schema validation failed: {problems}") from exc


def schema_text(model: type[BaseModel]) -> str:
    return json.dumps(model.model_json_schema(), separators=(",", ":"))


def output_instructions(model: type[BaseModel]) -> str:
    return (
        "When you have finished using tools, reply with only a JSON object (no prose) that "
        f"matches this JSON schema:\n{schema_text(model)}"
    )


def repair_prompt(error: StructuredOutputError, model: type[BaseModel]) -> str:
    return (
        f"Your final answer could not be accepted: {error}. Reply again with only a JSON object "
        f"matching this schema:\n{schema_text(model)}"
    )
