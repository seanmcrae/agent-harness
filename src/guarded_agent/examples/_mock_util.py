"""Helpers shared by the example agents' rule-based mock policies."""

from __future__ import annotations

import json
from typing import Any

from guarded_agent.providers import CompletionRequest
from guarded_agent.types import Message


def first_user_text(request: CompletionRequest) -> str:
    return next((m.content for m in request.messages if m.role == "user"), "")


def trailing_tool_messages(request: CompletionRequest) -> list[Message]:
    """Tool results answering the most recent assistant turn."""
    out: list[Message] = []
    for message in reversed(request.messages):
        if message.role != "tool":
            break
        out.append(message)
    return list(reversed(out))


def json_or_none(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
