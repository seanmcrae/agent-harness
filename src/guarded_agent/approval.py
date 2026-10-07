"""Human-approval hook for tools classified as write (side-effecting)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ApprovalRequest:
    agent: str
    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ApprovalDecision:
    approved: bool
    reviewer: str
    reason: str = ""


Approver = Callable[[ApprovalRequest], ApprovalDecision]


def approve_all(request: ApprovalRequest) -> ApprovalDecision:
    return ApprovalDecision(approved=True, reviewer="auto-approve")


def deny_all(request: ApprovalRequest) -> ApprovalDecision:
    """The runtime default: write tools never run unless an approver is configured."""
    return ApprovalDecision(
        approved=False, reviewer="auto-deny", reason="no approver configured for write tools"
    )
