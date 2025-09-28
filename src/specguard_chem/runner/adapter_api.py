from __future__ import annotations

"""Typed protocol for adapter <-> runner interactions."""

from typing import Any, Dict, Literal, Optional, TypedDict

Action = Literal["propose", "tool_call", "abstain"]


class ToolSpec(TypedDict):
    name: str
    schema: Dict[str, Any]


class AgentRequest(TypedDict, total=False):
    task: Dict[str, Any]
    round: int
    tools: list[ToolSpec]
    failure_vector: Optional[Dict[str, Any]]
    interrupt: Optional[Dict[str, Any]]


class AgentResponse(TypedDict, total=False):
    action: Action
    smiles: Optional[str]
    name: Optional[str]
    args: Optional[Dict[str, Any]]
    cited_specs: Optional[list[str]]
    confidence: Optional[float]
    reason: Optional[str]
