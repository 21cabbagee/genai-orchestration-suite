"""Shared state and the base agent contract.

Every agent reads and writes one typed ``TaskState`` (pydantic) and appends to its trace, so each
run is fully auditable: which agent did what, with which inputs, and how long it took.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field


class AgentStep(BaseModel):
    agent: str
    action: str
    detail: dict[str, Any] = Field(default_factory=dict)
    ms: float = 0.0


class TaskState(BaseModel):
    request: str
    redacted: str = ""
    pii: dict[str, int] = Field(default_factory=dict)
    summary: str = ""
    category: str = ""
    category_confidence: float = 0.0
    entities: dict[str, list[str]] = Field(default_factory=dict)
    priority: str = "normal"
    plan: list[str] = Field(default_factory=list)
    query: str = ""
    k: int = 3
    retrieved: list[dict[str, Any]] = Field(default_factory=list)
    draft: str = ""
    grounding: float = 0.0
    coverage: float = 0.0
    top_score: float = 0.0
    attempts: int = 0
    needs_retry: bool = False
    resolution: str = ""
    final: str = ""
    trace: list[AgentStep] = Field(default_factory=list)


class Agent:
    name = "agent"

    def run(self, state: TaskState) -> TaskState:  # pragma: no cover - interface
        raise NotImplementedError

    def log(self, state: TaskState, action: str, **detail: Any) -> None:
        state.trace.append(AgentStep(agent=self.name, action=action, detail=detail))

    def __call__(self, state: TaskState) -> TaskState:
        start = time.perf_counter()
        before = len(state.trace)
        state = self.run(state)
        elapsed = (time.perf_counter() - start) * 1000
        for step in state.trace[before:]:
            step.ms = round(elapsed / max(1, len(state.trace) - before), 2)
        return state
