"""The specialist agents that make up the orchestration graph."""

from __future__ import annotations

import re

from ..llm import Backend, strip_citations
from ..processors import FewShotClassifier, extract_entities, infer_priority, redact_pii, summarize
from ..rag.vector_store import TorchVectorStore, expand_query
from ..text import content_tokens, split_sentences
from .base import Agent, TaskState

CATEGORY_SOURCE = {
    "expense": "expense_policy.md",
    "it_access": "it_access_policy.md",
    "procurement": "procurement_policy.md",
    "refund": "refund_sop.md",
    "hr_leave": "leave_policy.md",
    "data_privacy": "data_retention_policy.md",
}

CATEGORY_QUEUE = {
    "expense": "Finance - Reimbursements",
    "it_access": "IT Service Desk",
    "procurement": "Procurement",
    "refund": "Customer Support - Billing",
    "hr_leave": "HR Operations",
    "data_privacy": "Privacy & Security",
}


class IntakeAgent(Agent):
    """Redacts PII, summarises long requests, classifies intent and extracts entities."""

    name = "intake"

    def __init__(self, classifier: FewShotClassifier):
        self.classifier = classifier

    def run(self, state: TaskState) -> TaskState:
        state.redacted, state.pii = redact_pii(state.request)
        self.log(state, "redact_pii", redacted=state.pii or "none found")
        words = len(state.redacted.split())
        state.summary = summarize(state.redacted, 2) if words > 60 else state.redacted
        if words > 60:
            self.log(state, "summarize", from_words=words, to_words=len(state.summary.split()))
        state.category, state.category_confidence = self.classifier.predict(state.summary)
        self.log(state, "classify", category=state.category, confidence=round(state.category_confidence, 3))
        state.entities = {k: v for k, v in extract_entities(state.redacted).items() if v}
        state.priority = infer_priority(state.redacted)
        self.log(state, "extract", entities=state.entities, priority=state.priority)
        return state


class RouterAgent(Agent):
    """Plans the remaining work: which agents run and how wide retrieval should be."""

    name = "router"

    def run(self, state: TaskState) -> TaskState:
        # Placeholders such as [EMAIL] carry no meaning for retrieval.
        state.query = re.sub(r"\[[A-Z]+\]", " ", state.summary)
        state.query = re.sub(r"\s+", " ", state.query).strip()
        state.k = 4 if state.category_confidence < 0.5 else 3
        state.plan = ["research", "analyst", "critic", "responder"]
        reason = "low classifier confidence, widening retrieval" if state.k == 4 else "confident intent"
        self.log(state, "plan", plan=state.plan, k=state.k, reason=reason)
        return state


class ResearchAgent(Agent):
    """Retrieves supporting passages; on retries it expands the query."""

    name = "research"

    def __init__(self, store: TorchVectorStore):
        self.store = store

    def run(self, state: TaskState) -> TaskState:
        query = state.query if state.attempts == 0 else expand_query(state.query)
        k = state.k + state.attempts
        prior = CATEGORY_SOURCE.get(state.category) if state.category_confidence >= 0.5 else None
        hits = self.store.search(query, k=k, boost_source=prior)
        state.retrieved = [
            {"id": d.metadata["chunk_id"], "title": d.metadata["title"], "section": d.metadata["section"],
             "text": d.page_content, "score": round(s, 4)}
            for d, s in hits
        ]
        state.top_score = state.retrieved[0]["score"] if state.retrieved else 0.0
        self.log(state, "retrieve", query=query, k=k, prior=prior,
                 hits=[f"{r['title']} > {r['section']} ({r['score']:.2f})" for r in state.retrieved])
        return state


class AnalystAgent(Agent):
    """Drafts a cited answer from the retrieved passages via the configured LLM backend."""

    name = "analyst"

    def __init__(self, backend: Backend, store: TorchVectorStore):
        self.backend, self.store = backend, store

    def run(self, state: TaskState) -> TaskState:
        by_id = {c.metadata["chunk_id"]: c for c in self.store.chunks}
        contexts = [(by_id[r["id"]], r["score"]) for r in state.retrieved]
        state.draft = self.backend.answer(state.query, contexts)
        self.log(state, "draft_answer", backend=self.backend.name, chars=len(state.draft))
        return state


class CriticAgent(Agent):
    """Checks the draft is grounded in the retrieved evidence and asks for a retry if not."""

    name = "critic"

    def __init__(self, min_grounding: float = 0.7, min_retrieval: float = 0.18, min_coverage: float = 0.25,
                 max_attempts: int = 2):
        self.min_grounding, self.min_retrieval, self.max_attempts = min_grounding, min_retrieval, max_attempts
        self.min_coverage = min_coverage

    @staticmethod
    def coverage_score(query: str, draft: str) -> float:
        """Share of the request's key terms the draft actually addresses (a cheap relevance check)."""
        q = set(content_tokens(query))
        return len(q & set(content_tokens(strip_citations(draft)))) / len(q) if q else 0.0

    @staticmethod
    def grounding_score(draft: str, passages: list[str]) -> float:
        evidence = set(content_tokens(" ".join(passages)))
        sentences = [s for s in split_sentences(strip_citations(draft)) if content_tokens(s)]
        if not sentences:
            return 0.0
        supported = 0
        for sentence in sentences:
            tokens = content_tokens(sentence)
            if sum(t in evidence for t in tokens) / len(tokens) >= 0.6:
                supported += 1
        return supported / len(sentences)

    def run(self, state: TaskState) -> TaskState:
        state.grounding = round(self.grounding_score(state.draft, [r["text"] for r in state.retrieved]), 3)
        query = state.query if state.attempts == 0 else expand_query(state.query)
        state.coverage = round(self.coverage_score(query, state.draft), 3)
        weak = (state.grounding < self.min_grounding or state.top_score < self.min_retrieval
                or state.coverage < self.min_coverage)
        state.needs_retry = weak and state.attempts < self.max_attempts
        verdict = "retry with expanded query" if state.needs_retry else ("accept" if not weak else "accept, low confidence")
        self.log(state, "review", grounding=state.grounding, coverage=state.coverage,
                 top_score=state.top_score, verdict=verdict)
        return state


class ResponderAgent(Agent):
    """Decides the resolution path and writes the final response."""

    name = "responder"

    def __init__(self, min_confidence: float = 0.45):
        self.min_confidence = min_confidence

    def run(self, state: TaskState) -> TaskState:
        confident = (state.grounding >= 0.7 and state.top_score >= 0.18 and state.coverage >= 0.25
                     and state.category_confidence >= self.min_confidence)
        if state.priority == "high":
            state.resolution = "escalated"
        elif confident:
            state.resolution = "auto_resolved"
        else:
            state.resolution = "agent_assist"
        queue = CATEGORY_QUEUE.get(state.category, "General")
        sources = sorted({f"{r['title']} - {r['section']}" for r in state.retrieved[:3]})
        lines = [state.draft, "", "Sources: " + "; ".join(sources)]
        if state.resolution == "escalated":
            lines.append(f"Routed to {queue} as HIGH priority for human follow-up.")
        elif state.resolution == "agent_assist":
            lines.append(f"Draft queued for review by {queue}.")
        state.final = "\n".join(lines)
        self.log(state, "respond", resolution=state.resolution, queue=queue)
        return state
