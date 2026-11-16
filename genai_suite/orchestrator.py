"""Wires the specialist agents into a supervised loop: intake -> route -> (research -> analyse -> critique)* -> respond."""

from __future__ import annotations

import csv
from pathlib import Path

from .agents import AnalystAgent, CriticAgent, IntakeAgent, ResearchAgent, ResponderAgent, RouterAgent, TaskState
from .llm import Backend, LocalExtractiveBackend, get_backend
from .processors import FewShotClassifier
from .rag import TorchVectorStore, chunk_documents, load_corpus

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class Orchestrator:
    def __init__(self, store: TorchVectorStore, classifier: FewShotClassifier, backend: Backend, max_attempts: int = 2):
        self.store, self.classifier, self.backend = store, classifier, backend
        if isinstance(backend, LocalExtractiveBackend) and backend.term_weights is None:
            backend.term_weights = store.term_weights()
        self.intake = IntakeAgent(classifier)
        self.router = RouterAgent()
        self.research = ResearchAgent(store)
        self.analyst = AnalystAgent(backend, store)
        self.critic = CriticAgent(max_attempts=max_attempts)
        self.responder = ResponderAgent()

    @classmethod
    def from_data(cls, data_dir: str | Path = DATA_DIR, backend: Backend | None = None) -> "Orchestrator":
        data_dir = Path(data_dir)
        store = TorchVectorStore(chunk_documents(load_corpus(data_dir / "corpus")))
        with open(data_dir / "tickets.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        classifier = FewShotClassifier().fit([r["text"] for r in rows], [r["label"] for r in rows])
        return cls(store, classifier, backend or get_backend())

    def run(self, request: str) -> TaskState:
        state = self.router(self.intake(TaskState(request=request)))
        while True:
            state = self.critic(self.analyst(self.research(state)))
            if not state.needs_retry:
                break
            state.attempts += 1
        return self.responder(state)
