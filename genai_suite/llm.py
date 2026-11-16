"""Answer-generation backends.

* ``LangChainBackend`` wraps any LangChain chat model (Claude, GPT, a local Ollama model ...).
* ``LocalExtractiveBackend`` needs no API key: it composes the answer from the best-matching
  retrieved sentences. It keeps the whole pipeline runnable offline, in CI and in the browser.

``get_backend()`` picks Claude via ``langchain-anthropic`` when ``ANTHROPIC_API_KEY`` is set.
"""

from __future__ import annotations

import os
import re
from typing import Protocol

from langchain_core.documents import Document

from .rag.vector_store import expand_query
from .text import content_tokens, split_sentences

PROMPT_SYSTEM = (
    "You are an enterprise operations assistant. Answer the employee or customer request using ONLY "
    "the numbered context passages. Cite every claim with its passage number like [1]. If the context "
    "does not contain the answer, say so plainly. Keep the answer under 90 words."
)


class Backend(Protocol):
    name: str

    def answer(self, question: str, contexts: list[tuple[Document, float]]) -> str: ...


def format_context(contexts: list[tuple[Document, float]]) -> str:
    return "\n\n".join(
        f"[{i}] ({doc.metadata['title']} - {doc.metadata['section']}) {doc.page_content}"
        for i, (doc, _) in enumerate(contexts, 1)
    )


class LocalExtractiveBackend:
    name = "local-extractive"

    def __init__(self, max_sentences: int = 3, term_weights: dict[str, float] | None = None):
        self.max_sentences = max_sentences
        # IDF weights from the vector store, so rare terms ("delete") outweigh common ones ("data").
        self.term_weights = term_weights

    def answer(self, question: str, contexts: list[tuple[Document, float]]) -> str:
        q = set(content_tokens(expand_query(question)))
        candidates = []
        for rank, (doc, score) in enumerate(contexts, 1):
            for pos, sentence in enumerate(split_sentences(doc.page_content)):
                shared = q & set(content_tokens(sentence))
                if shared:
                    weight = sum((self.term_weights or {}).get(t, 1.0) for t in shared)
                    candidates.append((weight * (1 + 2 * score), -rank, -pos, sentence, rank, len(shared)))
        # Prefer sentences sharing at least two key terms with the request when any exist.
        if any(c[-1] >= 2 for c in candidates):
            candidates = [c for c in candidates if c[-1] >= 2]
        if not candidates:
            return "I could not find this in the knowledge base."
        candidates = [c[:-1] for c in candidates]
        best = sorted(candidates, reverse=True)[: self.max_sentences]
        # Present in reading order: by passage rank, then position inside the passage.
        best.sort(key=lambda c: (-c[1], -c[2]))
        return " ".join(f"{s} [{rank}]" for *_, s, rank in best)


class LangChainBackend:
    def __init__(self, chat_model):
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [("system", PROMPT_SYSTEM), ("human", "Context:\n{context}\n\nRequest: {question}")]
        )
        self.chain = prompt | chat_model | StrOutputParser()
        self.name = getattr(chat_model, "model", None) or type(chat_model).__name__

    def answer(self, question: str, contexts: list[tuple[Document, float]]) -> str:
        return self.chain.invoke({"context": format_context(contexts), "question": question}).strip()


def get_backend() -> Backend:
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic

            model = os.getenv("GENAI_MODEL", "claude-sonnet-5-5")
            return LangChainBackend(ChatAnthropic(model=model, temperature=0, max_tokens=400))
        except ImportError:
            pass
    return LocalExtractiveBackend()


def strip_citations(text: str) -> str:
    return re.sub(r"\s*\[\d+\]", "", text)
