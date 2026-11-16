"""A small in-memory vector store: scikit-learn TF-IDF features, PyTorch similarity search.

The embedding function is pluggable (``embed_fn``); TF-IDF is the default because it needs no
model download and can be reproduced byte-for-byte in the browser demo.
"""

from __future__ import annotations

import re
from typing import Callable

import torch
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer

from ..text import STOPWORDS, content_tokens

# Light query expansion used when the critic agent asks for a second retrieval pass.
SYNONYMS = {
    "reimburse": "expense claim reimbursement",
    "reimbursement": "expense claim",
    "laptop": "hardware device replacement",
    "computer": "laptop device",
    "vendor": "supplier vendor onboarding",
    "supplier": "vendor",
    "money": "refund payment",
    "wfh": "remote work",
    "home": "remote work",
    "paternity": "parental leave non-birthing",
    "maternity": "parental leave birthing",
    "password": "password reset lockout account",
    "locked": "lockout password reset",
    "quote": "quotations",
    "quotes": "quotations",
    "delete": "data subject requests deletion",
    "breach": "incident reporting security",
    "late": "deadlines late-submission exception",
}


def expand_query(query: str) -> str:
    extra = [SYNONYMS[w] for w in re.findall(r"\w+", query.lower()) if w in SYNONYMS]
    return query + (" " + " ".join(extra) if extra else "")


class TorchVectorStore:
    def __init__(self, chunks: list[Document], embed_fn: Callable[[list[str]], torch.Tensor] | None = None):
        self.chunks = chunks
        texts = [self._index_text(c) for c in chunks]
        if embed_fn is None:
            self.vectorizer = TfidfVectorizer(sublinear_tf=True, analyzer=content_tokens)
            self.vectorizer.fit(texts)
            embed_fn = self._tfidf_embed
        self.embed_fn = embed_fn
        self.matrix = torch.nn.functional.normalize(self.embed_fn(texts), dim=1)

    @staticmethod
    def _index_text(chunk: Document) -> str:
        return f"{chunk.metadata.get('title', '')} {chunk.metadata.get('section', '')} {chunk.page_content}"

    def _tfidf_embed(self, texts: list[str]) -> torch.Tensor:
        return torch.tensor(self.vectorizer.transform(texts).toarray(), dtype=torch.float32)

    def search(self, query: str, k: int = 3, boost_source: str | None = None,
               boost: float = 0.1) -> list[tuple[Document, float]]:
        """Cosine top-k. ``boost_source`` adds a metadata prior (e.g. from the intent classifier)."""
        q = torch.nn.functional.normalize(self.embed_fn([query]), dim=1)[0]
        scores = self.matrix @ q
        if boost_source:
            prior = torch.tensor([boost if c.metadata["source"] == boost_source else 0.0 for c in self.chunks])
            scores = scores + prior * (scores > 0)
        top = torch.topk(scores, min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for s, i in zip(top.values.tolist(), top.indices.tolist()) if s > 0]

    def term_weights(self) -> dict[str, float]:
        return {t: float(self.vectorizer.idf_[i]) for t, i in self.vectorizer.vocabulary_.items()}

    def export(self) -> dict:
        """Everything the browser needs to rebuild identical TF-IDF vectors."""
        vocab = sorted(self.vectorizer.vocabulary_, key=self.vectorizer.vocabulary_.get)
        return {
            "vocab": vocab,
            "idf": [round(float(x), 6) for x in self.vectorizer.idf_],
            "chunks": [
                {"id": c.metadata["chunk_id"], "title": c.metadata["title"], "section": c.metadata["section"],
                 "source": c.metadata["source"], "text": c.page_content, "index_text": self._index_text(c)}
                for c in self.chunks
            ],
            "synonyms": SYNONYMS,
            "stopwords": sorted(STOPWORDS),
        }
