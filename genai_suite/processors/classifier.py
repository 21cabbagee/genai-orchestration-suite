"""Few-shot request classifier: TF-IDF features into a PyTorch softmax layer.

With ~8 labelled examples per class this trains in well under a second on CPU, and the trained
weights are small enough to export to JSON and run in the browser demo.
"""

from __future__ import annotations

import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from ..text import STOPWORDS, content_tokens


class FewShotClassifier:
    def __init__(self, epochs: int = 400, lr: float = 0.5, weight_decay: float = 1e-4, seed: int = 7):
        self.epochs, self.lr, self.weight_decay, self.seed = epochs, lr, weight_decay, seed
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, analyzer=content_tokens)
        self.labels: list[str] = []
        self.model: torch.nn.Linear | None = None

    def _features(self, texts: list[str]) -> torch.Tensor:
        return torch.tensor(self.vectorizer.transform(texts).toarray(), dtype=torch.float32)

    def fit(self, texts: list[str], labels: list[str]) -> "FewShotClassifier":
        torch.manual_seed(self.seed)
        self.labels = sorted(set(labels))
        self.vectorizer.fit(texts)
        x = self._features(texts)
        y = torch.tensor([self.labels.index(label) for label in labels])
        self.model = torch.nn.Linear(x.shape[1], len(self.labels))
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss()
        for _ in range(self.epochs):
            optim.zero_grad()
            loss = loss_fn(self.model(x), y)
            loss.backward()
            optim.step()
        self.final_loss = float(loss.detach())
        return self

    @torch.no_grad()
    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        assert self.model is not None, "call fit() first"
        probs = torch.softmax(self.model(self._features(texts)), dim=1)
        return [dict(zip(self.labels, row.tolist())) for row in probs]

    def predict(self, text: str) -> tuple[str, float]:
        probs = self.predict_proba([text])[0]
        label = max(probs, key=probs.get)
        return label, probs[label]

    def export(self) -> dict:
        assert self.model is not None
        vocab = sorted(self.vectorizer.vocabulary_, key=self.vectorizer.vocabulary_.get)
        return {
            "labels": self.labels,
            "vocab": vocab,
            "idf": [round(float(v), 6) for v in self.vectorizer.idf_],
            "weights": [[round(float(w), 5) for w in row] for row in self.model.weight.detach()],
            "bias": [round(float(b), 5) for b in self.model.bias.detach()],
        }
