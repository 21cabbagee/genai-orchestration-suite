"""Extractive summarisation by sentence centrality (TF-IDF cosine to the document centroid)."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..text import split_sentences


def summarize(text: str, max_sentences: int = 2) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    matrix = TfidfVectorizer().fit_transform(sentences).toarray()
    centroid = matrix.mean(axis=0)
    norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(centroid) or 1.0)
    scores = matrix @ centroid / np.where(norms == 0, 1, norms)
    keep = sorted(np.argsort(-scores)[:max_sentences])
    return " ".join(sentences[i] for i in keep)
