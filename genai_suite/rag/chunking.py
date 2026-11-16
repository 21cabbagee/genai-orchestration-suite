"""Load markdown knowledge-base files and split them into section-level chunks."""

from pathlib import Path

from langchain_core.documents import Document


def load_corpus(corpus_dir: str | Path) -> list[Document]:
    docs = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        first = text.splitlines()[0] if text else path.stem
        title = first.lstrip("# ").strip() if first.startswith("#") else path.stem
        docs.append(Document(page_content=text, metadata={"source": path.name, "title": title}))
    return docs


def chunk_documents(docs: list[Document], max_words: int = 120, overlap: int = 25) -> list[Document]:
    """Split on '## ' headings; window-split any section longer than ``max_words``."""
    chunks: list[Document] = []
    for doc in docs:
        sections: list[tuple[str, list[str]]] = []
        for line in doc.page_content.splitlines():
            if line.startswith("## "):
                sections.append((line[3:].strip(), []))
            elif line.startswith("# ") or not line.strip():
                continue
            elif sections:
                sections[-1][1].append(line.strip())
            else:
                sections.append(("Overview", [line.strip()]))
        for section, lines in sections:
            words = " ".join(lines).split()
            step = max(1, max_words - overlap)
            for start in range(0, max(1, len(words)), step):
                body = " ".join(words[start:start + max_words])
                if not body:
                    continue
                chunks.append(Document(
                    page_content=body,
                    metadata={**doc.metadata, "section": section, "chunk_id": f"{doc.metadata['source']}#{len(chunks)}"},
                ))
                if start + max_words >= len(words):
                    break
    return chunks
