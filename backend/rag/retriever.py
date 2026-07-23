"""
retriever.py
=============
A lightweight, dependency-minimal retrieval engine over the project's
knowledge base (backend/rag/knowledge/*.md) plus the project's own
README.md and REPORT.md. Uses TF-IDF + cosine similarity (scikit-learn) --
no external embedding API or model download required, so this works
identically with or without an LLM API key configured.

Chunking strategy: split each markdown file on level-2 headers ("## ") so
each chunk is a coherent, citeable section rather than an arbitrary
fixed-length window.
"""

import os
import re
from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(HERE, "knowledge")
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))


@dataclass
class Chunk:
    text: str
    source: str
    title: str


def _split_into_chunks(text: str, source_name: str) -> List[Chunk]:
    # Split on level-1 or level-2 markdown headers, keeping the header with its body.
    parts = re.split(r"\n(?=#{1,2} )", text)
    chunks = []
    for part in parts:
        part = part.strip()
        if not part or len(part) < 40:
            continue
        first_line = part.splitlines()[0].lstrip("#").strip()
        chunks.append(Chunk(text=part, source=source_name, title=first_line or source_name))
    return chunks


def _load_all_chunks() -> List[Chunk]:
    chunks: List[Chunk] = []

    if os.path.isdir(KNOWLEDGE_DIR):
        for fname in sorted(os.listdir(KNOWLEDGE_DIR)):
            if fname.endswith(".md"):
                path = os.path.join(KNOWLEDGE_DIR, fname)
                with open(path, encoding="utf-8") as f:
                    chunks.extend(_split_into_chunks(f.read(), fname))

    for doc_name in ["README.md", "REPORT.md"]:
        path = os.path.join(PROJECT_ROOT, doc_name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                chunks.extend(_split_into_chunks(f.read(), doc_name))

    return chunks


class Retriever:
    def __init__(self):
        self.chunks = _load_all_chunks()
        texts = [c.text for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9)
        self.matrix = self.vectorizer.fit_transform(texts) if texts else None

    def search(self, query: str, top_k: int = 4) -> List[dict]:
        if not self.chunks or self.matrix is None:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        ranked_idx = scores.argsort()[::-1][:top_k]
        results = []
        for i in ranked_idx:
            if scores[i] <= 0:
                continue
            c = self.chunks[i]
            results.append({"text": c.text, "source": c.source, "title": c.title, "score": float(scores[i])})
        return results


_retriever_instance: Retriever = None


def get_retriever() -> Retriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance
