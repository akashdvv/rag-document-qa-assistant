"""End-to-end RAG pipeline: ingest -> index -> retrieve -> generate (with citations)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag.generator import default_generator
from rag.ingest import chunk_documents, load_documents
from rag.retrievers import BM25Retriever, Hit, HybridRetriever, SemanticRetriever

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "peps"


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Hit]
    generator: str
    retriever: str


class RAGPipeline:
    def __init__(self, corpus_dir: Path = CORPUS, generator=None, top_k: int = 5):
        self.docs = load_documents(corpus_dir)
        self.chunks = chunk_documents(self.docs)
        self.bm25 = BM25Retriever(self.chunks)
        self.semantic = SemanticRetriever(self.chunks)
        self.hybrid = HybridRetriever([self.bm25, self.semantic])
        self.generator = generator or default_generator()
        self.top_k = top_k

    def retrievers(self) -> dict:
        return {"bm25": self.bm25, "semantic": self.semantic, "hybrid": self.hybrid}

    @property
    def default_retriever(self) -> str:
        """Evaluation (eval/run_eval.py) showed BM25 alone beats hybrid when only the offline
        LSA vectors are available; hybrid is used when neural embeddings are loaded."""
        return "bm25" if self.semantic.name == "Semantic (LSA)" else "hybrid"

    def ask(self, question: str, retriever: str | None = None) -> Answer:
        r = self.retrievers()[retriever or self.default_retriever]
        hits = r.search(question, self.top_k)
        return Answer(question, self.generator.generate(question, hits), hits, self.generator.name, r.name)
