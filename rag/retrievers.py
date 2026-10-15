"""
Three retrievers and a hybrid that fuses them.

- BM25Retriever       sparse keyword matching; great for exact names (e.g. "ParamSpec", "WSGI")
- SemanticRetriever   dense vectors; matches meaning even when the words differ.
                      Uses a sentence-transformers model when available, otherwise an
                      offline LSA model (TF-IDF + truncated SVD) so the project runs anywhere.
- HybridRetriever     Reciprocal Rank Fusion of the above; robust to either one failing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from rag.ingest import Chunk

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+|\d+")
STOPWORDS = set("""a an and are as at be by can do does for from has have how i if in into is it its
of on or that the their them then there these this to was were what when where which who why will
with would should could you your we our not no""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in (w.lower() for w in TOKEN_RE.findall(text)) if t not in STOPWORDS]


def embedding_text(chunk: Chunk) -> str:
    """Prepend title and section so the vector knows what document a chunk belongs to."""
    return f"{chunk.doc_id} {chunk.title}. {chunk.section}. {chunk.text}"


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


class BM25Retriever:
    name = "BM25"

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(embedding_text(c)) for c in chunks])

    def search(self, query: str, k: int = 5) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [Hit(self.chunks[i], float(scores[i]), r) for r, i in enumerate(top, 1)]


class SemanticRetriever:
    def __init__(self, chunks: list[Chunk], model_name: str = "all-MiniLM-L6-v2", n_components: int = 200):
        self.chunks = chunks
        texts = [embedding_text(c) for c in chunks]
        self._st_model = None
        try:  # dense neural embeddings if the model can be loaded
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(model_name)
            self.vectors = self._st_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            self.name = f"Semantic ({model_name})"
        except Exception:  # offline fallback: Latent Semantic Analysis
            self.tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                         ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            matrix = self.tfidf.fit_transform(texts)
            self.svd = TruncatedSVD(n_components=min(n_components, matrix.shape[1] - 1), random_state=0)
            self.vectors = normalize(self.svd.fit_transform(matrix))
            self.name = "Semantic (LSA)"

    def embed(self, query: str) -> np.ndarray:
        if self._st_model is not None:
            return self._st_model.encode([query], normalize_embeddings=True)[0]
        return normalize(self.svd.transform(self.tfidf.transform([query])))[0]

    def search(self, query: str, k: int = 5) -> list[Hit]:
        sims = self.vectors @ self.embed(query)
        top = np.argsort(sims)[::-1][:k]
        return [Hit(self.chunks[i], float(sims[i]), r) for r, i in enumerate(top, 1)]


class HybridRetriever:
    """Reciprocal Rank Fusion: score = sum(1 / (rrf_k + rank)) across retrievers.
    Rank-based, so BM25 scores and cosine similarities never need to be put on one scale."""

    name = "Hybrid (RRF)"

    def __init__(self, retrievers: list, rrf_k: int = 60, depth: int = 30):
        self.retrievers, self.rrf_k, self.depth = retrievers, rrf_k, depth

    def search(self, query: str, k: int = 5) -> list[Hit]:
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for r in self.retrievers:
            for hit in r.search(query, self.depth):
                fused[hit.chunk.chunk_id] = fused.get(hit.chunk.chunk_id, 0.0) + 1 / (self.rrf_k + hit.rank)
                by_id[hit.chunk.chunk_id] = hit.chunk
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [Hit(by_id[cid], score, r) for r, (cid, score) in enumerate(ranked, 1)]
