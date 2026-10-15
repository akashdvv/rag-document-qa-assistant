import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.generator import ExtractiveGenerator, LLMGenerator, SYSTEM_PROMPT  # noqa: E402
from rag.ingest import Document, chunk_documents, clean_rst, split_sections  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.retrievers import BM25Retriever, HybridRetriever  # noqa: E402

RST = """Abstract
========

This PEP adds the ``strict`` flag to :func:`zip`. See `the docs <https://x.y>`_ for more.

Specification
=============

When strict is true, zip raises ValueError if iterables have different lengths, which helps
catch bugs where data silently gets truncated during processing in production code paths.

Copyright
=========

Public domain.
"""


@pytest.fixture(scope="module")
def rag():
    return RAGPipeline(generator=ExtractiveGenerator())


def test_clean_rst_removes_markup():
    out = clean_rst("Use ``x`` with :func:`zip` and `docs <https://a.b>`_.")
    assert out == "Use x with zip and docs."


def test_split_sections_and_skip_copyright():
    assert [t for t, _ in split_sections(RST)] == ["Abstract", "Specification", "Copyright"]
    chunks = chunk_documents([Document("PEP 618", "zip strict", RST)], min_words=5)
    assert {c.section for c in chunks} == {"Abstract", "Specification"}
    assert all(c.doc_id == "PEP 618" for c in chunks)


def test_long_sections_are_windowed_with_overlap():
    body = "Intro\n=====\n\n" + " ".join(f"w{i}" for i in range(500))
    chunks = chunk_documents([Document("PEP 1", "t", body)], chunk_words=180, overlap_words=40)
    assert len(chunks) == 4
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-40:] == second[:40]


def test_corpus_indexes(rag):
    assert len(rag.docs) == 30
    assert len(rag.chunks) > 1000


def test_exact_feature_query_finds_right_pep(rag):
    hits = rag.bm25.search("zip strict ValueError different lengths", k=3)
    assert hits[0].chunk.doc_id == "PEP 618"


def test_hybrid_fuses_and_ranks(rag):
    hits = rag.hybrid.search("structural pattern matching tutorial", k=5)
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
    assert any(h.chunk.doc_id in {"PEP 634", "PEP 636"} for h in hits)


def test_extractive_answer_cites_passages(rag):
    ans = rag.ask("How do I make zip raise an error when iterables have different lengths?")
    assert "[1]" in ans.answer and "ValueError" in ans.answer


def test_llm_generator_sends_grounded_prompt(rag):
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="Use strict=True [1].")])

    gen = LLMGenerator(model="test-model", client=SimpleNamespace(messages=FakeMessages()))
    hits = rag.bm25.search("zip strict", k=2)
    assert gen.generate("How do I use strict zip?", hits) == "Use strict=True [1]."
    assert captured["system"] == SYSTEM_PROMPT and captured["temperature"] == 0
    prompt = captured["messages"][0]["content"]
    assert "[1] (PEP 618" in prompt and "Question: How do I use strict zip?" in prompt
