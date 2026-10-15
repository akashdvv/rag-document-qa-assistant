"""
Answer generation, grounded in retrieved chunks with numbered citations.

LLMGenerator      calls Claude through the Anthropic API (set ANTHROPIC_API_KEY).
ExtractiveGenerator  no API key needed: selects the most relevant sentences from the
                  retrieved chunks and cites them. Used as a fallback and as a baseline.
"""
from __future__ import annotations

import os
import re

from rag.retrievers import Hit, tokenize

SYSTEM_PROMPT = """You answer questions about Python Enhancement Proposals (PEPs) using ONLY the \
numbered context passages provided.

Rules:
- Cite every factual claim with the passage number in square brackets, e.g. [2].
- If the passages do not contain the answer, say "I couldn't find that in the indexed documents." \
Do not use outside knowledge.
- Be concise: 2-5 sentences, then a short code example only if it clarifies the answer."""


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{h.rank}] ({h.chunk.citation})\n{h.chunk.text}" for h in hits)


def build_user_prompt(question: str, hits: list[Hit]) -> str:
    return f"Context passages:\n\n{format_context(hits)}\n\nQuestion: {question}"


class LLMGenerator:
    name = "Claude"

    def __init__(self, model: str | None = None, max_tokens: int = 500, client=None):
        self.model = model or os.getenv("RAG_MODEL", "claude-haiku-5-5")
        self.max_tokens = max_tokens
        if client is None:
            import anthropic
            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.client = client

    def generate(self, question: str, hits: list[Hit]) -> str:
        msg = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(question, hits)}],
        )
        return "".join(block.text for block in msg.content if getattr(block, "type", "") == "text").strip()


class ExtractiveGenerator:
    name = "Extractive"

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    def generate(self, question: str, hits: list[Hit]) -> str:
        q = set(tokenize(question))
        scored = []
        for h in hits:
            for sent in re.split(r"(?<=[.!?])\s+", h.chunk.text):
                toks = tokenize(sent)
                if len(toks) < 5 or len(sent.split()) > 45:  # skip fragments and code/grammar dumps
                    continue
                overlap = len(q & set(toks)) / (len(q) or 1)
                scored.append((overlap - 0.02 * (h.rank - 1), sent.strip(), h.rank))
        if not scored or max(s[0] for s in scored) <= 0:
            return "I couldn't find that in the indexed documents."
        best = sorted(scored, key=lambda s: s[0], reverse=True)[: self.max_sentences]
        return " ".join(f"{sent} [{rank}]" for _, sent, rank in best)


def default_generator():
    """Use Claude when an API key is configured, otherwise the extractive fallback."""
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return LLMGenerator()
        except Exception:
            pass
    return ExtractiveGenerator()
