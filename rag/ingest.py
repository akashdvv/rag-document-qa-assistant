"""
Load reStructuredText documents, strip markup, and split them into overlapping,
section-aware chunks that keep their source metadata (document, title, section).

Why section-aware: a chunk that crosses from "Rejected Ideas" into "Specification"
mixes two unrelated contexts and confuses both retrieval and the LLM. Splitting on
headings first, then windowing long sections, keeps each chunk on one topic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

HEADER_RE = re.compile(r"^([A-Za-z-]+):\s*(.*)$")
UNDERLINE_RE = re.compile(r"^([=\-~^\"'`#*+])\1{2,}\s*$")


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"{self.doc_id} - {self.title} > {self.section}"


def parse_header(lines: list[str]) -> tuple[dict, int]:
    """PEP files start with RFC-822 style headers terminated by a blank line."""
    meta, key, i = {}, None, 0
    for i, line in enumerate(lines):
        if not line.strip():
            break
        m = HEADER_RE.match(line)
        if m:
            key = m.group(1)
            meta[key] = m.group(2).strip()
        elif key and line.startswith((" ", "\t")):
            meta[key] += " " + line.strip()
    return meta, i + 1


def clean_rst(text: str) -> str:
    """Remove reST markup that adds noise to embeddings but keep the words."""
    text = re.sub(r"^\.\. (?:code-block|code|highlight|note|warning|seealso|rejected|"
                  r"versionadded|_[^:]+|\[[^\]]+\]|image|figure|toctree|contents)::?.*$", "",
                  text, flags=re.MULTILINE)
    text = re.sub(r"^\.\. .*$", "", text, flags=re.MULTILINE)          # remaining comments/directives
    text = re.sub(r":[a-z]+:`~?([^`<]+?)(?:\s*<[^>]+>)?`", r"\1", text)  # :pep:`484` -> 484
    text = re.sub(r"`([^`<]+?)\s*<[^>]+>`_+", r"\1", text)              # `link <url>`_ -> link
    text = re.sub(r"``([^`]+)``", r"\1", text)                          # ``code`` -> code
    text = re.sub(r"`([^`]+)`_*", r"\1", text)
    text = re.sub(r"\[#?[\w-]*\]_", "", text)                           # footnote refs
    text = re.sub(r"\s*::\s*$", ":", text, flags=re.MULTILINE)           # literal-block markers
    text = re.sub(r"^\s*::\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(folder: Path) -> list[Document]:
    docs = []
    for path in sorted(folder.glob("*.rst")):
        lines = path.read_text(encoding="utf-8").splitlines()
        meta, body_start = parse_header(lines)
        number = meta.get("PEP", path.stem.split("-")[-1]).strip()
        docs.append(Document(doc_id=f"PEP {int(number)}", title=meta.get("Title", path.stem),
                             text="\n".join(lines[body_start:]),
                             metadata={"status": meta.get("Status", ""),
                                       "python_version": meta.get("Python-Version", ""),
                                       "source_file": path.name}))
    return docs


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split on reST headings (a line followed by an underline of =, -, ~ ...)."""
    lines = text.splitlines()
    sections, current_title, buf = [], "Introduction", []
    i = 0
    while i < len(lines):
        if (i + 1 < len(lines) and lines[i].strip() and UNDERLINE_RE.match(lines[i + 1])
                and len(lines[i + 1].strip()) >= len(lines[i].strip()) * 0.6):
            if buf:
                sections.append((current_title, "\n".join(buf)))
            current_title, buf = lines[i].strip(), []
            i += 2
            continue
        buf.append(lines[i])
        i += 1
    if buf:
        sections.append((current_title, "\n".join(buf)))
    return sections


def window(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if len(words) <= size:
        return [words]
    step = size - overlap
    return [words[i:i + size] for i in range(0, max(len(words) - overlap, 1), step)]


def chunk_documents(docs: list[Document], chunk_words: int = 180, overlap_words: int = 40,
                    min_words: int = 25) -> list[Chunk]:
    chunks = []
    for doc in docs:
        n = 0
        for section, raw in split_sections(doc.text):
            if section.lower() in {"copyright", "references", "footnotes", "acknowledgements",
                                   "acknowledgments", "change history"}:
                continue
            words = clean_rst(raw).split()
            if len(words) < min_words:
                continue
            for piece in window(words, chunk_words, overlap_words):
                n += 1
                chunks.append(Chunk(chunk_id=f"{doc.doc_id.replace(' ', '-')}#{n}", doc_id=doc.doc_id,
                                    title=clean_rst(doc.title), section=clean_rst(section),
                                    text=" ".join(piece),
                                    metadata=doc.metadata))
    return chunks
