"""
Evaluate retrieval quality on a labelled question set.

Metrics (document-level: a hit means a chunk from the expected PEP was retrieved):
- Hit@1, Hit@3, Hit@5  share of questions with a correct chunk in the top k
- MRR@10              mean reciprocal rank of the first correct chunk

Usage:
    python eval/run_eval.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from rag.generator import ExtractiveGenerator  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402

BLUE, ORANGE, AQUA, INK, INK_2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"


def first_correct_rank(hits, expected: str) -> int | None:
    return next((h.rank for h in hits if h.chunk.doc_id == expected), None)


def main() -> None:
    questions = [json.loads(l) for l in (ROOT / "eval" / "questions.jsonl").read_text().splitlines() if l.strip()]
    t0 = time.perf_counter()
    rag = RAGPipeline(generator=ExtractiveGenerator())
    print(f"Indexed {len(rag.docs)} documents into {len(rag.chunks)} chunks in {time.perf_counter() - t0:.1f}s\n")

    rows, misses = [], []
    for key, retriever in rag.retrievers().items():
        ranks, latencies = [], []
        for q in questions:
            t = time.perf_counter()
            hits = retriever.search(q["question"], k=10)
            latencies.append((time.perf_counter() - t) * 1000)
            r = first_correct_rank(hits, q["expected_doc"])
            ranks.append(r)
            if key == "hybrid" and (r is None or r > 3):
                misses.append({"question": q["question"], "expected": q["expected_doc"], "rank": r,
                               "top_result": hits[0].chunk.citation})
        n = len(questions)
        rows.append({
            "retriever": retriever.name,
            "hit@1": sum(r == 1 for r in ranks) / n,
            "hit@3": sum(r is not None and r <= 3 for r in ranks) / n,
            "hit@5": sum(r is not None and r <= 5 for r in ranks) / n,
            "mrr@10": sum(1 / r for r in ranks if r) / n,
            "avg_latency_ms": sum(latencies) / n,
        })

    df = pd.DataFrame(rows)
    out = ROOT / "eval" / "results.csv"
    df.round(3).to_csv(out, index=False)
    print(df.round(3).to_string(index=False))
    if misses:
        print("\nHybrid misses (correct PEP not in top 3):")
        for m in misses:
            print(f"  - {m['question']}  expected {m['expected']}, rank {m['rank']}, got '{m['top_result']}'")

    fig, ax = plt.subplots(figsize=(9, 3.8))
    metrics = ["hit@1", "hit@3", "hit@5", "mrr@10"]
    width = 0.26
    for i, (row, color) in enumerate(zip(rows, [BLUE, ORANGE, AQUA])):
        xs = [m + (i - 1) * width for m in range(len(metrics))]
        ax.bar(xs, [row[m] for m in metrics], width=width - 0.02, color=color, label=row["retriever"])
        for x, m in zip(xs, metrics):
            ax.text(x, row[m] + 0.01, f"{row[m]:.2f}", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(range(len(metrics)), [m.upper() for m in metrics])
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Retrieval accuracy on {len(questions)} labelled questions", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(ROOT / "images" / "retrieval_eval.png", dpi=150)


if __name__ == "__main__":
    main()
