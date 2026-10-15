"""
Ask a question from the command line.

    python ask.py "Why was the walrus operator added?"
    python ask.py --retriever hybrid "How do I type a decorator that keeps the wrapped signature?"
"""
import argparse

from rag.pipeline import RAGPipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--retriever", choices=["bm25", "semantic", "hybrid"], default=None)
    ap.add_argument("-k", type=int, default=5, help="number of passages to retrieve")
    args = ap.parse_args()

    rag = RAGPipeline(top_k=args.k)
    ans = rag.ask(args.question, args.retriever)
    print(f"\n{ans.answer}\n")
    print(f"Sources ({ans.retriever}, answered by {ans.generator}):")
    for h in ans.sources:
        print(f"  [{h.rank}] {h.chunk.citation}")


if __name__ == "__main__":
    main()
