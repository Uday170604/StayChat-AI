"""
End-to-end pipeline for StayChat Hotel RAG assessment.

Usage:
  python main.py --all              # dataset -> chunks -> index -> evaluate
  python main.py --evaluate         # run Q1-Q3 + metrics (index must exist)
  python main.py --ablation         # hallucination before/after demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import GROUND_TRUTH_PATH, OUTPUTS_DIR  # noqa: E402
from scripts.generate_dataset import main as generate_dataset  # noqa: E402
from src.embed_store import build_index  # noqa: E402
from src.evaluation import format_report, run_full_evaluation  # noqa: E402
from src.hallucination import run_ablation  # noqa: E402
from src.preprocessing import build_chunks  # noqa: E402
from src.retrieval import clear_retriever_cache, retrieve  # noqa: E402


def run_ablation_demo() -> None:
    gt = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    edge = gt["edge_case_query"]
    chunks = retrieve(edge["text"])
    result = run_ablation(edge["text"], chunks)
    lines = [
        "# Hallucination control ablation",
        "",
        f"**Query:** {edge['text']}",
        "",
        "## Without controls (permissive prompt)",
        result["without_controls"]["answer"],
        "",
        "## With controls (threshold + strict prompt + verification)",
        result["with_controls"]["answer"],
        "",
        "### Why this reduces hallucination",
        "- **Layer 1 — Retrieval threshold:** blocks generation when no chunk is sufficiently similar.",
        "- **Layer 2 — Query-term grounding:** abstains when key question terms (e.g. *tigers*) are absent from context.",
        "- **Layer 3 — Strict prompt:** instructs the model to use only provided chunks.",
        "- **Layer 4 — Lexical verification:** downgrades answers whose tokens lack support in retrieved text.",
        "",
    ]
    out = OUTPUTS_DIR / "hallucination_ablation.md"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hotel RAG pipeline")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation only")
    parser.add_argument("--ablation", action="store_true", help="Hallucination ablation")
    args = parser.parse_args()

    if not any([args.all, args.evaluate, args.ablation]):
        args.all = True

    if args.all:
        generate_dataset()
        chunks = build_chunks()
        print(f"Built {len(chunks)} chunks")
        build_index()
        clear_retriever_cache()
        print("FAISS index built")

    if args.all or args.evaluate:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        eval_result = run_full_evaluation()
        report = format_report(eval_result)
        report_path = OUTPUTS_DIR / "sample_outputs.md"
        report_path.write_text(report, encoding="utf-8")
        json_path = OUTPUTS_DIR / "evaluation_results.json"
        json_path.write_text(json.dumps(eval_result, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {report_path}")
        print(f"Wrote {json_path}")
        print(f"Mean Precision@{eval_result.get('top_k', 7)}: {eval_result['mean_precision_at_k']}")
        print(f"Mean RR: {eval_result['mean_reciprocal_rank']}")

    if args.all or args.ablation:
        run_ablation_demo()


if __name__ == "__main__":
    main()
