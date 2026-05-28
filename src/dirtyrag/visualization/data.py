from __future__ import annotations

from pathlib import Path
from typing import Any

from dirtyrag.data.io import read_jsonl
from dirtyrag.evaluation.metrics import score_prediction, summarize_metrics


METHOD_ORDER = [
    "direct_llm",
    "vanilla_rag",
    "relevance_filter_rag",
    "crag_style_rag",
    "evidenceboard_rag",
]

METHOD_DISPLAY = {
    "direct_llm": "Direct LLM",
    "vanilla_rag": "Vanilla RAG",
    "relevance_filter_rag": "RAG + Filter",
    "crag_style_rag": "CRAG-style",
    "evidenceboard_rag": "EB-RAG (Ours)",
}


def resolve_dataset_path(run_dir: Path) -> Path:
    f = run_dir / "dataset_path.txt"
    if not f.exists():
        raise FileNotFoundError(f"Missing dataset_path.txt in {run_dir}")
    return Path(f.read_text(encoding="utf-8").strip())


def load_predictions(run_dir: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(run_dir / "predictions.jsonl"))


def load_combined_metrics(
    primary_run: Path,
    override_runs: list[Path] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load predictions from primary_run; if override_runs are provided, any
    (qid, method) pair in an override run replaces the same pair in primary.

    Returns (per_case_rows, summary_rows).
    """
    examples = {row["qid"]: row for row in read_jsonl(resolve_dataset_path(primary_run))}

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for pred in load_predictions(primary_run):
        merged[(pred["qid"], pred["method"])] = pred
    for run in override_runs or []:
        for pred in load_predictions(run):
            merged[(pred["qid"], pred["method"])] = pred

    per_case = []
    for pred in merged.values():
        example = examples.get(pred["qid"])
        if example is None:
            continue
        scored = score_prediction(example, pred)
        scored["total_tokens"] = pred.get("total_tokens")
        scored["latency_sec"] = pred.get("latency_sec")
        per_case.append(scored)

    summary = summarize_metrics(per_case)
    return per_case, summary


def order_summary(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method = {row["method"]: row for row in summary}
    return [by_method[m] for m in METHOD_ORDER if m in by_method]
