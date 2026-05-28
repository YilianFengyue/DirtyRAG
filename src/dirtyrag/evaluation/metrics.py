from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from dirtyrag.data.io import read_jsonl, write_jsonl
from dirtyrag.evaluation.normalize import contains_answer, normalize_text


CONFLICT_TERMS = ("conflict", "contradict", "inconsistent", "multiple answers", "cannot determine")


def score_prediction(example: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    answer = str(prediction.get("answer", ""))
    gold_answers = [str(item) for item in example.get("gold_answers", [])]
    wrong_answers = [str(item) for item in example.get("wrong_answers", [])]
    has_gold = contains_answer(answer, gold_answers)
    has_wrong = contains_answer(answer, wrong_answers)
    conflict_flag = any(term in normalize_text(answer) for term in CONFLICT_TERMS)
    strict_success = has_gold and not has_wrong

    return {
        "qid": prediction.get("qid"),
        "method": prediction.get("method"),
        "strict_success": float(strict_success),
        "gold_coverage": float(has_gold),
        "wrong_leakage": float(has_wrong),
        "conflict_sensitivity": float(conflict_flag),
        "latency_sec": prediction.get("latency_sec"),
        "total_tokens": prediction.get("total_tokens"),
    }


def summarize_metrics(per_case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_case_rows:
        grouped[str(row["method"])].append(row)

    summary = []
    for method, rows in sorted(grouped.items()):
        summary.append(
            {
                "method": method,
                "num_examples": len(rows),
                "strict_success": mean(rows, "strict_success"),
                "gold_coverage": mean(rows, "gold_coverage"),
                "wrong_leakage": mean(rows, "wrong_leakage"),
                "conflict_sensitivity": mean(rows, "conflict_sensitivity"),
                "avg_latency": mean(rows, "latency_sec"),
                "avg_total_tokens": mean(rows, "total_tokens"),
            }
        )
    return summary


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def evaluate_files(dataset_path: Path, predictions_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    examples = {row["qid"]: row for row in read_jsonl(dataset_path)}
    predictions = read_jsonl(predictions_path)
    per_case = []
    for prediction in predictions:
        example = examples.get(prediction["qid"])
        if example is None:
            continue
        per_case.append(score_prediction(example, prediction))
    return per_case, summarize_metrics(per_case)


def write_metrics(run_dir: Path, dataset_path: Path) -> None:
    per_case, summary = evaluate_files(dataset_path, run_dir / "predictions.jsonl")
    write_jsonl(run_dir / "per_case_metrics.jsonl", per_case)
    write_csv(run_dir / "metrics.csv", summary)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0])
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(row.get(header, "")) for header in headers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

