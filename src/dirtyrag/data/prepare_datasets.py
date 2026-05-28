from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from dirtyrag.data.io import read_jsonl, write_jsonl
from dirtyrag.schemas import Document, QAExample


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def import_hf_load_dataset():
    """Import Hugging Face datasets even if ./datasets exists in the repo."""
    original_path = list(sys.path)
    sys.path = [
        path
        for path in sys.path
        if path not in {"", str(PROJECT_ROOT), str(PROJECT_ROOT.resolve())}
    ]
    try:
        from datasets import load_dataset
    finally:
        sys.path = original_path
    return load_dataset


DATASET_NAMES = {
    "ramdocs": "HanNight/RAMDocs",
    "faitheval_inconsistent": "Salesforce/FaithEval-inconsistent-v1.0",
    "faitheval_unanswerable": "Salesforce/FaithEval-unanswerable-v1.0",
}

LOCAL_RAMDOCS_PATH = Path("datasets/RAMDocs/raw/ramdocs/repo/RAMDocs_test.jsonl")


def load_ramdocs() -> Iterable[dict[str, Any]]:
    if LOCAL_RAMDOCS_PATH.exists():
        return read_jsonl(LOCAL_RAMDOCS_PATH)
    load_dataset = import_hf_load_dataset()
    return load_dataset(DATASET_NAMES["ramdocs"], split="test")


def load_hf_test(name: str):
    load_dataset = import_hf_load_dataset()
    return load_dataset(DATASET_NAMES[name], split="test")


def normalize_ramdocs(rows: Iterable[dict[str, Any]], limit: int | None) -> Iterable[QAExample]:
    for idx, row in enumerate(rows):
        if limit is not None and idx >= limit:
            break
        documents = []
        for doc_idx, doc in enumerate(row.get("documents", []), start=1):
            documents.append(
                Document(
                    doc_id=f"D{doc_idx}",
                    text=str(doc.get("text", "")),
                    source_type=doc.get("type"),
                    source_answer=doc.get("answer"),
                    metadata={
                        key: value
                        for key, value in doc.items()
                        if key not in {"text", "type", "answer"}
                    },
                )
            )
        yield QAExample(
            qid=f"ramdocs_{idx + 1:06d}",
            dataset="ramdocs",
            question=str(row.get("question", "")),
            documents=documents,
            gold_answers=[str(x) for x in row.get("gold_answers", [])],
            wrong_answers=[str(x) for x in row.get("wrong_answers", [])],
            task_type="conflict_qa",
            metadata={"disambig_entity": row.get("disambig_entity", [])},
        )


def normalize_faitheval(
    rows: Iterable[dict[str, Any]],
    *,
    dataset_name: str,
    task_type: str,
    limit: int | None,
) -> Iterable[QAExample]:
    for idx, row in enumerate(rows):
        if limit is not None and idx >= limit:
            break
        qid = str(row.get("qid") or f"{dataset_name}_{idx + 1:06d}")
        context = str(row.get("context", ""))
        yield QAExample(
            qid=qid,
            dataset=dataset_name,
            question=str(row.get("question", "")),
            documents=[
                Document(
                    doc_id="D1",
                    text=context,
                    source_type="unknown",
                    source_answer=None,
                    metadata={"context_source": "faitheval"},
                )
            ],
            gold_answers=[str(x) for x in row.get("answers", [])],
            wrong_answers=[],
            task_type=task_type,
            metadata={
                "subset": row.get("subset"),
                "justification": row.get("justification"),
            },
        )


def build_dataset(dataset: str, limit: int | None, output_dir: Path) -> Path:
    suffix = "all" if limit is None else str(limit)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "ramdocs":
        output_path = output_dir / f"ramdocs_{suffix}.jsonl"
        write_jsonl(output_path, normalize_ramdocs(load_ramdocs(), limit))
        return output_path

    if dataset == "faitheval_inconsistent":
        output_path = output_dir / f"faitheval_inconsistent_{suffix}.jsonl"
        rows = load_hf_test(dataset)
        write_jsonl(
            output_path,
            normalize_faitheval(
                rows,
                dataset_name="faitheval_inconsistent",
                task_type="inconsistency_detection",
                limit=limit,
            ),
        )
        return output_path

    if dataset == "faitheval_unanswerable":
        output_path = output_dir / f"faitheval_unanswerable_{suffix}.jsonl"
        rows = load_hf_test(dataset)
        write_jsonl(
            output_path,
            normalize_faitheval(
                rows,
                dataset_name="faitheval_unanswerable",
                task_type="unanswerable_detection",
                limit=limit,
            ),
        )
        return output_path

    raise ValueError(f"Unsupported dataset: {dataset}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["ramdocs", "faitheval_inconsistent", "faitheval_unanswerable", "all"],
        required=True,
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    datasets = list(DATASET_NAMES) if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        output_path = build_dataset(dataset, args.limit, args.output_dir)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
