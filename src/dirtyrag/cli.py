from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from dirtyrag.config import load_config
from dirtyrag.data.io import read_jsonl, write_jsonl
from dirtyrag.evaluation.metrics import score_prediction, write_metrics
from dirtyrag.llm_client import LLMClient
from dirtyrag.methods import METHOD_REGISTRY
from dirtyrag.schemas import QAExample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="dirtyrag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--methods", type=str, default="")
    run_parser.add_argument("--limit", type=int, default=None)
    run_parser.add_argument("--mock", action="store_true")

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--run-dir", type=Path, required=True)
    eval_parser.add_argument("--dataset-path", type=Path, default=None)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--run-dir", type=Path, required=True)
    inspect_parser.add_argument("--qid", type=str, required=True)
    inspect_parser.add_argument("--dataset-path", type=Path, default=None)

    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.command == "run":
        run_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "inspect":
        inspect_command(args)


def run_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.mock:
        config.setdefault("llm", {})["provider"] = "mock"

    dataset_path = Path(config["dataset"]["path"])
    examples = [QAExample.model_validate(row) for row in read_jsonl(dataset_path)]
    limit = args.limit if args.limit is not None else config.get("run", {}).get("max_examples")
    if limit is not None:
        examples = examples[: int(limit)]

    method_names = parse_methods(args.methods, config)
    run_dir = make_run_dir(Path(config.get("run", {}).get("output_dir", "outputs/runs")))
    shutil.copyfile(args.config, run_dir / "config.yaml")
    (run_dir / "dataset_path.txt").write_text(str(dataset_path), encoding="utf-8")

    llm = LLMClient.from_config(config.get("llm", {}), cache_path=run_dir / "llm_calls.jsonl")
    max_tokens = int(config.get("llm", {}).get("max_tokens", 512))
    predictions = []
    for method_name in method_names:
        method_cls = METHOD_REGISTRY[method_name]
        method = method_cls(llm, max_tokens=max_tokens, run_dir=run_dir)
        for example in tqdm(examples, desc=method_name):
            result = method.run(example)
            predictions.append(result)

    write_jsonl(run_dir / "predictions.jsonl", predictions)
    sync_latest(run_dir)
    print(f"Wrote {run_dir / 'predictions.jsonl'}")
    print(f"Run dir: {run_dir}")


def evaluate_command(args: argparse.Namespace) -> None:
    dataset_path = resolve_dataset_path(args.run_dir, args.dataset_path)
    write_metrics(args.run_dir, dataset_path)
    sync_latest(args.run_dir)
    print(f"Wrote {args.run_dir / 'metrics.csv'}")


def inspect_command(args: argparse.Namespace) -> None:
    dataset_path = resolve_dataset_path(args.run_dir, args.dataset_path)
    examples = {row["qid"]: row for row in read_jsonl(dataset_path)}
    predictions = [row for row in read_jsonl(args.run_dir / "predictions.jsonl") if row["qid"] == args.qid]
    example = examples.get(args.qid)
    if example is None:
        raise RuntimeError(f"QID not found in dataset: {args.qid}")
    if not predictions:
        raise RuntimeError(f"QID not found in predictions: {args.qid}")

    print(f"QID: {args.qid}")
    print(f"Question: {example.get('question', '')}")
    print(f"Gold answers: {example.get('gold_answers', [])}")
    print(f"Wrong answers: {example.get('wrong_answers', [])}")
    print("\nDocuments:")
    for doc in example.get("documents", []):
        text = str(doc.get("text", "")).replace("\n", " ")
        if len(text) > 220:
            text = text[:220].rstrip() + "..."
        print(
            f"- {doc.get('doc_id')}: source_type={doc.get('source_type')} "
            f"source_answer={doc.get('source_answer')} text={text}"
        )

    print("\nPredictions:")
    for prediction in predictions:
        score = score_prediction(example, prediction)
        print(f"\n[{prediction.get('method')}]")
        print(f"answer: {prediction.get('answer')}")
        print(f"used_doc_ids: {prediction.get('used_doc_ids')}")
        if prediction.get("evidence_board_path"):
            print(f"evidence_board_path: {prediction.get('evidence_board_path')}")
        print(
            "score: "
            f"strict={score['strict_success']} "
            f"gold={score['gold_coverage']} "
            f"wrong={score['wrong_leakage']} "
            f"conflict={score['conflict_sensitivity']}"
        )
        metadata = prediction.get("metadata") or {}
        for key in (
            "filtered_doc_ids",
            "dropped_doc_ids",
            "retrieval_verdict",
            "action",
            "reason",
            "candidate_mode",
            "candidate_answer",
            "verifier_verdict",
            "num_conflict_edges",
            "num_answer_clusters",
        ):
            if key in metadata:
                print(f"{key}: {metadata[key]}")


def resolve_dataset_path(run_dir: Path, dataset_path: Path | None) -> Path:
    if dataset_path is not None:
        return dataset_path
    dataset_path_file = run_dir / "dataset_path.txt"
    if not dataset_path_file.exists():
        raise RuntimeError("Missing --dataset-path and run_dir/dataset_path.txt")
    return Path(dataset_path_file.read_text(encoding="utf-8").strip())


def parse_methods(methods_arg: str, config: dict[str, Any]) -> list[str]:
    if methods_arg:
        method_names = [item.strip() for item in methods_arg.split(",") if item.strip()]
    else:
        method_names = list(config.get("run", {}).get("methods", ["direct_llm", "vanilla_rag"]))
    unsupported = [name for name in method_names if name not in METHOD_REGISTRY]
    if unsupported:
        raise ValueError(f"Unsupported methods: {unsupported}")
    return method_names


def make_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{timestamp}"
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = output_root / f"run_{timestamp}_{suffix}"
    run_dir.mkdir(parents=True)
    return run_dir


def sync_latest(run_dir: Path) -> None:
    output_root = run_dir.parent
    latest = output_root / "latest"
    if run_dir.resolve() == latest.resolve():
        return
    if latest.exists() or latest.is_symlink():
        if latest.is_dir() and not latest.is_symlink():
            shutil.rmtree(latest)
        else:
            latest.unlink()
    shutil.copytree(run_dir, latest)


if __name__ == "__main__":
    main()
