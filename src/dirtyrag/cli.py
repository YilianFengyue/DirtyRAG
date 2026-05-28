from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from dirtyrag.config import load_config
from dirtyrag.data.io import read_jsonl, write_jsonl
from dirtyrag.evaluation.metrics import write_metrics
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

    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.command == "run":
        run_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)


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
        method = method_cls(llm, max_tokens=max_tokens)
        for example in tqdm(examples, desc=method_name):
            result = method.run(example)
            predictions.append(result)

    write_jsonl(run_dir / "predictions.jsonl", predictions)
    sync_latest(run_dir)
    print(f"Wrote {run_dir / 'predictions.jsonl'}")
    print(f"Run dir: {run_dir}")


def evaluate_command(args: argparse.Namespace) -> None:
    dataset_path = args.dataset_path
    if dataset_path is None:
        dataset_path_file = args.run_dir / "dataset_path.txt"
        if not dataset_path_file.exists():
            raise RuntimeError("Missing --dataset-path and run_dir/dataset_path.txt")
        dataset_path = Path(dataset_path_file.read_text(encoding="utf-8").strip())
    write_metrics(args.run_dir, dataset_path)
    sync_latest(args.run_dir)
    print(f"Wrote {args.run_dir / 'metrics.csv'}")


def parse_methods(methods_arg: str, config: dict[str, Any]) -> list[str]:
    if methods_arg:
        method_names = [item.strip() for item in methods_arg.split(",") if item.strip()]
    else:
        method_names = list(config.get("run", {}).get("methods", ["direct_llm", "vanilla_rag"]))
    unsupported = [name for name in method_names if name not in METHOD_REGISTRY]
    if unsupported:
        raise ValueError(f"Unsupported methods in step1: {unsupported}")
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
