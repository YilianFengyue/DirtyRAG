from __future__ import annotations

import argparse
import json
import sys
from itertools import islice
from typing import Any

from dirtyrag.data.prepare_datasets import DATASET_NAMES, load_hf_test, load_ramdocs


def preview_row(row: dict[str, Any]) -> dict[str, Any]:
    preview = {}
    for key, value in row.items():
        if isinstance(value, str):
            preview[key] = value[:500]
        elif isinstance(value, list):
            preview[key] = value[:2]
        else:
            preview[key] = value
    return preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["ramdocs", "faitheval_inconsistent", "faitheval_unanswerable"],
        required=True,
    )
    parser.add_argument("--limit", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    rows = load_ramdocs() if args.dataset == "ramdocs" else load_hf_test(args.dataset)
    print(f"dataset_id={DATASET_NAMES[args.dataset]}")
    for idx, row in enumerate(islice(rows, args.limit), start=1):
        print(f"\nexample={idx}")
        print(json.dumps(preview_row(dict(row)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
