# DirtyRAG-Guard

Training-free, evidence-centric RAG robustness project for dirty knowledge bases.

Current first milestone:

1. Load public datasets with Hugging Face `datasets`.
2. Convert them into one internal QA JSONL format.
3. Keep method code from using evaluation-only labels such as `gold_answers`,
   `wrong_answers`, or document `source_type`.

## Dataset Sources

Primary:

- `HanNight/RAMDocs`

Supplementary:

- `Salesforce/FaithEval-inconsistent-v1.0`
- `Salesforce/FaithEval-unanswerable-v1.0`

RAMDocs can be loaded directly from Hugging Face, or from the local file already
present at:

```text
datasets/RAMDocs/raw/ramdocs/repo/RAMDocs_test.jsonl
```

The two FaithEval datasets may require Hugging Face login:

```powershell
huggingface-cli login
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Inspect Dataset Fields

```powershell
python -m dirtyrag.data.inspect_datasets --dataset ramdocs --limit 1
python -m dirtyrag.data.inspect_datasets --dataset faitheval_inconsistent --limit 1
python -m dirtyrag.data.inspect_datasets --dataset faitheval_unanswerable --limit 1
```

## Build Processed JSONL

Quick RAMDocs sample:

```powershell
python -m dirtyrag.data.prepare_datasets --dataset ramdocs --limit 50
```

All first-stage datasets:

```powershell
python -m dirtyrag.data.prepare_datasets --dataset all --limit 50
```

Expected outputs:

```text
data/processed/ramdocs_50.jsonl
data/processed/faitheval_inconsistent_50.jsonl
data/processed/faitheval_unanswerable_50.jsonl
```

## Acceptance Checklist

- `python -m dirtyrag.data.inspect_datasets --dataset ramdocs --limit 1`
  prints one example.
- `python -m dirtyrag.data.prepare_datasets --dataset ramdocs --limit 50`
  creates `data/processed/ramdocs_50.jsonl`.
- The processed RAMDocs file has exactly 50 lines.
- Each line is valid JSON with `qid`, `dataset`, `question`, `documents`,
  `gold_answers`, `wrong_answers`, `task_type`, and `metadata`.
- Each document has `doc_id`, `text`, `source_type`, `source_answer`, and
  `metadata`.
- Running `python -m pytest` passes.

## Step 1: Direct LLM and Vanilla RAG

No-API smoke test:

```powershell
python -m dirtyrag.data.prepare_datasets --dataset ramdocs --limit 50
python -m dirtyrag.cli run --config configs/step1_mock.yaml
python -m dirtyrag.cli evaluate --run-dir outputs/runs/latest
```

Real LLM run:

```powershell
$env:LLM_API_KEY="your_api_key"
$env:LLM_BASE_URL="https://api.openai.com/v1"
$env:LLM_MODEL="gpt-4o-mini"

python -m dirtyrag.cli run --config configs/quick.yaml --methods direct_llm,vanilla_rag --limit 10
python -m dirtyrag.cli evaluate --run-dir outputs/runs/latest
```

Expected step1 outputs:

```text
outputs/runs/latest/predictions.jsonl
outputs/runs/latest/per_case_metrics.jsonl
outputs/runs/latest/metrics.csv
outputs/runs/latest/llm_calls.jsonl
```
