# EvidenceBoard-RAG

**A Training-Free, Evidence-Centric Framework for Robust Retrieval-Augmented Generation over Dirty Knowledge Bases**

EvidenceBoard-RAG tackles *post-retrieval robustness*: given a fixed set of retrieved
documents that may be **conflicting, misleading, noisy, duplicated, or outdated**, it
structures each document into an **evidence card**, builds an explicit **conflict graph**,
aggregates answers with **anti-dilution scoring**, and verifies the result before
responding — abstaining or flagging conflict when the evidence does not support a
confident answer. No model training or fine-tuning is involved.

---

## Key Results (RAMDocs, 200-case subset)

Same LLM (DeepSeek `deepseek-chat`), same documents, `temperature = 0`, for all methods.

| Method | Strict Success ↑ | Gold Coverage ↑ | Wrong Leakage ↓ | Conflict Sens. ↑ | Avg Tokens |
|---|---:|---:|---:|---:|---:|
| Direct LLM | 0.050 | 0.055 | **0.010** | 0.000 | 141 |
| Vanilla RAG | 0.335 | 0.380 | 0.060 | 0.180 | 1091 |
| RAG + Relevance Filter | 0.360 | 0.400 | 0.055 | 0.195 | 2528 |
| CRAG-style | 0.220 | 0.220 | 0.015 | 0.140 | 2025 |
| **EvidenceBoard-RAG (Ours)** | **0.495** | **0.505** | 0.060 | **0.350** | 3596 |

- **+13.5 pts strict success** over the strongest baseline (≈ 37 % relative gain).
- **Conflict sensitivity nearly doubles** (0.35 vs 0.195).
- Residual failures are dominated by **conservative abstention** (73/200) rather than
  **confident wrong answers** (12/200) — a safer failure mode for dirty knowledge bases.
- Trade-off: ≈ **3.3× token cost** versus Vanilla RAG.

> We do **not** claim universal SOTA. Wrong-answer leakage is on par with Vanilla RAG and
> higher than abstention-heavy baselines; gains concentrate in conflict-heavy cases.

Figures and full numbers: [`paper_results/ramdocs_200_final/`](paper_results/ramdocs_200_final/).
Detailed analysis: [`docs/DirtyRAG_实验报告_RAMDocs200.md`](docs/DirtyRAG_实验报告_RAMDocs200.md).

---

## Method Overview

```
Question + retrieved documents (fixed)
        │
        ▼
  Evidence Card Extraction      one structured card per document
        │                       (answer candidate, stance, relevance, time cue, confidence)
        ▼
  Duplicate Grouping            cluster near-identical documents (Jaccard ≥ 0.82)
        │
        ▼
  Conflict Graph                support / contradict / duplicate edges between cards
        │
        ▼
  Anti-Dilution Aggregation     score answer clusters by *independent* support,
        │                       so repeated misinformation cannot win by volume
        ▼
  Candidate Decision            answer / conflict / unknown
        │
        ▼
  Verifier                      supported / revise / conflict / unknown
        │
        ▼
  Final Answer + Evidence Board JSON (fully auditable per case)
```

Every case persists an Evidence Board JSON capturing all intermediate state
(cards, duplicate groups, conflict edges, answer clusters, decisions, trace).

---

## Installation

Requires **Python 3.10+**.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
pytest -q                            # expect: 13 passed
```

<details>
<summary>Conda (recommended on HPC / servers where pip fails to build pyarrow or libcst)</summary>

```bash
conda create -n dirtyrag -c conda-forge -y \
  python=3.10 pyarrow datasets huggingface_hub openai \
  pydantic python-dotenv pyyaml tqdm pytest
conda activate dirtyrag
pip install -e . --no-deps
```
</details>

---

## Configuration

The LLM client targets any OpenAI-compatible endpoint. Provide credentials via a `.env`
file in the project root (never commit it):

```bash
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api.deepseek.com/v1   # or any OpenAI-compatible endpoint
LLM_MODEL=deepseek-chat
```

Experiment settings live in `configs/*.yaml`:

| Config | Cases | Purpose |
|---|---|---|
| `configs/ramdocs_200.yaml` | 200 | **Main experiment** (all methods) |
| `configs/default.yaml` | 50 | Standard run |
| `configs/quick.yaml` | 10 | Fast smoke run |
| `configs/step*_mock.yaml` | — | Offline mock runs (no API calls) |

---

## Reproducing the Main Result

```bash
# 1. Prepare the dataset (reads local RAMDocs if present, else downloads from HF)
python -m dirtyrag.data.prepare_datasets --dataset ramdocs --limit 200

# 2. Run all methods (use nohup / tmux for long runs)
python -m dirtyrag.cli run --config configs/ramdocs_200.yaml --limit 200

# 3. Compute metrics
python -m dirtyrag.cli evaluate --run-dir outputs/runs/latest

# 4. Generate figures
python -m dirtyrag.cli plot --run-dir outputs/runs/latest
```

- **Resume after interruption:** add `--resume-run outputs/runs/run_<timestamp>`;
  already-completed `(question, method)` pairs are skipped, and the LLM cache is reused.
- **Smoke test first:** `--limit 1` validates the full pipeline before a full run.
- **Re-plot without re-running:** point `plot --run-dir` at any completed run directory.

---

## Command-Line Interface

```bash
python -m dirtyrag.cli run      --config <yaml> [--limit N] [--methods a,b] [--mock] [--resume-run DIR]
python -m dirtyrag.cli evaluate --run-dir <DIR>
python -m dirtyrag.cli inspect  --run-dir <DIR> --qid <QID>     # single-case deep dive
python -m dirtyrag.cli plot     --run-dir <DIR> [--baseline-run DIR] [--cases q1,q2] [--no-cases]
```

---

## Methods

| ID | Method | Description |
|---|---|---|
| B0 | `direct_llm` | Closed-book; question only, no documents. |
| B1 | `vanilla_rag` | Concatenate all documents, then answer (Lewis et al., 2020). |
| B2 | `relevance_filter_rag` | LLM-judged per-document relevance filtering, then answer. |
| B3 | `crag_style_rag` | CRAG-style retrieval evaluator triages quality (no web search, no trained evaluator). |
| **Ours** | `evidenceboard_rag` | Evidence cards → conflict graph → anti-dilution aggregation → verifier. |

> The CRAG baseline is a **training-free approximation** of Corrective RAG
> (Yan et al., 2024): it keeps the retrieval-evaluator idea but replaces the trained
> evaluator with a zero-shot LLM judge and removes the web-search branch. We report it
> as *CRAG-style* and do not claim full reproduction.

---

## Evaluation Metrics

String-matching based (deterministic, no LLM judge), computed in
`src/dirtyrag/evaluation/metrics.py`:

- **Strict Success** — answer covers a gold answer **and** leaks no wrong answer (primary).
- **Gold Coverage** — answer contains any gold answer (lenient substring match).
- **Wrong Leakage** — answer contains any annotated wrong answer.
- **Conflict Sensitivity** — answer explicitly signals conflict/ambiguity.
- **Avg Latency / Avg Total Tokens** — per-case cost.

---

## Repository Structure

```
DirtyRAG/
├── src/dirtyrag/
│   ├── cli.py                  # CLI: run / evaluate / inspect / plot
│   ├── llm_client.py           # OpenAI-compatible client (cache + retry)
│   ├── prompts.py              # Prompt templates
│   ├── schemas.py              # QAExample / Document / MethodResult
│   ├── data/                   # Dataset loading & conversion (RAMDocs, FaithEval)
│   ├── methods/                # 5 methods (subclass BaseMethod)
│   ├── evidence/               # EvidenceBoard-RAG subsystem
│   │   ├── cards.py            #   evidence card extraction
│   │   ├── clustering.py       #   duplicate grouping
│   │   ├── graph.py            #   conflict graph
│   │   ├── scoring.py          #   anti-dilution aggregation & decision
│   │   └── board.py            #   evidence board assembly
│   ├── evaluation/             # Metrics
│   └── visualization/          # Scientific-style plotting
├── configs/                    # Experiment configs (ramdocs_200.yaml = main)
├── paper_results/              # Final curated results & figures (tracked in git)
│   └── ramdocs_200_final/
├── data/ , datasets/           # Processed & raw data (mostly git-ignored)
├── docs/                       # Project plan, design, experiment report
├── tests/                      # Unit tests (13)
├── requirements.txt , pyproject.toml
└── README.md
```

---

## Datasets

- **RAMDocs** — *Retrieval-Augmented Generation with Conflicting Evidence*, COLM 2025
  ([HanNight/RAMDocs](https://github.com/HanNight/RAMDocs)). Primary benchmark.
- **FaithEval-inconsistent / -unanswerable** — Salesforce
  ([SalesforceAIResearch/FaithEval](https://github.com/SalesforceAIResearch/FaithEval)).
  Loaders provided; not part of the main 200-case run.

Document type labels (`correct` / `misinfo` / `noise`) and `gold_answers` /
`wrong_answers` are used **only for evaluation** — methods never read them at inference
time (no data leakage).

---

## Limitations

- Single primary benchmark (RAMDocs); FaithEval / AmbigDocs runs are future work.
- Duplicate-stress and ablation studies are not yet included in this release.
- Lenient string-match evaluation may overestimate multi-answer coverage.
- ≈ 3.3× token cost; less favorable on clean / simple cases.

See [`docs/DirtyRAG_实验报告_RAMDocs200.md`](docs/DirtyRAG_实验报告_RAMDocs200.md) for the
full analysis, per-case breakdown, and case studies.

---

## Citation / References

- Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020.
- Yan et al. *Corrective Retrieval Augmented Generation.* 2024.
- Wang et al. *Retrieval-Augmented Generation with Conflicting Evidence (RAMDocs / MADAM-RAG).* COLM 2025.
- Ming et al. *FaithEval: Can Your Language Model Stay Faithful to Context …* 2024.
