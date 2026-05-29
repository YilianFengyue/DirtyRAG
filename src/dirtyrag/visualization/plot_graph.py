from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

from dirtyrag.data.io import read_jsonl
from dirtyrag.visualization.data import METHOD_DISPLAY, METHOD_ORDER, resolve_dataset_path
from dirtyrag.visualization.style import apply_scientific_style, save_figure


SOURCE_TYPE_COLOR = {
    "correct": "#3FA66A",
    "misinfo": "#D1495B",
    "noise": "#9CA3AF",
    "unknown": "#D1D5DB",
}

EDGE_STYLE = {
    "contradict": {"color": "#C0392B", "style": "solid", "width": 1.8},
    "support": {"color": "#3FA66A", "style": "dashed", "width": 1.2},
    "duplicate": {"color": "#6B7280", "style": "dotted", "width": 1.4},
}


def plot_case_evidence_graph(
    qid: str,
    board: dict[str, Any],
    example: dict[str, Any],
    predictions: list[dict[str, Any]],
    out_dir: Path,
) -> Path:
    apply_scientific_style()

    cards = board.get("cards", [])
    edges = board.get("conflict_edges", [])
    final_answer = board.get("final_answer", "")
    decision = board.get("candidate_decision", {})
    verifier = board.get("verifier_decision", {})

    doc_source = {d["doc_id"]: d.get("source_type", "unknown") for d in example.get("documents", [])}
    doc_answer = {d["doc_id"]: d.get("source_answer") for d in example.get("documents", [])}
    card_by_doc = {c["doc_id"]: c for c in cards}

    fig = plt.figure(figsize=(13.5, 7.4), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 0.55],
                          left=0.04, right=0.98, top=0.91, bottom=0.06,
                          wspace=0.18, hspace=0.32)
    ax_graph = fig.add_subplot(gs[:, 0])
    ax_table = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, 1])

    G = nx.Graph()
    for c in cards:
        G.add_node(c["doc_id"])
    for e in edges:
        G.add_edge(e["src"], e["dst"], relation=e.get("relation", "support"))

    if G.number_of_nodes() > 0:
        pos = nx.spring_layout(G, seed=7, k=1.6 / max(1, G.number_of_nodes() ** 0.5))
        # normalize layout into [0,1]^2 then leave headroom for labels/legend
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        xr = (x1 - x0) or 1.0
        yr = (y1 - y0) or 1.0
        pos = {n: ((p[0] - x0) / xr, (p[1] - y0) / yr) for n, p in pos.items()}
    else:
        pos = {}

    # edges
    for relation, style in EDGE_STYLE.items():
        edgelist = [(u, v) for u, v, d in G.edges(data=True) if d["relation"] == relation]
        if not edgelist:
            continue
        nx.draw_networkx_edges(
            G, pos, edgelist=edgelist, ax=ax_graph,
            edge_color=style["color"], style=style["style"], width=style["width"],
            alpha=0.85,
        )

    # nodes
    node_colors = [SOURCE_TYPE_COLOR.get(doc_source.get(n, "unknown"), "#D1D5DB") for n in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos, ax=ax_graph,
        node_color=node_colors, edgecolors="#1F2937", linewidths=1.0,
        node_size=1500,
    )
    nx.draw_networkx_labels(G, pos, ax=ax_graph, font_size=10, font_weight="bold")

    # node side labels (answer candidate)
    for n, (x, y) in pos.items():
        card = card_by_doc.get(n, {})
        ans = (card.get("answer_candidate") or "unknown")
        if len(ans) > 28:
            ans = ans[:28] + "…"
        ax_graph.text(
            x, y - 0.13, ans,
            ha="center", va="top", fontsize=8.5, color="#374151",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#E5E7EB", linewidth=0.6),
        )

    # leave margins so node side-labels never clip
    ax_graph.set_xlim(-0.18, 1.18)
    ax_graph.set_ylim(-0.30, 1.18)
    ax_graph.set_title(f"Evidence Graph  ·  {qid}", loc="left", fontweight="bold")
    ax_graph.set_axis_off()

    # graph legend — anchored below the axes so it never overlaps nodes
    legend_handles = [
        mpatches.Patch(facecolor=SOURCE_TYPE_COLOR["correct"], edgecolor="#1F2937", label="correct"),
        mpatches.Patch(facecolor=SOURCE_TYPE_COLOR["misinfo"], edgecolor="#1F2937", label="misinfo"),
        mpatches.Patch(facecolor=SOURCE_TYPE_COLOR["noise"], edgecolor="#1F2937", label="noise"),
        plt.Line2D([0], [0], color=EDGE_STYLE["contradict"]["color"], lw=1.8, label="contradict"),
        plt.Line2D([0], [0], color=EDGE_STYLE["support"]["color"], lw=1.2, ls="--", label="support"),
        plt.Line2D([0], [0], color=EDGE_STYLE["duplicate"]["color"], lw=1.4, ls=":", label="duplicate"),
    ]
    ax_graph.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        fontsize=8.5,
        columnspacing=1.4,
        handletextpad=0.5,
    )

    # answers table
    ax_table.set_title("Method answers", loc="left", fontweight="bold")
    ax_table.set_axis_off()
    by_method = {p["method"]: p for p in predictions}
    table_rows = []
    for m in METHOD_ORDER:
        if m not in by_method:
            continue
        ans = str(by_method[m].get("answer", "")).strip().replace("\n", " ")
        if len(ans) > 70:
            ans = ans[:70] + "…"
        marker = _verdict_marker(by_method[m].get("answer", ""), example)
        table_rows.append([METHOD_DISPLAY.get(m, m), marker, ans])

    if table_rows:
        table = ax_table.table(
            cellText=table_rows,
            colLabels=["Method", "", "Answer"],
            colWidths=[0.32, 0.08, 0.60],
            cellLoc="left",
            loc="upper left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.8)
        table.scale(1, 1.35)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#E5E7EB")
            if row == 0:
                cell.set_text_props(weight="bold", color="white")
                cell.set_facecolor("#374151")
            else:
                m_name = METHOD_ORDER[row - 1] if row - 1 < len(METHOD_ORDER) else None
                if m_name == "evidenceboard_rag":
                    cell.set_facecolor("#FFF4F5")

    # text panel: question + decision trace
    ax_text.set_axis_off()
    question = str(example.get("question", ""))
    gold = ", ".join(example.get("gold_answers", []) or [])
    wrong = ", ".join(example.get("wrong_answers", []) or [])

    decision_reason = decision.get("reason", "")
    verifier_verdict = verifier.get("verdict", "")
    verifier_reason = verifier.get("reason", "")

    text = (
        f"Question  {_wrap(question, 78)}\n"
        f"Gold      {_wrap(gold or '—', 78)}\n"
        f"Wrong     {_wrap(wrong or '—', 78)}\n"
        f"\n"
        f"EB-RAG candidate  [{decision.get('mode', '?')}] → {decision.get('answer', '?')}\n"
        f"   {_wrap(decision_reason, 78)}\n"
        f"EB-RAG verifier   [{verifier_verdict}] → final = {final_answer}\n"
        f"   {_wrap(verifier_reason, 78)}"
    )
    ax_text.text(0, 1, text, ha="left", va="top", fontsize=8.6,
                 family="monospace", color="#1F2937")
    ax_text.set_title("Case context", loc="left", fontweight="bold")

    fig.suptitle(
        f"Case Study  ·  {qid}",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )

    out_path = out_dir / f"case_{qid}.png"
    save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def _verdict_marker(answer: str, example: dict[str, Any]) -> str:
    from dirtyrag.evaluation.normalize import contains_answer

    has_gold = contains_answer(answer, example.get("gold_answers", []))
    has_wrong = contains_answer(answer, example.get("wrong_answers", []))
    if has_gold and not has_wrong:
        return "OK"
    if has_wrong:
        return "leak"
    return "—"


def _wrap(text: str, width: int) -> str:
    import textwrap

    if not text:
        return ""
    return "\n          ".join(textwrap.wrap(text, width=width)) or text


def pick_case_studies(
    per_case: list[dict[str, Any]],
    n_win: int = 2,
    n_loss: int = 1,
    n_neutral: int = 0,
) -> list[str]:
    """Pick representative qids for case-study figures.

    Categories (in priority order):
      * strong-win  : Vanilla wrong-leakage, Ours strict success (clearest story)
      * win         : Vanilla not strict, Ours strict success
      * loss        : Ours fails (wrong-leak or missed gold) — for honest limitations
      * neutral     : both Ours and Vanilla strict (sanity)
    """
    by_qid_method: dict[tuple[str, str], dict[str, Any]] = {
        (r["qid"], r["method"]): r for r in per_case
    }
    qids = sorted({r["qid"] for r in per_case})

    strong_wins: list[str] = []
    wins: list[str] = []
    losses_leak: list[str] = []
    losses_other: list[str] = []
    neutrals: list[str] = []

    for qid in qids:
        ours = by_qid_method.get((qid, "evidenceboard_rag"))
        vanilla = by_qid_method.get((qid, "vanilla_rag"))
        if ours is None:
            continue
        ours_ok = ours["strict_success"] == 1.0
        ours_leak = ours["wrong_leakage"] == 1.0
        vanilla_ok = vanilla is not None and vanilla["strict_success"] == 1.0
        vanilla_leak = vanilla is not None and vanilla["wrong_leakage"] == 1.0

        if ours_ok and vanilla_leak:
            strong_wins.append(qid)
        elif ours_ok and not vanilla_ok:
            wins.append(qid)
        elif ours_leak:
            losses_leak.append(qid)
        elif not ours_ok:
            losses_other.append(qid)
        elif ours_ok and vanilla_ok:
            neutrals.append(qid)

    picked: list[str] = []
    picked.extend(strong_wins[:n_win])
    if len(picked) < n_win:
        picked.extend(wins[: n_win - len(picked)])
    picked.extend(losses_leak[:n_loss])
    if len(picked) - len(strong_wins) - len(wins[: max(0, n_win - len(strong_wins))]) < n_loss:
        # not enough leak losses; fall back to other failure modes
        already = set(picked)
        for qid in losses_other:
            if qid not in already and len(picked) < n_win + n_loss:
                picked.append(qid)
    picked.extend(neutrals[:n_neutral])

    if not picked:
        picked = qids[:1]
    return picked


def load_case_inputs(run_dir: Path, qid: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    board_path = run_dir / "evidence_boards" / f"{qid}_evidenceboard_rag.json"
    if not board_path.exists():
        raise FileNotFoundError(board_path)
    board = json.loads(board_path.read_text(encoding="utf-8"))
    dataset_path = resolve_dataset_path(run_dir)
    examples = {row["qid"]: row for row in read_jsonl(dataset_path)}
    example = examples[qid]
    return board, example, []


def collect_predictions_for_case(
    qid: str,
    runs: list[Path],
) -> list[dict[str, Any]]:
    by_method: dict[str, dict[str, Any]] = {}
    for run in runs:
        path = run / "predictions.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if row.get("qid") == qid:
                by_method[row["method"]] = row
    return [by_method[m] for m in METHOD_ORDER if m in by_method]
