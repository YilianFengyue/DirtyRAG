from __future__ import annotations

from dirtyrag.schemas import Document


SYSTEM_PROMPT = (
    "You are a careful retrieval-augmented question answering system. "
    "Use only the provided information. Keep the final answer short."
)


def direct_prompt(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Answer the question. If you do not know, answer \"unknown\".\n\n"
                f"Question:\n{question}\n\nFinal answer:"
            ),
        },
    ]


def vanilla_rag_prompt(question: str, documents: list[Document]) -> list[dict[str, str]]:
    docs_text = format_documents(documents)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Answer the question using only the provided documents.\n"
                "If the documents contain multiple valid answers, list all valid answers.\n"
                "If the answer is not supported by the documents, answer \"unknown\".\n"
                "If the documents are contradictory and no reliable resolution is possible, "
                "answer \"conflict\".\n\n"
                f"Question:\n{question}\n\nDocuments:\n{docs_text}\n\nFinal answer:"
            ),
        },
    ]


def relevance_judge_prompt(question: str, document: Document) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "You are judging whether a document is useful for answering a question.\n"
                "Use only the document text. Output JSON only.\n\n"
                f"Question:\n{question}\n\n"
                f"Document ID:\n{document.doc_id}\n\n"
                f"Document:\n{document.text}\n\n"
                "Output schema:\n"
                "{\n"
                '  "relevance": 0 | 1 | 2,\n'
                '  "reason": "short reason"\n'
                "}\n\n"
                "Definitions:\n"
                "0 = irrelevant/noise\n"
                "1 = partially relevant\n"
                "2 = directly relevant\n"
            ),
        },
    ]


def crag_retrieval_eval_prompt(question: str, documents: list[Document]) -> list[dict[str, str]]:
    docs_text = format_documents(documents, max_chars_per_doc=1200)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "You are a retrieval quality evaluator for a RAG system.\n"
                "Decide whether the retrieved documents are enough and reliable enough "
                "to answer the question. Output JSON only.\n\n"
                f"Question:\n{question}\n\nDocuments:\n{docs_text}\n\n"
                "Output schema:\n"
                "{\n"
                '  "retrieval_verdict": "correct|ambiguous|incorrect|insufficient",\n'
                '  "action": "answer|filter_then_answer|abstain",\n'
                '  "reason": "short reason"\n'
                "}\n\n"
                "Guidance:\n"
                "- correct: documents mostly support a clear answer.\n"
                "- ambiguous: documents contain multiple plausible answers or conflict.\n"
                "- incorrect: documents appear mostly irrelevant or misleading.\n"
                "- insufficient: documents do not provide enough support.\n"
            ),
        },
    ]


def crag_conservative_answer_prompt(
    question: str,
    documents: list[Document],
    *,
    retrieval_verdict: str,
    reason: str,
) -> list[dict[str, str]]:
    docs_text = format_documents(documents)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Answer the question using only the provided documents and the retrieval "
                "quality evaluation.\n"
                "If documents conflict, answer \"conflict\" unless the conflict can be "
                "resolved from the evidence.\n"
                "If documents are insufficient, answer \"unknown\".\n\n"
                f"Question:\n{question}\n\n"
                f"Retrieval verdict: {retrieval_verdict}\n"
                f"Retrieval reason: {reason}\n\n"
                f"Documents:\n{docs_text}\n\nFinal answer:"
            ),
        },
    ]


def format_documents(documents: list[Document], *, max_chars_per_doc: int = 1800) -> str:
    rendered = []
    for doc in documents:
        text = doc.text.strip().replace("\r\n", "\n")
        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc].rstrip() + "..."
        rendered.append(f"[{doc.doc_id}]\n{text}")
    return "\n\n".join(rendered)
