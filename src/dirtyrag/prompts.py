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


def format_documents(documents: list[Document], *, max_chars_per_doc: int = 1800) -> str:
    rendered = []
    for doc in documents:
        text = doc.text.strip().replace("\r\n", "\n")
        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc].rstrip() + "..."
        rendered.append(f"[{doc.doc_id}]\n{text}")
    return "\n\n".join(rendered)

