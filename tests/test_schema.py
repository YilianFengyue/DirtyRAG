from dirtyrag.schemas import Document, QAExample


def test_qa_example_schema_accepts_minimal_example() -> None:
    example = QAExample(
        qid="x1",
        dataset="unit",
        question="What is the answer?",
        documents=[Document(doc_id="D1", text="The answer is 42.")],
    )

    assert example.qid == "x1"
    assert example.documents[0].doc_id == "D1"
    assert example.gold_answers == []

