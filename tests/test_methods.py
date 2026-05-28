from dirtyrag.llm_client import LLMClient
from dirtyrag.methods.crag_style_rag import CRAGStyleRAG
from dirtyrag.methods.evidenceboard_rag import EvidenceBoardRAG
from dirtyrag.methods.relevance_filter_rag import RelevanceFilterRAG
from dirtyrag.methods.vanilla_rag import VanillaRAG
from dirtyrag.schemas import Document, QAExample


def test_vanilla_rag_uses_documents_but_not_eval_labels() -> None:
    example = QAExample(
        qid="q1",
        dataset="unit",
        question="What is the population?",
        documents=[
            Document(
                doc_id="D1",
                text="The population is 3,559 people.",
                source_type="correct",
                source_answer="3,559 people",
            )
        ],
        gold_answers=["3,559 people"],
        wrong_answers=["10,000 people"],
    )
    method = VanillaRAG(LLMClient(provider="mock", model="mock"))

    result = method.run(example)

    assert result.method == "vanilla_rag"
    assert result.used_doc_ids == ["D1"]
    assert result.answer == "mock_answer"


def test_relevance_filter_rag_records_filter_metadata() -> None:
    example = build_example()
    method = RelevanceFilterRAG(LLMClient(provider="mock", model="mock"))

    result = method.run(example)

    assert result.method == "relevance_filter_rag"
    assert result.answer == "mock_answer"
    assert result.metadata["filtered_doc_ids"] == ["D1"]
    assert result.metadata["relevance_judgments"][0]["relevance"] == 2


def test_crag_style_rag_records_retrieval_metadata() -> None:
    example = build_example()
    method = CRAGStyleRAG(LLMClient(provider="mock", model="mock"))

    result = method.run(example)

    assert result.method == "crag_style_rag"
    assert result.answer == "mock_answer"
    assert result.metadata["retrieval_verdict"] == "correct"
    assert result.metadata["action"] == "answer"


def test_crag_style_rag_survives_retrieval_eval_failure() -> None:
    class BrokenJsonClient(LLMClient):
        def json_chat(self, messages, *, max_tokens=512):
            raise ValueError("bad json")

    example = build_example()
    method = CRAGStyleRAG(BrokenJsonClient(provider="mock", model="mock"))

    result = method.run(example)

    assert result.method == "crag_style_rag"
    assert result.answer == "mock_answer"
    assert result.metadata["retrieval_verdict"] == "insufficient"


def test_evidenceboard_rag_builds_board_metadata(tmp_path) -> None:
    example = build_example()
    method = EvidenceBoardRAG(LLMClient(provider="mock", model="mock"), run_dir=tmp_path)

    result = method.run(example)

    assert result.method == "evidenceboard_rag"
    assert result.answer == "3,559 people"
    assert result.evidence_board_path is not None
    assert result.metadata["num_cards"] == 1
    assert result.metadata["num_answer_clusters"] == 1


def build_example() -> QAExample:
    return QAExample(
        qid="q1",
        dataset="unit",
        question="What is the population?",
        documents=[
            Document(
                doc_id="D1",
                text="The population is 3,559 people.",
                source_type="correct",
                source_answer="3,559 people",
            )
        ],
        gold_answers=["3,559 people"],
        wrong_answers=["10,000 people"],
    )
