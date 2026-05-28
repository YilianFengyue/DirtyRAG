from dirtyrag.evaluation.metrics import score_prediction, summarize_metrics


def test_score_prediction_tracks_gold_and_wrong_leakage() -> None:
    example = {
        "qid": "q1",
        "gold_answers": ["3,559 people"],
        "wrong_answers": ["10,000 people"],
    }
    prediction = {
        "qid": "q1",
        "method": "vanilla_rag",
        "answer": "The answer is 3,559 people.",
    }

    row = score_prediction(example, prediction)

    assert row["strict_success"] == 1.0
    assert row["gold_coverage"] == 1.0
    assert row["wrong_leakage"] == 0.0


def test_summarize_metrics_groups_by_method() -> None:
    summary = summarize_metrics(
        [
            {"method": "a", "strict_success": 1, "gold_coverage": 1, "wrong_leakage": 0},
            {"method": "a", "strict_success": 0, "gold_coverage": 0, "wrong_leakage": 1},
        ]
    )

    assert summary[0]["method"] == "a"
    assert summary[0]["num_examples"] == 2
    assert summary[0]["strict_success"] == 0.5

