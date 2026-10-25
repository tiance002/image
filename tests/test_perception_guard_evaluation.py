from scripts.evaluate_perception_guard import select_guarded_records


def record(name: str, confidence: float | None) -> dict:
    predictions = {}
    if confidence is not None:
        predictions = {
            "2": [{"confidence": confidence, "box": [10.0, 10.0, 30.0, 30.0]}]
        }
    return {
        "file_name": name,
        "weather": "fog",
        "ground_truth": {"2": [[10.0, 10.0, 30.0, 30.0]]},
        "predictions": predictions,
        "true_positives": int(confidence is not None),
        "false_positives": 0,
        "false_negatives": int(confidence is None),
    }


def test_select_guarded_records_falls_back_to_reference() -> None:
    reference = [record("frame.png", 0.9)]
    candidate = [record("frame.png", None)]

    selected, decisions = select_guarded_records(reference, candidate, threshold=0.7)

    assert selected[0]["predictions"] == reference[0]["predictions"]
    assert not decisions[0]["accept_candidate"]


def test_select_guarded_records_requires_aligned_images() -> None:
    try:
        select_guarded_records([record("a.png", 0.8)], [record("b.png", 0.8)], 0.7)
    except ValueError as error:
        assert "order" in str(error)
    else:
        raise AssertionError("Expected misaligned records to fail")
