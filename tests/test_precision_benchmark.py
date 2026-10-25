from scripts.benchmark_precision import compare_selection_sequences


def test_compare_selection_sequences_reports_safety_changes() -> None:
    fp32 = [
        {"guard_selection": "input_fallback"},
        {"guard_selection": "temporal_candidate"},
        {"guard_selection": "input_fallback"},
    ]
    fp16 = [
        {"guard_selection": "temporal_candidate"},
        {"guard_selection": "temporal_candidate"},
        {"guard_selection": "input_fallback"},
    ]

    comparison = compare_selection_sequences(fp32, fp16)

    assert comparison == {
        "compared_frames": 3,
        "selection_disagreements": 1,
        "fp16_candidate_on_fp32_fallback": 1,
    }
