from scripts.benchmark_video_modes import MEAN_METRICS, aggregate_mode, render_report


def _run(value: float) -> dict[str, float]:
    return {metric: value for metric in MEAN_METRICS}


def test_aggregate_mode_reports_mean_and_dispersion() -> None:
    row = aggregate_mode("risk_adaptive", [_run(1.0), _run(3.0)])

    assert row["mode"] == "risk_adaptive"
    assert row["repeats"] == 2
    assert row["achieved_fps"] == 2.0
    assert row["achieved_fps_std"] == 1.0
    assert row["candidate_on_baseline_fallback"] == 2.0


def test_render_report_includes_repeats_device_and_uncertainty() -> None:
    row = aggregate_mode("safety_every_frame", [_run(10.0), _run(12.0)])

    report = render_report(
        [row],
        frames=30,
        max_side=640,
        repeats=2,
        system={
            "device": "Test GPU",
            "torch": "2.x",
            "cuda": "12.x",
            "python": "3.x",
        },
    )

    assert "2 轮顺序轮换" in report
    assert "Test GPU" in report
    assert "11.00 ± 1.00" in report
