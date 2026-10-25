from collections import Counter

from scripts.generate_factorial_data import build_plan


def test_factorial_plan_balances_every_weather_severity_cell() -> None:
    rows = build_plan("test", samples_per_cell=2, seed_offset=100)
    counts = Counter((row["weather"], row["severity"]) for row in rows)
    assert len(rows) == 18
    assert len(counts) == 9
    assert set(counts.values()) == {2}
    assert len({row["seed"] for row in rows}) == len(rows)
    assert len({row["scene_id"] for row in rows}) == len(rows)
