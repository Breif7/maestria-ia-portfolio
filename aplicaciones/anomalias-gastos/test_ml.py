"""ML analyses run on synthetic data only, so they never depend on a real ledger."""

import pytest

pytest.importorskip("sklearn")

import ml  # noqa: E402


def test_anomalies_find_planted_outliers():
    rows = ml.synthetic_movements(n=300, outliers=5)
    flagged = ml.spending_anomalies(rows, contamination=0.03)
    assert sum(1 for r in flagged[:5] if r.get("planted")) >= 4
    assert flagged == sorted(flagged, key=lambda r: -r["score"])


def test_anomalies_need_enough_rows_and_ignore_income():
    assert ml.spending_anomalies(ml.synthetic_movements(n=10, outliers=0)) == []
    income = [dict(r, kind="ingreso") for r in ml.synthetic_movements(n=100)]
    assert ml.spending_anomalies(income) == []


def test_categorizers_beat_baseline():
    result = ml.compare_categorizers(ml.synthetic_movements(n=300, outliers=0))
    assert result["classes"] == 5
    for name in ("naive_bayes", "logistic_regression"):
        assert result[name] > result["baseline_accuracy"] + 0.3


def test_categorizer_predicts_and_rejects_single_class():
    model = ml.train_categorizer(ml.synthetic_movements(n=300, outliers=0))
    assert model.predict(["UBER TRIP 555"])[0] == "transporte"
    one_class = [{"description": "x", "category": "a"}] * 10
    assert "error" in ml.compare_categorizers(one_class)
    with pytest.raises(ValueError):
        ml.train_categorizer(one_class)


def test_day_types_recover_three_profiles():
    groups = ml.day_types(ml.synthetic_days(n=120), k=3)
    assert all(35 <= g["size"] <= 45 for g in groups) and sum(g["size"] for g in groups) == 120
    assert min(g["sleep"] for g in groups) < 6 < max(g["sleep"] for g in groups)
    assert ml.day_types(ml.synthetic_days(n=5), k=3) == []


def test_report_on_empty_database():
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE movements(day, kind, amount, description, category);"
        "CREATE TABLE days(day, sleep, mood, energy, stress);"
        "CREATE TABLE habits(id, bad);"
        "CREATE TABLE habit_log(day, habit_id, n);"
    )
    text = ml.report(conn)
    assert "Movimientos: 0" in text and "hacen falta" in text
