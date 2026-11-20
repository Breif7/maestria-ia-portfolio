"""Text analysis on fixed sentences and a throwaway project folder; never reads real episodes."""
import json

import text_analysis as ta


def test_syllables_spanish():
    assert ta.syllables("casa") == 2
    assert ta.syllables("tarjeta") == 3
    assert ta.syllables("deuda") == 2          # diphthong eu
    assert ta.syllables("poeta") == 3          # hiatus oe
    assert ta.syllables("día") == 2            # accented í breaks the diphthong


def test_readability_orders_plain_above_dense():
    plain = "Tu tarjeta cobra mucho. Paga más que el mínimo. Así baja la deuda."
    dense = ("La capitalización compuesta de intereses moratorios incrementa significativamente "
             "el costo financiero acumulado correspondiente al endeudamiento revolvente.")
    assert ta.readability(plain) > ta.readability(dense)
    assert ta.readability("") is None


def test_naive_bayes_learns_and_loo_beats_baseline():
    questions = [f"¿Sabes cuánto pagas de {w}?" for w in ("luz", "renta", "tarjeta", "agua", "gas", "seguro")]
    numbers = [f"{n} pesos al mes se van en {w}" for n, w in zip(range(100, 700, 100),
                                                                 ("luz", "renta", "tarjeta", "agua", "gas", "seguro"))]
    texts, labels = questions + numbers, ["pregunta"] * 6 + ["numero"] * 6
    model = ta.NaiveBayes().fit(texts, labels)
    assert model.predict("¿Sabes cuánto pagas de internet?") == "pregunta"
    result = ta.loo_accuracy(texts, labels)
    assert result["accuracy"] > result["baseline"] == 0.5
    assert ta.loo_accuracy(["a", "b"], ["x", "y"]) is None  # each class below min_per_class


def test_spearman():
    assert ta.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert ta.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert ta.spearman([1, 2], [1, 2]) is None
    assert ta.spearman([1, 1, 1], [1, 2, 3]) is None


def test_report_on_fake_project(tmp_path):
    for i, (arch, hook) in enumerate([("pregunta", "¿Cuánto pagas?"), ("numero", "Cinco mil pesos")] * 3):
        ep = tmp_path / "episodes" / f"EP{i:03d}"
        ep.mkdir(parents=True)
        (ep / "script.json").write_text(json.dumps({
            "episode_id": f"EP{i:03d}", "hook_archetype": arch,
            "segments": [{"phase": "hook", "text": hook}, {"phase": "body", "text": "Paga a tiempo."}]}),
            encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "tiktok_snapshots.csv").write_text(
        "snapshot_date,episode_id,views\n2026-11-01,EP000,10\n2026-11-02,EP000,50\n2026-11-02,EP001,20\n"
        "2026-11-02,EP002,30\n", encoding="utf-8")
    assert ta.latest_views(tmp_path) == {"EP000": 50, "EP001": 20, "EP002": 30}
    text = ta.report(tmp_path)
    assert "Guiones: 6" in text and "'n': 3" in text
    assert ta.report(tmp_path / "empty").startswith("No hay guiones")
