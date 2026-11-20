"""Estimator on synthetic boxes and fake slicer output in a temp dir; never reads real data/."""
import zipfile

import pytest

import estimate as est


def box_3mf(path, w, d, h):
    """Closed axis-aligned box mesh, outward-facing triangles."""
    v = [(x, y, z) for z in (0, h) for y in (0, d) for x in (0, w)]
    faces = [(0, 2, 1), (1, 2, 3), (4, 5, 6), (5, 7, 6), (0, 1, 4), (1, 5, 4),
             (2, 6, 3), (3, 6, 7), (0, 4, 2), (2, 4, 6), (1, 3, 5), (3, 7, 5)]
    xml = ('<model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"><resources><object id="1">'
           '<mesh><vertices>' + "".join(f'<vertex x="{a}" y="{b}" z="{c}"/>' for a, b, c in v)
           + '</vertices><triangles>' + "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in faces)
           + '</triangles></mesh></object></resources></model>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("3D/Objects/model.stl_1.model", xml)


def fake_preview(root, name, w, d, h):
    p = root / name
    p.mkdir()
    box_3mf(p / "preview.3mf", w, d, h)
    vol = w * d * h / 1000
    grams, hours = 3 + 0.3 * vol, 0.2 + 0.01 * vol + 0.02 * h
    hh, mm = int(hours), round((hours % 1) * 60)
    (p / "plate_1.gcode").write_text(
        f"; total filament used [g] = {grams:.2f}\n; estimated printing time (normal mode) = {hh}h {mm}m 0s\n")


BOXES = [(20, 20, 10), (40, 30, 20), (60, 40, 15), (30, 30, 50), (80, 50, 25), (50, 50, 60)]


def test_mesh_stats_box(tmp_path):
    box_3mf(tmp_path / "b.3mf", 20, 30, 40)
    s = est.mesh_stats(tmp_path / "b.3mf")
    assert s["volume_cm3"] == pytest.approx(24.0)
    assert s["height_mm"] == pytest.approx(40.0)


def test_parse_hours():
    assert est.parse_hours("2h 2m 35s") == pytest.approx(2 + 2 / 60 + 35 / 3600)
    assert est.parse_hours("1d 1h") == 25
    with pytest.raises(ValueError):
        est.parse_hours("soon")


def test_fit_recovers_linear_relation_and_beats_baseline(tmp_path):
    for i, box in enumerate(BOXES):
        fake_preview(tmp_path, f"p{i}", *box)
    (tmp_path / "broken").mkdir()  # no files: skipped, not fatal
    rows = est.load_history(tmp_path)
    assert len(rows) == 6
    model = est.fit(rows)
    for t in ("grams", "hours"):
        assert model[t]["loo_mae"] < model[t]["baseline_mae"]
    out = est.estimate(model, volume_cm3=50, height_mm=30)
    assert out["grams"] == pytest.approx(3 + 0.3 * 50, rel=0.05)
    assert out["hours"] == pytest.approx(0.2 + 0.5 + 0.6, rel=0.1)


def test_fit_needs_enough_rows():
    with pytest.raises(ValueError, match="need"):
        est.fit([{"volume_cm3": 1, "height_mm": 1, "grams": 1, "hours": 1}])


