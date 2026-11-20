"""Estimate grams and print hours from a mesh, before slicing (Maestría IA · Aprendizaje Automático).

Learns from your own sliced previews in data/slicer-previews/*/ (preview.3mf + plate_1.gcode):
- grams ~ volume_cm3                  (filament is volume x density x fill, so close to linear)
- hours ~ volume_cm3 + height_mm      (height sets the layer count, each layer costs fixed travel time)

Ridge regression written out by hand (no numpy), with the penalty chosen by leave-one-out error (Tema 2 regression
metrics, Tema 13 hyperparameter search). The report always shows the mean-only baseline, so a model that does not
beat "always guess the average" is visible. Read-only; nothing here slices, sends or prices anything by itself.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

LAMBDAS = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)
MIN_SAMPLES = 4


def mesh_stats(path: Path) -> dict:
    """Volume (cm3) and height (mm) of every mesh in a .3mf, via the signed-tetrahedron sum.

    shortcut: ignores the 3MF build transform (assumes the slicer kept the part upright, as the daily pipeline does);
    read <build><item transform> if parts start arriving rotated."""
    volume, zs = 0.0, []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not (name.startswith("3D/") and name.endswith(".model")):
                continue
            root = ET.fromstring(z.read(name))
            for mesh in root.iter():
                if not mesh.tag.endswith("mesh"):
                    continue
                verts = [(float(v.get("x")), float(v.get("y")), float(v.get("z")))
                         for v in mesh.iter() if v.tag.endswith("vertex")]
                zs += [v[2] for v in verts]
                for t in mesh.iter():
                    if t.tag.endswith("triangle"):
                        a, b, c = (verts[int(t.get(k))] for k in ("v1", "v2", "v3"))
                        volume += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                                   + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6
    if not zs:
        raise ValueError(f"no mesh in {path}")
    return {"volume_cm3": abs(volume) / 1000, "height_mm": max(zs) - min(zs)}


def parse_hours(text: str) -> float:
    """'1d 2h 3m 4s' -> hours."""
    units = {"d": 24, "h": 1, "m": 1 / 60, "s": 1 / 3600}
    parts = re.findall(r"(\d+(?:\.\d+)?)\s*([dhms])", text)
    if not parts:
        raise ValueError(f"unreadable duration: {text!r}")
    return sum(float(n) * units[u] for n, u in parts)


def gcode_stats(path: Path) -> dict:
    """Grams and normal-mode hours from the slicer's G-code header."""
    head = path.read_text(encoding="utf-8", errors="replace")
    grams = re.search(r"total filament used \[g\]\s*=\s*([\d.]+)", head)
    time = re.search(r"estimated printing time \(normal mode\)\s*=\s*([^\n]+)", head) or \
        re.search(r"estimated printing time[^=\n]*=\s*([^\n]+)", head)
    if not grams or not time:
        raise ValueError(f"no grams/time in {path}")
    return {"grams": float(grams.group(1)), "hours": parse_hours(time.group(1))}


def load_history(previews: Path) -> list[dict]:
    """One row per preview folder that has both files and parses; broken folders are skipped."""
    rows = []
    for d in sorted(p for p in previews.glob("*") if p.is_dir()):
        try:
            rows.append({**mesh_stats(d / "preview.3mf"), **gcode_stats(d / "plate_1.gcode"), "id": d.name})
        except (OSError, ValueError, KeyError, zipfile.BadZipFile, ET.ParseError):
            continue
    return rows


# ---------------------------------------------------------------- ridge regression


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            raise ValueError("singular system: features are collinear or constant")
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col])]
    return [m[i][n] / m[i][i] for i in range(n)]


def ridge(X: list[list[float]], y: list[float], lam: float) -> list[float]:
    """Coefficients [intercept, b1, ...]. Features are standardized; the intercept is not penalized."""
    k = len(X[0])
    mu = [sum(r[j] for r in X) / len(X) for j in range(k)]
    sd = [(sum((r[j] - mu[j]) ** 2 for r in X) / len(X)) ** 0.5 or 1.0 for j in range(k)]
    Z = [[1.0] + [(r[j] - mu[j]) / sd[j] for j in range(k)] for r in X]
    ata = [[sum(z[i] * z[j] for z in Z) + (lam if i == j and i > 0 else 0.0) for j in range(k + 1)]
           for i in range(k + 1)]
    beta = _solve(ata, [sum(z[i] * t for z, t in zip(Z, y)) for i in range(k + 1)])
    coef = [beta[j + 1] / sd[j] for j in range(k)]  # back to raw units
    return [beta[0] - sum(c * m for c, m in zip(coef, mu))] + coef


def predict(coef: list[float], x: list[float]) -> float:
    return coef[0] + sum(c * v for c, v in zip(coef[1:], x))


def loo_mae(X, y, lam) -> float:
    errs = [abs(predict(ridge(X[:i] + X[i + 1:], y[:i] + y[i + 1:], lam), X[i]) - y[i]) for i in range(len(y))]
    return sum(errs) / len(errs)


FEATURES = {"grams": ("volume_cm3",), "hours": ("volume_cm3", "height_mm")}


def fit(rows: list[dict]) -> dict:
    """Per target: best lambda by leave-one-out MAE, its coefficients, and the mean-only baseline MAE."""
    if len(rows) < MIN_SAMPLES:
        raise ValueError(f"need {MIN_SAMPLES}+ sliced previews, found {len(rows)}")
    model = {"n": len(rows)}
    for target, feats in FEATURES.items():
        X = [[r[f] for f in feats] for r in rows]
        y = [r[target] for r in rows]
        mae, lam = min((loo_mae(X, y, lam), lam) for lam in LAMBDAS)
        base = [abs(sum(y[:i] + y[i + 1:]) / (len(y) - 1) - y[i]) for i in range(len(y))]
        model[target] = {"features": feats, "lambda": lam, "coef": ridge(X, y, lam),
                         "loo_mae": round(mae, 3), "baseline_mae": round(sum(base) / len(base), 3)}
    return model


def estimate(model: dict, volume_cm3: float, height_mm: float) -> dict:
    values = {"volume_cm3": volume_cm3, "height_mm": height_mm}
    return {t: round(max(0.0, predict(model[t]["coef"], [values[f] for f in model[t]["features"]])), 2)
            for t in FEATURES}
