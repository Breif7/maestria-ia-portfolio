"""Small, explainable ML over the local ledger and day log (Maestría IA · Aprendizaje Automático).

Three read-only analyses, each tied to a course topic:
- `spending_anomalies`   Tema 11 (detección de anomalías): Isolation Forest over gastos.
- `compare_categorizers` Temas 3-4 (evaluación, Naive Bayes): which text model guesses a movement's category best.
- `day_types`            Tema 10 (k-medias): groups days by sleep, mood, energy, stress and habits done.

Nothing here writes to the database or calls the network. scikit-learn is optional (`pip install -e ".[ml]"`).
`synthetic_movements` / `synthetic_days` make fake data for tests and demos, so no real record leaves this PC.
"""

from __future__ import annotations

import datetime as dt
import math
import random
import sqlite3
from collections import Counter
from typing import Any

MIN_ANOMALY_ROWS = 20  # fewer gastos than this and "unusual" means nothing
MIN_PER_CATEGORY = 5  # cross-validation needs a few examples of each class


def _sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError('Instala el extra de ML: pip install -e ".[ml]"') from exc


# ---------------------------------------------------------------- anomalies (Tema 11)


def spending_anomalies(rows: list[dict[str, Any]], contamination: float = 0.05, seed: int = 0) -> list[dict]:
    """Gastos that look unusual for their category, weekday and size; most unusual first.

    Features: log(amount), the amount's robust z-score inside its category and the weekday. A 900 taco is odd
    even if 900 is a normal rent, so the per-category z-score carries most of the signal. Median/MAD instead of
    mean/std, so the anomalies themselves don't inflate the spread and hide each other."""
    _sklearn()
    import numpy as np
    from sklearn.ensemble import IsolationForest

    spend = [r for r in rows if r.get("kind") == "gasto" and r.get("amount")]
    if len(spend) < MIN_ANOMALY_ROWS:
        return []
    by_cat: dict[str, list[float]] = {}
    for r in spend:
        by_cat.setdefault(r.get("category") or "-", []).append(abs(r["amount"]))
    stats = {}
    for c, v in by_cat.items():
        med = float(np.median(v))
        stats[c] = (med, 1.4826 * float(np.median(np.abs(np.array(v) - med))) or 1.0)

    def features(r: dict[str, Any]) -> list[float]:
        mean, sd = stats[r.get("category") or "-"]
        day = dt.date.fromisoformat(str(r["day"])[:10])
        return [math.log1p(abs(r["amount"])), (abs(r["amount"]) - mean) / sd, day.weekday()]

    X = np.array([features(r) for r in spend])
    model = IsolationForest(contamination=contamination, random_state=seed).fit(X)
    scores = -model.score_samples(X)  # higher = more unusual
    flagged = np.flatnonzero(model.predict(X) == -1)
    out = [dict(spend[i], score=round(float(scores[i]), 3)) for i in flagged]
    return sorted(out, key=lambda r: -r["score"])


# ---------------------------------------------------------------- categorizer (Temas 3-4)


def _usable(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    rows = [r for r in rows if (r.get("description") or "").strip() and r.get("category")]
    counts = Counter(r["category"] for r in rows)
    keep = [r for r in rows if counts[r["category"]] >= MIN_PER_CATEGORY]
    return [r["description"] for r in keep], [r["category"] for r in keep]


def _pipelines() -> dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline

    # char n-grams survive bank-statement noise ("OXXO 1234 GDL") better than whole words
    vec = dict(analyzer="char_wb", ngram_range=(2, 4), lowercase=True, strip_accents="unicode")
    return {
        "naive_bayes": make_pipeline(TfidfVectorizer(**vec), MultinomialNB()),
        "logistic_regression": make_pipeline(TfidfVectorizer(**vec), LogisticRegression(max_iter=1000)),
    }


def compare_categorizers(rows: list[dict[str, Any]], folds: int = 5, seed: int = 0) -> dict[str, Any]:
    """Macro-F1 of each text model under stratified k-fold, plus a majority-class baseline to beat."""
    _sklearn()
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    texts, labels = _usable(rows)
    classes = Counter(labels)
    if len(classes) < 2:
        return {"error": f"hacen falta 2+ categorías con {MIN_PER_CATEGORY}+ movimientos descritos"}
    k = max(2, min(folds, min(classes.values())))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    result: dict[str, Any] = {"n": len(texts), "folds": k, "classes": len(classes),
                              "baseline_accuracy": round(classes.most_common(1)[0][1] / len(labels), 3)}
    for name, pipe in _pipelines().items():
        scores = cross_val_score(pipe, texts, labels, cv=cv, scoring="f1_macro")
        result[name] = round(float(scores.mean()), 3)
    return result


def train_categorizer(rows: list[dict[str, Any]], model: str = "logistic_regression"):
    """Fit on every usable row; `.predict(["OXXO GDL"])` then guesses a category."""
    _sklearn()
    texts, labels = _usable(rows)
    if len(set(labels)) < 2:
        raise ValueError("hacen falta al menos 2 categorías con datos")
    return _pipelines()[model].fit(texts, labels)


# ---------------------------------------------------------------- day types (Tema 10)

DAY_FEATURES = ("sleep", "mood", "energy", "stress", "habits")


def day_types(days: list[dict[str, Any]], k: int = 3, seed: int = 0) -> list[dict[str, Any]]:
    """K-means over standardized day features; each group comes back with its size and average values."""
    _sklearn()
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    full = [d for d in days if all(d.get(f) is not None for f in DAY_FEATURES)]
    if len(full) < k * 3:
        return []
    X = np.array([[float(d[f]) for f in DAY_FEATURES] for d in full])
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(StandardScaler().fit_transform(X))
    groups = []
    for c in range(k):
        member = X[labels == c]
        means = {f: round(float(v), 1) for f, v in zip(DAY_FEATURES, member.mean(0), strict=True)}
        groups.append({"size": int(len(member)), **means})
    return sorted(groups, key=lambda g: -g["size"])


# ---------------------------------------------------------------- data access


def load_movements(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    cur = conn.execute("SELECT day, kind, amount, description, category FROM movements")
    return [dict(zip(("day", "kind", "amount", "description", "category"), r, strict=True)) for r in cur]


def load_days(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    cur = conn.execute(
        "SELECT d.day, d.sleep, d.mood, d.energy, d.stress,"
        " (SELECT COALESCE(SUM(n), 0) FROM habit_log h JOIN habits b ON b.id = h.habit_id"
        "  WHERE h.day = d.day AND b.bad = 0) FROM days d"
    )
    return [dict(zip(("day",) + DAY_FEATURES, r, strict=True)) for r in cur]


def report(conn: sqlite3.Connection) -> str:
    """Plain-text summary for `python -m lifeos_rpg ml`."""
    moves, days = load_movements(conn), load_days(conn)
    lines = [f"Movimientos: {len(moves)} · Días registrados: {len(days)}", ""]
    odd = spending_anomalies(moves)
    empty = f"Gastos inusuales: hacen falta {MIN_ANOMALY_ROWS}+ gastos"
    lines.append(f"Gastos inusuales ({len(odd)}):" if odd else empty)
    lines += [f"  {r['day']}  {r['amount']:>10.2f}  {r.get('category') or '-':<14} {r.get('description') or ''}"
              for r in odd[:10]]
    lines += ["", f"Clasificador de categorías: {compare_categorizers(moves)}", ""]
    groups = day_types(days)
    lines.append("Tipos de día:" if groups else "Tipos de día: hacen falta más días completos")
    lines += [f"  {g['size']:>3} días · " + ", ".join(f"{f} {g[f]}" for f in DAY_FEATURES) for g in groups]
    return "\n".join(lines)


# ---------------------------------------------------------------- synthetic data

_SYNTH = {  # category -> (typical amount, spread, merchant words)
    "comida": (180, 60, ["OXXO", "TACOS EL GUERO", "SUPER AKI", "STARBUCKS", "TORTAS LUPITA"]),
    "transporte": (90, 30, ["UBER TRIP", "DIDI", "GASOLINERA PEMEX", "METROBUS RECARGA"]),
    "hogar": (4500, 300, ["RENTA DEPTO", "CFE LUZ", "TELMEX INTERNET", "AGUA SIAPA"]),
    "ocio": (350, 120, ["CINEPOLIS", "SPOTIFY", "NETFLIX", "BOLETOS CONCIERTO"]),
    "salud": (600, 250, ["FARMACIA GUADALAJARA", "DR CONSULTA", "LAB CHOPO"]),
}


def synthetic_movements(n: int = 300, seed: int = 0, outliers: int = 5) -> list[dict[str, Any]]:
    """Fake gastos with realistic merchants; the last `outliers` rows are planted anomalies (flag `planted`)."""
    rng = random.Random(seed)
    start, rows = dt.date(2026, 1, 1), []
    for i in range(n):
        cat = rng.choice(list(_SYNTH))
        mean, sd, names = _SYNTH[cat]
        rows.append({"day": (start + dt.timedelta(days=i % 300)).isoformat(), "kind": "gasto",
                     "amount": round(max(10.0, rng.gauss(mean, sd)), 2), "category": cat,
                     "description": f"{rng.choice(names)} {rng.randint(100, 999)}"})
    for i in range(outliers):  # a "comida" charge the size of rent
        rows.append({"day": (start + dt.timedelta(days=i * 40)).isoformat(), "kind": "gasto",
                     "amount": round(rng.uniform(5000, 9000), 2), "category": "comida",
                     "description": f"TACOS EL GUERO {rng.randint(100, 999)}", "planted": True})
    return rows


def synthetic_days(n: int = 120, seed: int = 0) -> list[dict[str, Any]]:
    """Three hidden day types: rested-productive, tired-stressed, weekend-social."""
    rng = random.Random(seed)
    profiles = [(7.5, 4, 4, 2, 4), (5.5, 2, 2, 4, 1), (8.5, 4, 3, 1, 2)]
    out = []
    for i in range(n):
        p = profiles[i % 3]
        scales = {f: min(5, max(1, round(rng.gauss(v, 0.4)))) for f, v in zip(DAY_FEATURES[1:4], p[1:4], strict=True)}
        out.append({"day": (dt.date(2026, 1, 1) + dt.timedelta(days=i)).isoformat(),
                    "sleep": round(rng.gauss(p[0], 0.4), 1), **scales,
                    "habits": max(0, round(rng.gauss(p[4], 0.5)))})
    return out
