"""Text analysis of episode scripts (Maestría IA · Procesamiento de Lenguaje Natural). Advisory only.

- `readability`: Fernández-Huerta index, the Spanish adaptation of Flesch (higher = easier; 60-70 is plain speech).
- `NaiveBayes` + `loo_accuracy`: multinomial Naive Bayes with Laplace smoothing, written out by hand so every step is
  visible, and evaluated with leave-one-out against the majority-class baseline. It learns to guess `hook_archetype`
  from the hook's words: if it can't beat the baseline, the archetypes are not distinct in the text.
- `hook_vs_views`: Spearman correlation between hook length/readability and TikTok views. With a few dozen videos it is
  a hint, not proof; the report prints n so nobody reads more into it.

Read-only: `python -m pipeline.text_analysis`. Never gates a script; the deterministic gates in script_gate.py decide.
"""
import csv
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_WORD = re.compile(r"[a-záéíóúüñ]+", re.I)
_VOWELS = re.compile(r"[aeiouáéíóúü]+", re.I)
_STRONG = set("aeoáéó")
STOPWORDS = set("el la los las un una unos unas de del al y o a en que por para con se su sus tu tus mi mis es lo le "
                "me te no si ya muy mas más pero como cuando qué cómo".split())


def syllables(word):
    """Spanish syllable count: vowel groups, plus one per hiatus (two strong vowels, or an accented í/ú)."""
    count = 0
    for group in _VOWELS.findall(word.lower()):
        count += 1
        for a, b in zip(group, group[1:]):
            if (a in _STRONG and b in _STRONG) or a in "íú" or b in "íú":
                count += 1
    return max(1, count)


def readability(text):
    """Fernández-Huerta: 206.84 - 0.60 * syllables per 100 words - 1.02 * sentences per 100 words."""
    words = _WORD.findall(text)
    if not words:
        return None
    sentences = max(1, len(re.findall(r"[.!?¿¡]+(?:\s|$)", text.strip() + " ")))
    per100 = 100 / len(words)
    return round(206.84 - 0.60 * sum(map(syllables, words)) * per100 - 1.02 * sentences * per100, 1)


def tokens(text):
    plain = unicodedata.normalize("NFD", text.lower())
    plain = "".join(c for c in plain if unicodedata.category(c) != "Mn")  # "pagarás" and "pagaras" are one word
    return [w for w in _WORD.findall(plain) if w not in STOPWORDS and not w.isdigit()]


class NaiveBayes:
    """Multinomial NB: P(class | words) ∝ P(class) · Π P(word | class), Laplace-smoothed, in log space."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, texts, labels):
        self.prior = Counter(labels)
        self.counts = defaultdict(Counter)
        for text, label in zip(texts, labels):
            self.counts[label].update(tokens(text))
        self.vocab = {w for c in self.counts.values() for w in c}
        self.total = {label: sum(c.values()) for label, c in self.counts.items()}
        return self

    def log_scores(self, text):
        n, v = sum(self.prior.values()), len(self.vocab)
        return {label: math.log(self.prior[label] / n)
                + sum(math.log((self.counts[label][w] + self.alpha) / (self.total[label] + self.alpha * v))
                      for w in tokens(text) if w in self.vocab)
                for label in self.prior}

    def predict(self, text):
        scores = self.log_scores(text)
        return max(sorted(scores), key=scores.get)  # sorted: ties break the same way every run


def loo_accuracy(texts, labels, min_per_class=3):
    """Leave-one-out accuracy of NaiveBayes vs always guessing the commonest class. Rare classes are dropped."""
    counts = Counter(labels)
    pairs = [(t, label) for t, label in zip(texts, labels) if counts[label] >= min_per_class]
    if len({label for _, label in pairs}) < 2:
        return None
    hits = 0
    for i, (text, label) in enumerate(pairs):
        rest = pairs[:i] + pairs[i + 1:]
        model = NaiveBayes().fit([t for t, _ in rest], [lab for _, lab in rest])
        hits += model.predict(text) == label
    kept = Counter(label for _, label in pairs)
    return {"n": len(pairs), "classes": len(kept), "accuracy": round(hits / len(pairs), 3),
            "baseline": round(kept.most_common(1)[0][1] / len(pairs), 3)}


def _ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):  # average rank for ties
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return ranks


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sd = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return round(cov / sd, 3) if sd else None


# ---------------------------------------------------------------- project data


def load_scripts(root=ROOT):
    out = []
    for path in sorted((root / "episodes").glob("*/script.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        segs = s.get("segments", [])
        out.append({"episode_id": s.get("episode_id"), "hook_archetype": s.get("hook_archetype"),
                    "hook": " ".join(g.get("text", "") for g in segs if g.get("phase") == "hook"),
                    "text": " ".join(g.get("text", "") for g in segs)})
    return out


def latest_views(root=ROOT):
    """episode_id -> views from its most recent snapshot row."""
    path, best = root / "data" / "tiktok_snapshots.csv", {}
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ep, day = row.get("episode_id"), row.get("snapshot_date", "")
            if ep and row.get("views", "").isdigit() and day >= best.get(ep, ("", 0))[0]:
                best[ep] = (day, int(row["views"]))
    return {ep: v for ep, (_, v) in best.items()}


def hook_vs_views(scripts, views):
    rows = [(s, views[s["episode_id"]]) for s in scripts if s["episode_id"] in views and s["hook"]]
    if not rows:
        return {"n": 0}
    v = [r[1] for r in rows]
    return {"n": len(rows),
            "hook_words": spearman([len(_WORD.findall(s["hook"])) for s, _ in rows], v),
            "hook_readability": spearman([readability(s["hook"]) or 0 for s, _ in rows], v)}


def report(root=ROOT):
    scripts = load_scripts(root)
    if not scripts:
        return "No hay guiones en episodes/*/script.json"
    scores = [(s["episode_id"], readability(s["text"])) for s in scripts if s["text"]]
    lines = [f"Guiones: {len(scripts)}", "", "Legibilidad Fernández-Huerta (60-70 = habla sencilla):"]
    lines += [f"  {ep}: {score}" for ep, score in sorted(scores, key=lambda r: r[1] or 0)[:5]]
    lines.append("  (los 5 más difíciles)")
    labeled = [s for s in scripts if s["hook"] and s["hook_archetype"]]
    lines += ["", f"Naive Bayes: arquetipo de gancho desde el texto (leave-one-out): "
                  f"{loo_accuracy([s['hook'] for s in labeled], [s['hook_archetype'] for s in labeled])}"]
    lines.append(f"Gancho vs vistas en TikTok (Spearman): {hook_vs_views(scripts, latest_views(root))}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
