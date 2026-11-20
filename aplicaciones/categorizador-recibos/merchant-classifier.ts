// Learns each user's own merchant -> category habits from their confirmed transactions (Maestría IA, Naive Bayes).
// Claude guesses a category from the receipt image with generic knowledge; your history knows that *your* "Costco"
// is groceries, not shopping. When the history is confident, its guess wins and Claude's is kept for reference.
// No dependencies and no network: pure functions, so they are unit-tested in tests/ with node --test.

export type Example = { merchant: string; category: string };
export type Guess = { category: string; probability: number; support: number };

const MIN_SUPPORT = 3; // a category needs this many past examples before it can override Claude
export const MIN_PROBABILITY = 0.8;

// Character trigrams of the normalized merchant: "OXXO #1234 GDL" and "Oxxo Gdl" share most of them.
export function features(merchant: string): string[] {
  const plain = merchant.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
  const s = ` ${plain.replace(/[^a-z ]+/g, " ").replace(/\s+/g, " ").trim()} `;
  if (!s.trim()) return [];
  const grams: string[] = [];
  for (let i = 0; i + 3 <= s.length; i++) grams.push(s.slice(i, i + 3));
  return grams;
}

export type Model = {
  prior: Map<string, number>;
  counts: Map<string, Map<string, number>>;
  totals: Map<string, number>;
  vocab: Set<string>;
  n: number;
};

export function train(examples: Example[]): Model {
  const m: Model = { prior: new Map(), counts: new Map(), totals: new Map(), vocab: new Set(), n: 0 };
  for (const { merchant, category } of examples) {
    const f = features(merchant);
    if (!f.length || !category) continue;
    m.n++;
    m.prior.set(category, (m.prior.get(category) ?? 0) + 1);
    const c = m.counts.get(category) ?? new Map<string, number>();
    for (const g of f) {
      c.set(g, (c.get(g) ?? 0) + 1);
      m.vocab.add(g);
    }
    m.counts.set(category, c);
    m.totals.set(category, (m.totals.get(category) ?? 0) + f.length);
  }
  return m;
}

// Multinomial Naive Bayes with Laplace smoothing; log scores turned into probabilities with log-sum-exp.
export function predict(m: Model, merchant: string): Guess | null {
  const f = features(merchant).filter((g) => m.vocab.has(g));
  if (!m.n || !f.length) return null;
  const v = m.vocab.size;
  const scores = [...m.prior].map(([cat, count]) => {
    const c = m.counts.get(cat)!;
    const total = m.totals.get(cat)!;
    let s = Math.log(count / m.n);
    for (const g of f) s += Math.log(((c.get(g) ?? 0) + 1) / (total + v));
    return { cat, s, count };
  });
  const max = Math.max(...scores.map((x) => x.s));
  const z = scores.reduce((a, x) => a + Math.exp(x.s - max), 0);
  const best = scores.reduce((a, b) => (b.s > a.s || (b.s === a.s && b.cat < a.cat) ? b : a));
  return { category: best.cat, probability: Math.exp(best.s - max) / z, support: best.count };
}

// The history's guess only when it is confident and backed by enough past examples; otherwise null (keep Claude's).
export function confidentGuess(m: Model, merchant: string): Guess | null {
  const g = predict(m, merchant);
  return g && g.probability >= MIN_PROBABILITY && g.support >= MIN_SUPPORT ? g : null;
}

// Leave-one-out accuracy of the history model vs always guessing the commonest category.
export function looAccuracy(examples: Example[]): { n: number; accuracy: number; baseline: number } | null {
  const usable = examples.filter((e) => features(e.merchant).length && e.category);
  if (usable.length < 2) return null;
  let hits = 0;
  usable.forEach((e, i) => {
    const g = predict(train(usable.filter((_, j) => j !== i)), e.merchant);
    if (g?.category === e.category) hits++;
  });
  const counts = new Map<string, number>();
  for (const e of usable) counts.set(e.category, (counts.get(e.category) ?? 0) + 1);
  return { n: usable.length, accuracy: hits / usable.length, baseline: Math.max(...counts.values()) / usable.length };
}
