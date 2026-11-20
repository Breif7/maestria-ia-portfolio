// Measures how well retrieval finds the right notes (Maestría IA · Procesamiento de Lenguaje Natural).
// The assistant can only answer from the 8 chunks match_chunks() returns, so retrieval quality caps answer quality.
// recall@k and MRR over a small labeled question set; BM25 is the classic lexical baseline the embeddings must beat.
// Pure functions, no network: unit-tested in tests/, used by scripts/eval-retrieval.ts.

export type Doc = { id: string; text: string };
export type Question = { q: string; relevant: string[] };
export type Retriever = (query: string, k: number) => Promise<string[]> | string[];

/** Share of relevant docs that appear in the top k. */
export function recallAtK(ranked: string[], relevant: string[], k: number): number {
  if (!relevant.length) return 0;
  const top = new Set(ranked.slice(0, k));
  return relevant.filter((id) => top.has(id)).length / relevant.length;
}

/** 1 / rank of the first relevant doc (0 if none in the list). */
export function reciprocalRank(ranked: string[], relevant: string[]): number {
  const i = ranked.findIndex((id) => relevant.includes(id));
  return i < 0 ? 0 : 1 / (i + 1);
}

export async function evaluate(retrieve: Retriever, questions: Question[], k = 8) {
  let recall = 0;
  let mrr = 0;
  for (const { q, relevant } of questions) {
    const ranked = await retrieve(q, k);
    recall += recallAtK(ranked, relevant, k);
    mrr += reciprocalRank(ranked, relevant);
  }
  const n = questions.length || 1;
  return { n: questions.length, k, recall: recall / n, mrr: mrr / n };
}

const STOP = new Set(
  ("the a an of to in on for and or is are was be it this that with as at by from do i my what how " +
   "de la el los las un una y o en que por para con se su es lo del al").split(" ")
);

export function tokenize(text: string): string[] {
  const plain = text.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
  return plain.match(/[a-z0-9]+/g)?.filter((t) => !STOP.has(t)) ?? [];
}

/** Okapi BM25 (k1 = 1.2, b = 0.75). Returns a retriever over the given docs. */
export function bm25(docs: Doc[], k1 = 1.2, b = 0.75): Retriever {
  const toks = docs.map((d) => tokenize(d.text));
  const avg = toks.reduce((a, t) => a + t.length, 0) / (toks.length || 1);
  const df = new Map<string, number>();
  for (const t of toks) for (const w of new Set(t)) df.set(w, (df.get(w) ?? 0) + 1);
  const idf = (w: string) => Math.log(1 + (docs.length - (df.get(w) ?? 0) + 0.5) / ((df.get(w) ?? 0) + 0.5));
  return (query, k) => {
    const q = tokenize(query);
    return docs
      .map((d, i) => {
        const tf = new Map<string, number>();
        for (const w of toks[i]) tf.set(w, (tf.get(w) ?? 0) + 1);
        const score = q.reduce((s, w) => {
          const f = tf.get(w) ?? 0;
          return s + (f ? (idf(w) * f * (k1 + 1)) / (f + k1 * (1 - b + (b * toks[i].length) / avg)) : 0);
        }, 0);
        return { id: d.id, score };
      })
      .filter((x) => x.score > 0)
      .sort((x, y) => y.score - x.score || x.id.localeCompare(y.id))
      .slice(0, k)
      .map((x) => x.id);
  };
}

/** Dense retriever from precomputed vectors (cosine), mirroring what match_chunks() does in Postgres. */
export function dense(ids: string[], vectors: number[][], embedQuery: (q: string) => Promise<number[]>): Retriever {
  const norm = (v: number[]) => Math.sqrt(v.reduce((a, x) => a + x * x, 0)) || 1;
  const docNorms = vectors.map(norm);
  return async (query, k) => {
    const qv = await embedQuery(query);
    const qn = norm(qv);
    return vectors
      .map((v, i) => ({ id: ids[i], s: v.reduce((a, x, j) => a + x * qv[j], 0) / (docNorms[i] * qn) }))
      .sort((x, y) => y.s - x.s)
      .slice(0, k)
      .map((x) => x.id);
  };
}
