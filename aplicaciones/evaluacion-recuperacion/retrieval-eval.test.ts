// Run with: npm test   (node --test). Synthetic notes only; no network, database or env vars.
import assert from "node:assert/strict";
import { test } from "node:test";

import { bm25, dense, evaluate, recallAtK, reciprocalRank, tokenize } from "./retrieval-eval.ts";

const docs = [
  { id: "gym", text: "Rutina de gimnasio: sentadilla, peso muerto y press de banca tres veces por semana." },
  { id: "money", text: "Presupuesto mensual: renta, comida y ahorro del 20% de la quincena." },
  { id: "book", text: "Notes on Atomic Habits: make good habits obvious, attractive, easy and satisfying." },
  { id: "trip", text: "Viaje a Oaxaca: hotel en el centro, mezcal y tlayudas." },
];

test("metrics", () => {
  assert.equal(recallAtK(["a", "b", "c"], ["c", "z"], 2), 0);
  assert.equal(recallAtK(["a", "b", "c"], ["c", "z"], 3), 0.5);
  assert.equal(recallAtK(["a"], [], 1), 0);
  assert.equal(reciprocalRank(["a", "b", "c"], ["b"]), 0.5);
  assert.equal(reciprocalRank(["a"], ["z"]), 0);
});

test("tokenize drops accents and stopwords", () => {
  assert.deepEqual(tokenize("La Ración de Ahorro"), ["racion", "ahorro"]);
});

test("bm25 ranks the matching note first", async () => {
  const r = bm25(docs);
  assert.equal((await r("ahorro de la quincena", 2))[0], "money");
  assert.equal((await r("habits atomic", 2))[0], "book");
  assert.deepEqual(await r("qqqq", 2), []);
  const qs = [{ q: "sentadilla", relevant: ["gym"] }, { q: "mezcal oaxaca", relevant: ["trip"] }];
  assert.deepEqual(await evaluate(r, qs, 1), { n: 2, k: 1, recall: 1, mrr: 1 });
});

test("dense retriever uses cosine similarity", async () => {
  const r = dense(["x", "y"], [[1, 0], [0, 1]], async () => [0.1, 0.9]);
  assert.deepEqual(await r("anything", 2), ["y", "x"]);
});

