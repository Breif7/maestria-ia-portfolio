// Run with: npm test   (node --test; Node 23.6+ runs TypeScript directly). Synthetic merchants only.
import assert from "node:assert/strict";
import { test } from "node:test";

import { confidentGuess, features, looAccuracy, predict, train } from "./merchant-classifier.ts";

const history = [
  ...["OXXO #1234 GDL", "Oxxo Chapultepec", "OXXO 5521", "Costco Zapopan", "COSTCO WHOLESALE 0412", "Costco Plaza"]
    .map((merchant) => ({ merchant, category: "Food" })),
  ...["UBER *TRIP", "Uber Trip Help", "DIDI MOBILITY", "Didi Rides"]
    .map((merchant) => ({ merchant, category: "Transport" })),
  ...["CINEPOLIS VIP", "Cinepolis Andares", "NETFLIX.COM", "Netflix MX"]
    .map((merchant) => ({ merchant, category: "Entertainment" })),
];

test("features are accent- and case-insensitive trigrams", () => {
  assert.deepEqual(features("Café"), features("CAFE"));
  assert.deepEqual(features("12345"), []);
});

test("learns this user's habits", () => {
  const m = train(history);
  assert.equal(predict(m, "OXXO 9999 ZAPOPAN")?.category, "Food");
  assert.equal(predict(m, "uber trip")?.category, "Transport");
  assert.equal(predict(m, "Costco Guadalajara")?.category, "Food"); // generic knowledge would say Shopping
  assert.equal(predict(m, "qqqq"), null); // nothing it has seen
  assert.equal(predict(train([]), "OXXO"), null);
});

test("only confident, well-supported guesses override Claude", () => {
  const m = train(history);
  const g = confidentGuess(m, "OXXO 7777");
  assert.ok(g && g.category === "Food" && g.probability >= 0.8 && g.support >= 3, JSON.stringify(g));
  const thin = train([{ merchant: "Starbucks", category: "Food" }, { merchant: "Starbucks Reserve", category: "Food" }]);
  assert.equal(confidentGuess(thin, "Starbucks"), null); // 2 examples < MIN_SUPPORT
});

test("leave-one-out beats the majority baseline", () => {
  const r = looAccuracy(history)!;
  assert.equal(r.n, history.length);
  assert.ok(r.accuracy > r.baseline, JSON.stringify(r));
  assert.equal(looAccuracy([{ merchant: "x", category: "a" }]), null);
});
