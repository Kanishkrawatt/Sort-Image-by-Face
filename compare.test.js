import assert from "node:assert/strict";
import test from "node:test";
import { compareResult } from "./lib/api.js";

const vector = (axis, magnitude) => Object.assign(new Array(128).fill(0), { [axis]: magnitude });
const face = (url, axis, magnitude, score) => ({
  url, score, box: { x: 0, y: 0, width: 100, height: 100 },
  descriptor: vector(axis, magnitude),
});
const result = (faces, failed = []) => ({ faces, failed, noFaces: [] });

test("two photos of the same face match, with the distance reported", () => {
  const { status, body } = compareResult(
    result([face("a.jpg", 0, 1, 0.9), face("b.jpg", 0, 1, 0.9)]), "a.jpg", "b.jpg", 0.65);
  assert.equal(status, 200);
  assert.equal(body.match, true);
  assert.equal(body.distance, 0);
  assert.equal(body.threshold, 0.65);
});

test("faces far apart do not match", () => {
  const { status, body } = compareResult(
    result([face("a.jpg", 0, 5, 0.9), face("b.jpg", 1, 5, 0.9)]), "a.jpg", "b.jpg", 0.65);
  assert.equal(status, 200);
  assert.equal(body.match, false);
  assert.ok(body.distance > 0.65, `expected a large distance, got ${body.distance}`);
});

test("the threshold decides, and is echoed back", () => {
  const pair = result([face("a.jpg", 0, 0, 0.9), face("b.jpg", 0, 0.5, 0.9)]);
  assert.equal(compareResult(pair, "a.jpg", "b.jpg", 0.6).body.match, true);
  assert.equal(compareResult(pair, "a.jpg", "b.jpg", 0.4).body.match, false);
});

test("a photo with no face is refused, saying which one", () => {
  const { status, body } = compareResult(
    result([face("a.jpg", 0, 1, 0.9)]), "a.jpg", "b.jpg", 0.65);
  assert.equal(status, 422);
  assert.match(body.error, /No face found in b/);
  assert.deepEqual(body.facesFound, { a: 1, b: 0 });
});

test("neither photo having a face names both", () => {
  const { body } = compareResult(result([]), "a.jpg", "b.jpg", 0.65);
  assert.match(body.error, /a and b/);
});

test("the clearest face is used, and the ambiguity is surfaced", () => {
  const { body } = compareResult(
    result([
      face("a.jpg", 0, 5, 0.40), // a bystander
      face("a.jpg", 1, 1, 0.99), // the subject
      face("b.jpg", 1, 1, 0.95),
    ]), "a.jpg", "b.jpg", 0.65);

  assert.equal(body.distance, 0, "should compare the two highest-scoring faces");
  assert.equal(body.a.faces, 2, "caller is told the photo held more than one face");
  assert.equal(body.a.score, 0.99);
});

test("an unreadable image reports the failure rather than a verdict", () => {
  const { status, body } = compareResult(
    result([], [{ url: "a.jpg", error: "responded 404" }]), "a.jpg", "b.jpg", 0.65);
  assert.equal(status, 422);
  assert.match(body.error, /Could not read/);
  assert.equal(body.failed[0].error, "responded 404");
});
