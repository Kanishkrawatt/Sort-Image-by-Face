import assert from "node:assert/strict";
import test from "node:test";
import { clusterFaces, photosIn, euclidean } from "./public/cluster.js";

const DIMS = 128;

/** A descriptor that is `spread` away from the origin in one axis. */
function descriptor(axis, spread = 0) {
  const vector = new Array(DIMS).fill(0);
  vector[axis] = spread;
  return vector;
}

/** Nudge a descriptor slightly, as two photos of one person would differ. */
function jitter(vector, amount) {
  const copy = Array.from(vector);
  copy[DIMS - 1] += amount;
  return copy;
}

test("identical descriptors land in one cluster", () => {
  const alice = descriptor(0, 1);
  const clusters = clusterFaces([
    { fileName: "a.jpg", descriptor: alice },
    { fileName: "b.jpg", descriptor: alice },
    { fileName: "c.jpg", descriptor: alice },
  ]);

  assert.equal(clusters.length, 1);
  assert.deepEqual(photosIn(clusters[0]), ["a.jpg", "b.jpg", "c.jpg"]);
});

test("distant descriptors stay in separate clusters", () => {
  const clusters = clusterFaces([
    { fileName: "a.jpg", descriptor: descriptor(0, 5) },
    { fileName: "b.jpg", descriptor: descriptor(1, 5) },
  ]);

  assert.equal(clusters.length, 2);
  assert.equal(clusters[0].faces.length, 1);
  assert.equal(clusters[1].faces.length, 1);
});

test("nearly identical descriptors are treated as the same person", () => {
  const alice = descriptor(0, 1);
  const clusters = clusterFaces([
    { fileName: "a.jpg", descriptor: alice },
    { fileName: "b.jpg", descriptor: jitter(alice, 0.1) },
  ]);

  assert.equal(clusters.length, 1);
});

test("a photo with two different faces appears in two clusters", () => {
  const alice = descriptor(0, 5);
  const bob = descriptor(1, 5);

  const clusters = clusterFaces([
    { fileName: "group.jpg", descriptor: alice },
    { fileName: "group.jpg", descriptor: bob },
    { fileName: "alice-solo.jpg", descriptor: jitter(alice, 0.05) },
  ]);

  assert.equal(clusters.length, 2);
  // Largest first: Alice has two faces, Bob has one.
  assert.deepEqual(photosIn(clusters[0]), ["group.jpg", "alice-solo.jpg"]);
  assert.deepEqual(photosIn(clusters[1]), ["group.jpg"]);
});

test("regression: the old comparison grouped different people together", () => {
  // The shipped code kept faces whose distance was GREATER than the threshold,
  // so these two would have been grouped. They must not be.
  const alice = descriptor(0, 5);
  const bob = descriptor(1, 5);
  assert.ok(euclidean(alice, bob) > 0.6);

  const clusters = clusterFaces([
    { fileName: "alice.jpg", descriptor: alice },
    { fileName: "bob.jpg", descriptor: bob },
  ]);

  assert.equal(clusters.length, 2);
});

test("clusters come back ordered largest first", () => {
  const faces = [
    { fileName: "solo.jpg", descriptor: descriptor(1, 9) },
    { fileName: "a.jpg", descriptor: descriptor(0, 1) },
    { fileName: "b.jpg", descriptor: descriptor(0, 1) },
  ];

  const clusters = clusterFaces(faces);
  assert.equal(clusters.length, 2);
  assert.equal(clusters[0].faces.length, 2);
  assert.equal(clusters[0].id, 0);
  assert.equal(clusters[1].id, 1);
});

test("no faces yields no clusters", () => {
  assert.deepEqual(clusterFaces([]), []);
});
