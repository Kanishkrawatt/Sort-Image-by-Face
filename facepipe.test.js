import assert from "node:assert/strict";
import test from "node:test";
import {
  DETECT_SIZE, FACE_SIZE, TEMPLATE,
  addVectors, decodeDetections, letterbox, mirrorFace, normalise,
  similarityTransform, suppressOverlaps, toTensor, warpFace,
} from "./public/facepipe.js";

const close = (a, b, tolerance = 1e-6) =>
  assert.ok(Math.abs(a - b) < tolerance, `expected ${a} to be within ${tolerance} of ${b}`);

test("letterbox preserves aspect ratio and fits the square", () => {
  const wide = letterbox(1920, 1080);
  assert.equal(wide.width, DETECT_SIZE);
  assert.ok(wide.height <= DETECT_SIZE);
  close(wide.width / wide.height, 1920 / 1080, 0.01);

  const tall = letterbox(1080, 1920);
  assert.equal(tall.height, DETECT_SIZE);
  assert.ok(tall.width <= DETECT_SIZE);

  // An image already square and small is scaled up to fill the input.
  close(letterbox(320, 320).scale, 2);
});

test("toTensor writes planar channels, centred on zero", () => {
  // One red pixel, at full intensity.
  const pixels = new Uint8Array([255, 0, 0, 255]);
  const tensor = toTensor(pixels, 1, 1, 4, 2);

  assert.equal(tensor.length, 3 * 2 * 2);
  close(tensor[0], (255 - 127.5) / 128); // red plane
  close(tensor[4], (0 - 127.5) / 128); // green plane
  close(tensor[8], (0 - 127.5) / 128); // blue plane
});

test("mid-grey maps to roughly zero", () => {
  const tensor = toTensor(new Uint8Array([128, 128, 128]), 1, 1, 3, 1);
  for (const v of tensor) assert.ok(Math.abs(v) < 0.01);
});

test("decodeDetections turns cell offsets into image coordinates", () => {
  // One anchor above threshold, in the first cell of the stride-8 level.
  const cells = (DETECT_SIZE / 8) ** 2 * 2;
  const score = new Float32Array(cells);
  const bbox = new Float32Array(cells * 4);
  const kps = new Float32Array(cells * 10);

  score[0] = 0.9;
  bbox.set([1, 1, 1, 1], 0); // one stride in each direction from the centre
  for (let p = 0; p < 5; p++) kps.set([p, p], p * 2);

  const empty = (n) => new Float32Array(n);
  const found = decodeDetections(
    {
      score: [score, empty(1), empty(1)],
      bbox: [bbox, empty(1), empty(1)],
      kps: [kps, empty(1), empty(1)],
    },
    1, 0.5,
  );

  assert.equal(found.length, 1);
  assert.equal(found[0].x1, -8);
  assert.equal(found[0].x2, 8);
  assert.equal(found[0].keypoints.length, 5);
  assert.deepEqual(found[0].keypoints[0], [0, 0]);
  assert.deepEqual(found[0].keypoints[4], [32, 32]);
});

test("decodeDetections rescales back to the original image", () => {
  const cells = (DETECT_SIZE / 8) ** 2 * 2;
  const score = new Float32Array(cells);
  const bbox = new Float32Array(cells * 4);
  score[0] = 0.9;
  bbox.set([1, 1, 1, 1], 0);
  const empty = (n) => new Float32Array(n);

  const halved = decodeDetections(
    { score: [score, empty(1), empty(1)], bbox: [bbox, empty(1), empty(1)], kps: [new Float32Array(cells * 10), empty(1), empty(1)] },
    0.5, 0.5,
  );
  // At half scale, everything is twice as far apart in the original.
  assert.equal(halved[0].x1, -16);
  assert.equal(halved[0].x2, 16);
});

test("scores below the threshold are dropped", () => {
  const cells = (DETECT_SIZE / 8) ** 2 * 2;
  const score = new Float32Array(cells);
  score[0] = 0.3;
  const empty = (n) => new Float32Array(n);
  const found = decodeDetections(
    { score: [score, empty(1), empty(1)], bbox: [new Float32Array(cells * 4), empty(1), empty(1)], kps: [new Float32Array(cells * 10), empty(1), empty(1)] },
    1, 0.5,
  );
  assert.equal(found.length, 0);
});

test("overlapping boxes collapse to the best one", () => {
  const boxes = [
    { score: 0.9, x1: 0, y1: 0, x2: 10, y2: 10 },
    { score: 0.8, x1: 1, y1: 1, x2: 11, y2: 11 }, // almost the same face
    { score: 0.7, x1: 50, y1: 50, x2: 60, y2: 60 }, // somewhere else
  ];
  const kept = suppressOverlaps(boxes);
  assert.equal(kept.length, 2);
  assert.equal(kept[0].score, 0.9, "the strongest of a pair survives");
  assert.equal(kept[1].x1, 50);
});

test("boxes that merely touch are both kept", () => {
  const kept = suppressOverlaps([
    { score: 0.9, x1: 0, y1: 0, x2: 10, y2: 10 },
    { score: 0.8, x1: 10, y1: 0, x2: 20, y2: 10 },
  ]);
  assert.equal(kept.length, 2);
});

test("similarityTransform recovers an identity", () => {
  const t = similarityTransform(TEMPLATE, TEMPLATE);
  close(t.a, 1, 1e-6);
  close(t.b, 0, 1e-6);
  close(t.tx, 0, 1e-6);
  close(t.ty, 0, 1e-6);
});

test("similarityTransform recovers a rotation", () => {
  // Rotate the template by 90 degrees about the origin.
  const rotated = TEMPLATE.map(([x, y]) => [-y, x]);
  const t = similarityTransform(rotated, TEMPLATE);

  // Mapping back should undo it: a = cos(-90) = 0, b = sin(-90) = -1.
  close(t.a, 0, 1e-6);
  close(t.b, -1, 1e-6);

  for (let i = 0; i < TEMPLATE.length; i++) {
    const [x, y] = rotated[i];
    close(t.a * x - t.b * y + t.tx, TEMPLATE[i][0], 1e-6);
    close(t.b * x + t.a * y + t.ty, TEMPLATE[i][1], 1e-6);
  }
});

test("similarityTransform recovers a scale", () => {
  const doubled = TEMPLATE.map(([x, y]) => [x * 2, y * 2]);
  const t = similarityTransform(doubled, TEMPLATE);
  close(t.a, 0.5, 1e-6);
  close(t.b, 0, 1e-6);
});

test("warpFace samples the aligned crop, and fills the preview", () => {
  // A 224x224 image whose red channel is a horizontal ramp.
  const size = 224;
  const pixels = new Uint8Array(size * size * 3);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 3;
      pixels[i] = Math.round((x / (size - 1)) * 255);
      pixels[i + 1] = 128;
      pixels[i + 2] = 128;
    }

  // Halving maps the whole image onto the 112 crop.
  const transform = { a: 0.5, b: 0, tx: 0, ty: 0 };
  const preview = new Uint8Array(FACE_SIZE * FACE_SIZE * 3);
  const tensor = warpFace(pixels, size, size, 3, transform, preview);

  assert.equal(tensor.length, 3 * FACE_SIZE * FACE_SIZE);
  // The ramp should still run left to right across the crop.
  const left = preview[(10 * FACE_SIZE + 2) * 3];
  const right = preview[(10 * FACE_SIZE + FACE_SIZE - 3) * 3];
  assert.ok(right > left + 200, `expected a ramp, got ${left} then ${right}`);
  // Flat channels come back flat.
  close(preview[(10 * FACE_SIZE + 50) * 3 + 1], 128, 2);
});

test("normalise gives unit length, and survives an all-zero vector", () => {
  const unit = normalise([3, 4]);
  close(Math.hypot(...unit), 1, 1e-9);
  close(unit[0], 0.6, 1e-9);

  const zero = normalise([0, 0, 0]);
  assert.ok(zero.every((v) => v === 0), "must not divide by zero");
});

test("mirrorFace flips left to right, and twice returns the original", () => {
  const plane = FACE_SIZE * FACE_SIZE;
  const tensor = new Float32Array(3 * plane);
  // Mark one pixel near the left edge of the red plane.
  tensor[5 * FACE_SIZE + 2] = 1;

  const mirrored = mirrorFace(tensor);
  assert.equal(mirrored[5 * FACE_SIZE + 2], 0, "the mark should have moved");
  assert.equal(mirrored[5 * FACE_SIZE + (FACE_SIZE - 3)], 1, "to the far side");

  const twice = mirrorFace(mirrored);
  assert.deepEqual(Array.from(twice), Array.from(tensor));
});

test("mirrorFace keeps each colour plane separate", () => {
  const plane = FACE_SIZE * FACE_SIZE;
  const tensor = new Float32Array(3 * plane);
  tensor[2 * plane + 7 * FACE_SIZE + 1] = 0.5; // blue plane only

  const mirrored = mirrorFace(tensor);
  assert.equal(mirrored[2 * plane + 7 * FACE_SIZE + (FACE_SIZE - 2)], 0.5);
  assert.equal(mirrored[7 * FACE_SIZE + (FACE_SIZE - 2)], 0, "red plane untouched");
});

test("addVectors sums element-wise", () => {
  assert.deepEqual(Array.from(addVectors([1, 2, 3], [10, 20, 30])), [11, 22, 33]);
});

test("a face and its mirror average to a unit vector", () => {
  const a = [3, 0, 4];
  const b = [0, 5, 0];
  const averaged = normalise(addVectors(a, b));
  close(Math.hypot(...averaged), 1, 1e-9);
});
