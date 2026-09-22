// The face pipeline's arithmetic, with no runtime dependencies.
//
// Both the browser and the server run exactly these functions; they differ only
// in how they decode an image and how they execute an ONNX model. Keeping the
// maths here means the two cannot drift apart, and it can all be tested under
// node without loading a model.

/** Square input the detector expects. */
export const DETECT_SIZE = 640;

/** Feature-map strides SCRFD emits, each with two anchors per location. */
const STRIDES = [8, 16, 32];
const ANCHORS_PER_CELL = 2;

/** Square crop the recogniser expects. */
export const FACE_SIZE = 112;

/**
 * Where the five landmarks must land in that crop. This is ArcFace's canonical
 * template; matching it is what makes two photos of one person comparable.
 */
export const TEMPLATE = [
  [38.2946, 51.6963], // left eye
  [73.5318, 51.5014], // right eye
  [56.0252, 71.7366], // nose tip
  [41.5493, 92.3655], // left mouth corner
  [70.7299, 92.2041], // right mouth corner
];

/** Scale and offsets that fit a w×h image into a square, preserving shape. */
export function letterbox(width, height, size = DETECT_SIZE) {
  const scale = Math.min(size / width, size / height);
  return { scale, width: Math.round(width * scale), height: Math.round(height * scale) };
}

/**
 * Pack interleaved pixel data into the planar, normalised float tensor the
 * models take: 1 × 3 × size × size, with values centred on zero.
 */
export function toTensor(pixels, width, height, channels, size) {
  const out = new Float32Array(3 * size * size);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const from = (y * width + x) * channels;
      const to = y * size + x;
      for (let c = 0; c < 3; c++) {
        out[c * size * size + to] = (pixels[from + c] - 127.5) / 128;
      }
    }
  }
  return out;
}

/**
 * Turn SCRFD's raw outputs into boxes with five keypoints, in the coordinates
 * of the original image.
 *
 * @param {{score: Float32Array[], bbox: Float32Array[], kps: Float32Array[]}} outputs
 *        one array per stride, in stride order
 */
export function decodeDetections(outputs, scale, minScore, size = DETECT_SIZE) {
  const found = [];

  STRIDES.forEach((stride, level) => {
    const scores = outputs.score[level];
    const boxes = outputs.bbox[level];
    const points = outputs.kps[level];
    const columns = size / stride;

    for (let i = 0; i < scores.length; i++) {
      if (scores[i] < minScore) continue;

      // Two anchors share each cell, so the cell index is the anchor index
      // halved. The prediction is a distance from the cell centre, in strides.
      const cell = Math.floor(i / ANCHORS_PER_CELL);
      const cx = (cell % columns) * stride;
      const cy = Math.floor(cell / columns) * stride;

      const b = i * 4;
      const k = i * 10;
      found.push({
        score: scores[i],
        x1: (cx - boxes[b] * stride) / scale,
        y1: (cy - boxes[b + 1] * stride) / scale,
        x2: (cx + boxes[b + 2] * stride) / scale,
        y2: (cy + boxes[b + 3] * stride) / scale,
        keypoints: Array.from({ length: 5 }, (_, p) => [
          (cx + points[k + p * 2] * stride) / scale,
          (cy + points[k + p * 2 + 1] * stride) / scale,
        ]),
      });
    }
  });

  return found;
}

const area = (b) => Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);

/** Keep the best of each cluster of overlapping boxes. */
export function suppressOverlaps(boxes, maxOverlap = 0.4) {
  const ordered = [...boxes].sort((a, b) => b.score - a.score);
  const kept = [];

  for (const box of ordered) {
    const overlapsSomethingBetter = kept.some((other) => {
      const width = Math.min(box.x2, other.x2) - Math.max(box.x1, other.x1);
      const height = Math.min(box.y2, other.y2) - Math.max(box.y1, other.y1);
      if (width <= 0 || height <= 0) return false;
      const intersection = width * height;
      return intersection / (area(box) + area(other) - intersection) > maxOverlap;
    });
    if (!overlapsSomethingBetter) kept.push(box);
  }

  return kept;
}

const centroid = (points) =>
  points.reduce((acc, p) => [acc[0] + p[0] / points.length, acc[1] + p[1] / points.length], [0, 0]);

/**
 * The rotation, scale and shift that best maps one set of points onto another.
 *
 * Two dimensions admit a closed form, so no matrix decomposition is needed.
 * Rotation is what makes a face lying on its side comparable with an upright
 * one — the single biggest source of wrong groupings before this.
 */
export function similarityTransform(from, to) {
  const fromCentre = centroid(from);
  const toCentre = centroid(to);

  let dot = 0;
  let cross = 0;
  let norm = 0;
  for (let i = 0; i < from.length; i++) {
    const fx = from[i][0] - fromCentre[0];
    const fy = from[i][1] - fromCentre[1];
    const tx = to[i][0] - toCentre[0];
    const ty = to[i][1] - toCentre[1];
    dot += fx * tx + fy * ty;
    cross += fx * ty - fy * tx;
    norm += fx * fx + fy * fy;
  }

  const a = dot / norm;
  const b = cross / norm;
  return {
    a,
    b,
    tx: toCentre[0] - (a * fromCentre[0] - b * fromCentre[1]),
    ty: toCentre[1] - (b * fromCentre[0] + a * fromCentre[1]),
  };
}

/**
 * Sample the aligned face out of an image, bilinearly, straight into the
 * normalised tensor the recogniser wants. `preview`, if given, receives the
 * same crop as ordinary pixel values.
 */
export function warpFace(pixels, width, height, channels, transform, preview) {
  const out = new Float32Array(3 * FACE_SIZE * FACE_SIZE);
  const determinant = transform.a * transform.a + transform.b * transform.b;
  const ia = transform.a / determinant;
  const ib = -transform.b / determinant;

  for (let v = 0; v < FACE_SIZE; v++) {
    for (let u = 0; u < FACE_SIZE; u++) {
      const dx = u - transform.tx;
      const dy = v - transform.ty;
      const x = ia * dx - ib * dy;
      const y = ib * dx + ia * dy;

      const x0 = Math.floor(x);
      const y0 = Math.floor(y);
      const fx = x - x0;
      const fy = y - y0;

      for (let c = 0; c < 3; c++) {
        let value = 0;
        for (const [ox, oy, weight] of [
          [0, 0, (1 - fx) * (1 - fy)], [1, 0, fx * (1 - fy)],
          [0, 1, (1 - fx) * fy], [1, 1, fx * fy],
        ]) {
          const px = Math.min(width - 1, Math.max(0, x0 + ox));
          const py = Math.min(height - 1, Math.max(0, y0 + oy));
          value += weight * pixels[(py * width + px) * channels + c];
        }
        out[c * FACE_SIZE * FACE_SIZE + v * FACE_SIZE + u] = (value - 127.5) / 128;
        if (preview) preview[(v * FACE_SIZE + u) * 3 + c] = Math.max(0, Math.min(255, value));
      }
    }
  }

  return out;
}

/**
 * Mirror an aligned crop.
 *
 * A face and its mirror describe the same person, so averaging the two
 * descriptions cancels some of the noise that pose and lighting introduce. It
 * costs a second pass over the recogniser and nothing in memory.
 */
export function mirrorFace(tensor) {
  const mirrored = new Float32Array(tensor.length);
  const plane = FACE_SIZE * FACE_SIZE;
  for (let c = 0; c < 3; c++) {
    for (let v = 0; v < FACE_SIZE; v++) {
      for (let u = 0; u < FACE_SIZE; u++) {
        mirrored[c * plane + v * FACE_SIZE + u] =
          tensor[c * plane + v * FACE_SIZE + (FACE_SIZE - 1 - u)];
      }
    }
  }
  return mirrored;
}

/** Element-wise sum, for averaging a face with its mirror before normalising. */
export function addVectors(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

/** Scale an embedding to unit length, so distances between them are comparable. */
export function normalise(values) {
  let sum = 0;
  for (const v of values) sum += v * v;
  const length = Math.sqrt(sum) || 1;
  return Array.from(values, (v) => v / length);
}
