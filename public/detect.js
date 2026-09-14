// Face detection in the browser. Nothing here touches the network except to
// fetch the model weights; photos stay on the device.

import * as faceapi from "./vendor/face-api.esm.js";

const MODEL_URL = "/models";

/**
 * Longest side a photo is scaled to before detection. Detection cost scales
 * with pixel count, and the recognition net only ever sees a small aligned
 * crop, so a 12MP original buys nothing but memory pressure.
 */
const MAX_DIM = 1280;

/** Size of the square face crop kept for the results grid. */
const THUMB_PX = 96;

/** Faces scoring below this are usually blurred bystanders. */
const SCORE_THRESHOLD = 0.5;

let ready = null;
let backend = "unknown";

export function activeBackend() {
  return backend;
}

/** Pick the fastest backend this browser actually supports. */
async function selectBackend() {
  for (const name of ["webgl", "wasm", "cpu"]) {
    try {
      if (await faceapi.tf.setBackend(name)) {
        await faceapi.tf.ready();
        return name;
      }
    } catch {
      // Unavailable here; fall through to the next one.
    }
  }
  await faceapi.tf.ready();
  return faceapi.tf.getBackend();
}

/** Load the backend and the three model nets. Safe to call repeatedly. */
export function init(onProgress = () => {}) {
  if (ready) return ready;

  ready = (async () => {
    onProgress("Starting up");
    backend = await selectBackend();

    onProgress("Loading models");
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL),
      faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
    ]);

    // The first inference pays for shader compilation — several seconds on
    // WebGL. Spend it here, where a wait is expected, rather than stalling on
    // the user's first photo.
    onProgress("Warming up");
    const scratch = document.createElement("canvas");
    scratch.width = 160;
    scratch.height = 160;
    scratch.getContext("2d").fillRect(0, 0, 160, 160);
    await faceapi
      .detectAllFaces(scratch, detectorOptions)
      .withFaceLandmarks(true)
      .withFaceDescriptors();

    return backend;
  })();

  return ready;
}

/**
 * Decode a file into a canvas, scaled down if it is large.
 *
 * It must be a canvas: face-api rejects an ImageBitmap, and in the chained
 * `.withFaceLandmarks().withFaceDescriptors()` form that rejection is swallowed
 * and the promise never settles, so the caller hangs rather than throwing.
 */
async function loadCanvas(file) {
  let bitmap = await createImageBitmap(file);

  const longest = Math.max(bitmap.width, bitmap.height);
  if (longest > MAX_DIM) {
    const scale = MAX_DIM / longest;
    const scaled = await createImageBitmap(bitmap, {
      resizeWidth: Math.round(bitmap.width * scale),
      resizeHeight: Math.round(bitmap.height * scale),
      resizeQuality: "medium",
    });
    bitmap.close();
    bitmap = scaled;
  }

  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  canvas.getContext("2d").drawImage(bitmap, 0, 0);
  bitmap.close();
  return canvas;
}

/** A square JPEG crop around one face, with a little headroom. */
function cropFace(source, box) {
  const canvas = document.createElement("canvas");
  canvas.width = THUMB_PX;
  canvas.height = THUMB_PX;

  const padding = box.width * 0.25;
  const x = Math.max(0, box.x - padding);
  const y = Math.max(0, box.y - padding);
  const width = Math.min(source.width - x, box.width + padding * 2);
  const height = Math.min(source.height - y, box.height + padding * 2);

  canvas
    .getContext("2d")
    .drawImage(source, x, y, width, height, 0, 0, THUMB_PX, THUMB_PX);
  return canvas.toDataURL("image/jpeg", 0.8);
}

const detectorOptions = new faceapi.TinyFaceDetectorOptions({
  inputSize: 416,
  scoreThreshold: SCORE_THRESHOLD,
});

/**
 * Every face found in one photo.
 * @returns {Promise<Array<{descriptor: number[], thumbnail: string, score: number}>>}
 */
export async function detectFaces(file) {
  const image = await loadCanvas(file);
  try {
    const results = await faceapi
      .detectAllFaces(image, detectorOptions)
      .withFaceLandmarks(true) // true selects the tiny landmark net
      .withFaceDescriptors();

    return results.map((result) => ({
      descriptor: Array.from(result.descriptor),
      thumbnail: cropFace(image, result.detection.box),
      score: result.detection.score,
    }));
  } finally {
    // Let the backing store go now rather than at the next collection.
    image.width = 0;
    image.height = 0;
  }
}
