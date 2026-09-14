// Face detection in the browser. Nothing here touches the network except to
// fetch the model weights; photos stay on the device.

import * as faceapi from "./vendor/face-api.esm.js";

const MODEL_URL = "/models";

/**
 * Longest side a photo is scaled to before detection. Detection cost scales
 * with pixel count, and the recognition net only ever sees a small aligned
 * crop, so a 12MP original buys nothing but memory pressure.
 *
 * 2048 was tried and made descriptors no more separable — it only let smaller
 * faces clear the size floor below — while crashing the tab with three photos
 * in flight at once.
 */
const MAX_DIM = 1280;

/** Size of the square face crop shown next to each person. */
const THUMB_PX = 96;

/**
 * Longest side of the per-photo preview shown in the results grid. The grid
 * draws tiles about 110px wide, and decoding a 12MP original for each one is
 * what makes a large album crawl.
 */
const PREVIEW_PX = 320;

/**
 * Minimum detector confidence. At 0.5 the detector reported patterned fabric
 * as a face, which then showed up as a person of its own; that scored 0.50.
 * Raising it further starts costing real faces in dark photos, so this sits
 * just above the false positive rather than comfortably above it.
 */
const MIN_CONFIDENCE = 0.5;

/**
 * Faces smaller than this on the processed image are dropped. A face of twenty
 * pixels produces a descriptor close to every other descriptor, so it does not
 * simply form its own group — it pulls unrelated people together. Kept low, so
 * that real faces survive; the merge control handles what is left.
 */
const MIN_FACE_PX = 24;

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
      faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
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

// ssdMobilenetv1, not tinyFaceDetector. Tiny is a third of the speed and a
// tenth of the size, but on ordinary phone photos it missed almost everything:
// on one real folder it found 1 face where this finds 12. Accuracy wins.
/** A small JPEG of the whole photo, for the results grid. */
function previewOf(source) {
  const scale = PREVIEW_PX / Math.max(source.width, source.height);
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(source.width * Math.min(1, scale)));
  canvas.height = Math.max(1, Math.round(source.height * Math.min(1, scale)));
  canvas.getContext("2d").drawImage(source, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.7);
}

const detectorOptions = new faceapi.SsdMobilenetv1Options({
  minConfidence: MIN_CONFIDENCE,
});

/**
 * Every face found in one photo, plus a small preview of the photo itself.
 * @returns {Promise<{faces: Array<{descriptor: number[], thumbnail: string, score: number}>, preview: string}>}
 */
export async function detectFaces(file) {
  const image = await loadCanvas(file);
  try {
    const results = await faceapi
      .detectAllFaces(image, detectorOptions)
      // Full 68-point landmarks, not the tiny net: descriptors are extracted
      // from a crop aligned by these points, and many real photos have faces
      // at an angle where the alignment matters.
      .withFaceLandmarks()
      .withFaceDescriptors();

    return {
      preview: previewOf(image),
      faces: results
        .filter(
          (result) =>
            Math.min(result.detection.box.width, result.detection.box.height) >=
            MIN_FACE_PX,
        )
        .map((result) => ({
          descriptor: Array.from(result.descriptor),
          thumbnail: cropFace(image, result.detection.box),
          score: result.detection.score,
        })),
    };
  } finally {
    // Let the backing store go now rather than at the next collection.
    image.width = 0;
    image.height = 0;
  }
}
