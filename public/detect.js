// Face detection in the browser. Nothing here touches the network except to
// fetch the model weights; photos stay on the device.

import {
  DETECT_SIZE, FACE_SIZE, TEMPLATE,
  addVectors, decodeDetections, letterbox, mirrorFace, normalise,
  similarityTransform, suppressOverlaps, toTensor, warpFace,
} from "./facepipe.js";

const MODEL_URL = "/models";

/**
 * Longest side a photo is scaled to before anything else.
 *
 * The detector sees a 640px letterbox regardless, so this governs how much
 * detail the face crops keep. Larger finds a few more distant faces but splits
 * people more often, and costs memory; this matched a hand count best.
 */
const MAX_DIM = 1024;

/** Detector confidence below which a box is more likely to be a texture. */
const MIN_SCORE = 0.5;

/** Faces smaller than this on the processed image describe too poorly to use. */
const MIN_FACE_PX = 24;

/** Longest side of the per-photo preview shown in the results grid. */
const PREVIEW_PX = 320;

let ready = null;
let detector = null;
let recogniser = null;
let backend = "unknown";

export function activeBackend() {
  return backend;
}

/** Load the runtime and both models. Safe to call repeatedly. */
export function init(onProgress = () => {}) {
  if (ready) return ready;

  ready = (async () => {
    onProgress("Starting up");
    // The runtime's own WebAssembly lives alongside it, not on a CDN.
    ort.env.wasm.wasmPaths = "/vendor/";
    ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
    ort.env.logLevel = "error";

    onProgress("Loading models");
    [detector, recogniser] = await Promise.all([
      ort.InferenceSession.create(`${MODEL_URL}/detection.onnx`),
      ort.InferenceSession.create(`${MODEL_URL}/recognition.onnx`),
    ]);

    backend = `wasm×${ort.env.wasm.numThreads}`;
    return backend;
  })();

  return ready;
}

/**
 * SCRFD emits nine tensors whose names carry no meaning, so they are sorted by
 * shape: one value per anchor is a score, four are a box, ten are keypoints.
 */
function groupOutputs(session, results) {
  const grouped = { score: [], bbox: [], kps: [] };
  for (const name of session.outputNames) {
    const tensor = results[name];
    const width = tensor.dims[tensor.dims.length - 1];
    if (width === 1) grouped.score.push(tensor.data);
    else if (width === 4) grouped.bbox.push(tensor.data);
    else if (width === 10) grouped.kps.push(tensor.data);
  }
  return grouped;
}

/** Decode a photo, scaled down, as pixels we can sample from directly. */
async function loadPixels(file) {
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
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(bitmap, 0, 0);
  bitmap.close();

  return {
    canvas,
    image: context.getImageData(0, 0, canvas.width, canvas.height),
  };
}

/** The 640px letterboxed copy the detector wants. */
function letterboxed(canvas) {
  const fit = letterbox(canvas.width, canvas.height);
  const square = document.createElement("canvas");
  square.width = DETECT_SIZE;
  square.height = DETECT_SIZE;

  const context = square.getContext("2d", { willReadFrequently: true });
  context.fillStyle = "#000";
  context.fillRect(0, 0, DETECT_SIZE, DETECT_SIZE);
  context.drawImage(canvas, 0, 0, fit.width, fit.height);

  return { fit, pixels: context.getImageData(0, 0, DETECT_SIZE, DETECT_SIZE).data };
}

/** A small JPEG of the whole photo, for the results grid. */
function previewOf(canvas) {
  const scale = Math.min(1, PREVIEW_PX / Math.max(canvas.width, canvas.height));
  const small = document.createElement("canvas");
  small.width = Math.max(1, Math.round(canvas.width * scale));
  small.height = Math.max(1, Math.round(canvas.height * scale));
  small.getContext("2d").drawImage(canvas, 0, 0, small.width, small.height);
  return small.toDataURL("image/jpeg", 0.7);
}

/** The aligned face itself, as a picture, for the person's avatar. */
function thumbnailFrom(preview) {
  const canvas = document.createElement("canvas");
  canvas.width = FACE_SIZE;
  canvas.height = FACE_SIZE;
  const context = canvas.getContext("2d");
  const image = context.createImageData(FACE_SIZE, FACE_SIZE);
  for (let i = 0; i < FACE_SIZE * FACE_SIZE; i++) {
    image.data[i * 4] = preview[i * 3];
    image.data[i * 4 + 1] = preview[i * 3 + 1];
    image.data[i * 4 + 2] = preview[i * 3 + 2];
    image.data[i * 4 + 3] = 255;
  }
  context.putImageData(image, 0, 0);
  return canvas.toDataURL("image/jpeg", 0.8);
}

/**
 * Every face found in one photo, plus a small preview of the photo itself.
 * @returns {Promise<{faces: Array<{descriptor: number[], thumbnail: string, score: number}>, preview: string}>}
 */
export async function detectFaces(file) {
  const { canvas, image } = await loadPixels(file);

  try {
    const { fit, pixels } = letterboxed(canvas);
    const detections = await detector.run({
      "input.1": new ort.Tensor(
        "float32",
        toTensor(pixels, DETECT_SIZE, DETECT_SIZE, 4, DETECT_SIZE),
        [1, 3, DETECT_SIZE, DETECT_SIZE],
      ),
    });

    const boxes = suppressOverlaps(
      decodeDetections(groupOutputs(detector, detections), fit.scale, MIN_SCORE),
    ).filter((box) => Math.min(box.x2 - box.x1, box.y2 - box.y1) >= MIN_FACE_PX);

    const faces = [];
    for (const box of boxes) {
      const preview = new Uint8Array(FACE_SIZE * FACE_SIZE * 3);
      const aligned = warpFace(
        image.data, image.width, image.height, 4,
        similarityTransform(box.keypoints, TEMPLATE),
        preview,
      );

      const describe = async (tensor) => {
        const output = await recogniser.run({
          "input.1": new ort.Tensor("float32", tensor, [1, 3, FACE_SIZE, FACE_SIZE]),
        });
        return output[recogniser.outputNames[0]].data;
      };

      // Averaging a face with its mirror cancels some of the noise that pose
      // and lighting introduce, for one extra pass and no extra memory.
      const combined = addVectors(await describe(aligned), await describe(mirrorFace(aligned)));

      faces.push({
        descriptor: normalise(combined),
        thumbnail: thumbnailFrom(preview),
        score: box.score,
      });
    }

    return { preview: previewOf(canvas), faces };
  } finally {
    // Let the backing store go now rather than at the next collection.
    canvas.width = 0;
    canvas.height = 0;
  }
}
