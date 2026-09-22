// Forked child process: downloads images, finds and describes faces, exits.
//
// It exits so its memory goes back to the operating system, and it handles only
// a few images before being replaced — see lib/runner.js.

import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  DETECT_SIZE, TEMPLATE,
  addVectors, decodeDetections, letterbox, mirrorFace, normalise,
  similarityTransform, suppressOverlaps, toTensor, warpFace,
} from "../public/facepipe.js";
import { fetchImage } from "./image-source.js";

const require = createRequire(import.meta.url);
const ort = require("onnxruntime-node");
const sharp = require("sharp");

const modelsDir = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "models");

let detector;
let recogniser;

async function start() {
  detector = await ort.InferenceSession.create(path.join(modelsDir, "detection.onnx"));
  recogniser = await ort.InferenceSession.create(path.join(modelsDir, "recognition.onnx"));
}

/**
 * SCRFD emits nine tensors and the names are meaningless, so they are sorted by
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

async function facesIn(buffer, maxDim, minScore, minFacePx) {
  // sharp() with no arguments applies the EXIF orientation, which phone photos
  // rely on; without it a sideways face is often missed entirely.
  const upright = sharp(buffer).rotate().removeAlpha();
  const { data: pixels, info } = await upright
    .resize({ width: maxDim, height: maxDim, fit: "inside", withoutEnlargement: true })
    .raw()
    .toBuffer({ resolveWithObject: true });

  const fit = letterbox(info.width, info.height);
  const { data: padded } = await sharp(pixels, {
    raw: { width: info.width, height: info.height, channels: info.channels },
  })
    .resize(fit.width, fit.height)
    .extend({
      top: 0, left: 0,
      bottom: DETECT_SIZE - fit.height, right: DETECT_SIZE - fit.width,
      background: { r: 0, g: 0, b: 0 },
    })
    .raw()
    .toBuffer({ resolveWithObject: true });

  const detections = await detector.run({
    "input.1": new ort.Tensor(
      "float32",
      toTensor(padded, DETECT_SIZE, DETECT_SIZE, 3, DETECT_SIZE),
      [1, 3, DETECT_SIZE, DETECT_SIZE],
    ),
  });

  const boxes = suppressOverlaps(
    decodeDetections(groupOutputs(detector, detections), fit.scale, minScore),
  ).filter((box) => Math.min(box.x2 - box.x1, box.y2 - box.y1) >= minFacePx);

  const faces = [];
  for (const box of boxes) {
    const aligned = warpFace(
      pixels, info.width, info.height, info.channels,
      similarityTransform(box.keypoints, TEMPLATE),
    );
    const describe = async (tensor) => {
      const output = await recogniser.run({
        "input.1": new ort.Tensor("float32", tensor, [1, 3, 112, 112]),
      });
      return output[recogniser.outputNames[0]].data;
    };

    // Averaging a face with its mirror cancels some of the noise that pose and
    // lighting introduce, for one extra pass and no extra memory.
    const combined = addVectors(await describe(aligned), await describe(mirrorFace(aligned)));

    faces.push({
      descriptor: normalise(combined),
      score: Number(box.score.toFixed(4)),
      box: {
        x: Math.round(box.x1), y: Math.round(box.y1),
        width: Math.round(box.x2 - box.x1), height: Math.round(box.y2 - box.y1),
      },
    });
  }

  return faces;
}

process.on("message", async (job) => {
  try {
    await start();

    const faces = [];
    const noFaces = [];
    const failed = [];

    for (const url of job.urls) {
      try {
        const buffer = await fetchImage(url, { hosts: job.hosts });
        const found = await facesIn(
          buffer,
          job.maxDim ?? 1024,
          job.minScore ?? 0.5,
          job.minFacePx ?? 24,
        );
        if (found.length === 0) noFaces.push(url);
        for (const face of found) faces.push({ ...face, url });
      } catch (error) {
        // One unreachable or broken image must not lose the whole batch.
        failed.push({ url, error: error.message });
      }
    }

    reply({ ok: true, faces, noFaces, failed });
  } catch (error) {
    reply({ ok: false, error: error.message });
  }
});

/**
 * Send, then exit once it has actually gone.
 *
 * process.send is asynchronous, and exiting in the same tick drops the message:
 * the parent then sees a clean exit with no result and reports every image as
 * failed. Waiting for the callback is what makes the result arrive.
 */
function reply(payload) {
  process.send(payload, () => process.exit(0));
}
