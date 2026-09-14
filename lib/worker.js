// Forked child process: downloads images, finds faces, reports descriptors,
// exits. It exits so its memory goes back to the operating system — TensorFlow
// settles around 400MB of resident memory and the box only has 512MB.

import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { fetchImage } from "./image-source.js";

const require = createRequire(import.meta.url);
const tf = require("@tensorflow/tfjs");
const wasmBackend = require("@tensorflow/tfjs-backend-wasm");
const faceapi = require("@vladmandic/face-api/dist/face-api.node-wasm.js");
const sharp = require("sharp");

const modelsDir = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "models");

async function start() {
  const wasmDir = path.join(
    path.dirname(require.resolve("@tensorflow/tfjs-backend-wasm/package.json")),
    "dist/",
  );
  wasmBackend.setWasmPaths(wasmDir);
  await tf.setBackend("wasm");
  await tf.ready();

  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelsDir);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelsDir);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelsDir);
}

async function facesIn(buffer, maxDim, options, minFacePx) {
  // sharp() with no arguments applies the EXIF orientation, which phone photos
  // rely on; without it a sideways face is often missed entirely.
  const { data, info } = await sharp(buffer)
    .rotate()
    .resize({ width: maxDim, height: maxDim, fit: "inside", withoutEnlargement: true })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const tensor = tf.tensor3d(data, [info.height, info.width, info.channels], "int32");
  try {
    const found = await faceapi
      .detectAllFaces(tensor, options)
      .withFaceLandmarks()
      .withFaceDescriptors();

    // A tiny face gives a descriptor that sits close to every other
    // descriptor, so it drags unrelated people together rather than simply
    // forming a group of its own.
    return found
      .filter(
        (r) => Math.min(r.detection.box.width, r.detection.box.height) >= minFacePx,
      )
      .map((result) => ({
        descriptor: Array.from(result.descriptor),
        score: Number(result.detection.score.toFixed(4)),
        box: {
          x: Math.round(result.detection.box.x),
          y: Math.round(result.detection.box.y),
          width: Math.round(result.detection.box.width),
          height: Math.round(result.detection.box.height),
        },
      }));
  } finally {
    tensor.dispose();
  }
}

process.on("message", async (job) => {
  try {
    await start();
    // Matches the browser: ssdMobilenetv1, because tinyFaceDetector misses
    // most faces in ordinary photos.
    const options = new faceapi.SsdMobilenetv1Options({
      minConfidence: job.minConfidence ?? 0.5,
    });

    const faces = [];
    const noFaces = [];
    const failed = [];

    for (const url of job.urls) {
      try {
        const buffer = await fetchImage(url, { hosts: job.hosts });
        const found = await facesIn(
          buffer,
          job.maxDim ?? 1600,
          options,
          job.minFacePx ?? 24,
        );
        if (found.length === 0) noFaces.push(url);
        for (const face of found) faces.push({ ...face, url });
      } catch (error) {
        // One unreachable or broken image must not lose the whole batch.
        failed.push({ url, error: error.message });
      }
    }

    process.send({ ok: true, faces, noFaces, failed });
  } catch (error) {
    process.send({ ok: false, error: error.message });
  } finally {
    process.exit(0);
  }
});
