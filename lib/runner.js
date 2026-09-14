// Runs detection in short-lived child processes, a few images at a time.

import { fork } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const workerPath = path.join(path.dirname(fileURLToPath(import.meta.url)), "worker.js");

/** Per-image budget; a chunk's timeout scales with how many images it carries. */
const MS_PER_IMAGE = 30000;
const BASE_TIMEOUT_MS = 30000;

/**
 * Images handled by one child process before it is replaced.
 *
 * TensorFlow's WebAssembly heap grows as images are processed and does not give
 * memory back, so a long run creeps towards the instance limit and is killed —
 * on a 512MB box, eight images at 1600px was enough to take the whole service
 * down. Memory returns to the operating system only when the process exits, so
 * the run is cut into chunks and each chunk gets a fresh one. Peak memory then
 * depends on the chunk size rather than on how many images were asked for.
 */
const CHUNK_SIZE = Math.max(1, Number(process.env.API_CHUNK_SIZE ?? 4));

// One chunk at a time. Two TensorFlow processes would not fit in 512MB.
let queue = Promise.resolve();
let depth = 0;

export function queueDepth() {
  return depth;
}

function runChunk(job) {
  return new Promise((resolve, reject) => {
    const child = fork(workerPath, [], {
      // Cap the heap so V8 collects rather than growing into the memory limit.
      execArgv: ["--max-old-space-size=320"],
      stdio: ["ignore", "inherit", "inherit", "ipc"],
    });

    const timeout = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error("detection timed out"));
    }, BASE_TIMEOUT_MS + job.urls.length * MS_PER_IMAGE);

    let settled = false;
    const finish = (fn, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      fn(value);
    };

    child.on("message", (message) => {
      if (message.ok) finish(resolve, message);
      else finish(reject, new Error(message.error));
      child.kill();
    });

    child.on("error", (error) => finish(reject, error));
    child.on("exit", (code, signal) => {
      // Exiting before sending a result means it died — usually out of memory.
      finish(reject, new Error(
        signal
          ? `detection worker was killed (${signal}), most likely out of memory`
          : `detection worker exited with code ${code}`,
      ));
    });

    child.send(job);
  });
}

/** Put one chunk on the shared queue, so only one ever runs at a time. */
function enqueue(task) {
  const run = queue.then(task, task);
  queue = run.then(() => {}, () => {});
  return run;
}

/**
 * Detect faces across a list of images.
 *
 * A chunk that dies takes only its own images with it: they are reported as
 * failed and the remaining chunks still run. Before this, one oversized image
 * could kill the process and cost the caller the entire batch.
 *
 * @returns {Promise<{faces: object[], noFaces: string[], failed: object[]}>}
 */
export async function detect(job) {
  const chunks = [];
  for (let i = 0; i < job.urls.length; i += CHUNK_SIZE) {
    chunks.push(job.urls.slice(i, i + CHUNK_SIZE));
  }

  const merged = { faces: [], noFaces: [], failed: [] };
  depth++;
  try {
    for (const urls of chunks) {
      try {
        const part = await enqueue(() => runChunk({ ...job, urls }));
        merged.faces.push(...part.faces);
        merged.noFaces.push(...part.noFaces);
        merged.failed.push(...part.failed);
      } catch (error) {
        for (const url of urls) merged.failed.push({ url, error: error.message });
      }
    }
  } finally {
    depth--;
  }

  return merged;
}
