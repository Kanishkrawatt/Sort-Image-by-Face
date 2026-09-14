// Runs detection jobs one at a time in a short-lived child process.

import { fork } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const workerPath = path.join(path.dirname(fileURLToPath(import.meta.url)), "worker.js");

/** Per-image budget; a job's timeout scales with how many images it carries. */
const MS_PER_IMAGE = 20000;
const BASE_TIMEOUT_MS = 30000;

// One job at a time. Two TensorFlow processes would not fit in 512MB.
let queue = Promise.resolve();
let depth = 0;

export function queueDepth() {
  return depth;
}

function runOnce(job) {
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
        signal === "SIGKILL"
          ? "detection worker was killed, likely out of memory"
          : `detection worker exited with code ${code}`,
      ));
    });

    child.send(job);
  });
}

/** Queue a detection job. Resolves with `{ faces, noFaces, failed }`. */
export function detect(job) {
  depth++;
  const run = queue.then(() => runOnce(job), () => runOnce(job));
  queue = run.then(
    () => { depth--; },
    () => { depth--; },
  );
  return run;
}
