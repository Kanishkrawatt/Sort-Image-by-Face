// The machine-facing API, for callers such as cloudbox. Separate from the
// browser app: it authenticates with a key header rather than a session
// cookie, and it does the face work server-side instead of in a browser.

import crypto from "node:crypto";
import express from "express";
import { clusterFaces, euclidean, MATCH_THRESHOLD } from "../public/cluster.js";
import { allowedHosts } from "./image-source.js";
import { detect, queueDepth } from "./runner.js";
import { secretMatches } from "./auth.js";

/** Images accepted in one blocking request. Keep it small; the box is small. */
export const MAX_SYNC_IMAGES = 40;
/** Images accepted for a background job. */
export const MAX_JOB_IMAGES = 250;

const JOB_RETENTION_MS = 30 * 60 * 1000;
const jobs = new Map();

function readList(body) {
  // Accept `imageUrls` (what cloudbox already sends) or `urls`.
  const raw = body?.imageUrls ?? body?.urls;
  if (!Array.isArray(raw)) throw new Error("imageUrls must be an array of URLs");

  const urls = [...new Set(raw.filter((u) => typeof u === "string" && u.trim()))];
  if (urls.length === 0) throw new Error("imageUrls is empty");
  return urls;
}

function shape(result, urls, startedAt, threshold) {
  const clusters = clusterFaces(
    result.faces.map((face) => ({ ...face, fileName: face.url })),
    threshold,
  );

  const people = clusters.map((cluster, id) => ({
    id,
    photos: [...new Set(cluster.faces.map((face) => face.url))],
    faceCount: cluster.faces.length,
    faces: cluster.faces.map(({ url, score, box }) => ({ url, score, box })),
  }));

  return {
    people,
    noFaces: result.noFaces,
    failed: result.failed,
    stats: {
      images: urls.length,
      processed: urls.length - result.failed.length,
      faces: result.faces.length,
      people: people.length,
      threshold,
      ms: Date.now() - startedAt,
    },
  };
}

async function group(urls, threshold) {
  const startedAt = Date.now();
  const result = await detect({
    urls,
    hosts: allowedHosts(),
    maxDim: Number(process.env.API_MAX_DIM ?? 1600),
  });
  return shape(result, urls, startedAt, threshold);
}

function thresholdFrom(body) {
  const value = Number(body?.threshold);
  if (!Number.isFinite(value)) return MATCH_THRESHOLD;
  return Math.min(1.5, Math.max(0.1, value));
}

function sweep(jobsMap, now = Date.now()) {
  for (const [id, job] of jobsMap) {
    if (now - job.updatedAt > JOB_RETENTION_MS) jobsMap.delete(id);
  }
}

/**
 * Turn a detection result for two photos into a comparison.
 *
 * Pure, so the awkward parts — a photo with no face, a photo with several —
 * can be tested without running the models.
 *
 * @returns {{status: number, body: object}}
 */
export function compareResult(result, a, b, threshold) {
  if (result.failed.length) {
    return {
      status: 422,
      body: { error: "Could not read one of the images.", failed: result.failed },
    };
  }

  const facesFor = (url) =>
    result.faces.filter((face) => face.url === url).sort((x, y) => y.score - x.score);

  const facesA = facesFor(a);
  const facesB = facesFor(b);
  const missing = [facesA.length === 0 && "a", facesB.length === 0 && "b"].filter(Boolean);
  if (missing.length) {
    return {
      status: 422,
      body: {
        error: `No face found in ${missing.join(" and ")}.`,
        facesFound: { a: facesA.length, b: facesB.length },
      },
    };
  }

  // The clearest face in each photo. More than one makes the answer ambiguous,
  // so the count is reported rather than hidden.
  const distance = euclidean(facesA[0].descriptor, facesB[0].descriptor);
  return {
    status: 200,
    body: {
      match: distance < threshold,
      distance: Number(distance.toFixed(4)),
      threshold,
      a: { faces: facesA.length, score: facesA[0].score, box: facesA[0].box },
      b: { faces: facesB.length, score: facesB[0].score, box: facesB[0].box },
    },
  };
}

export function createApiRouter() {
  const router = express.Router();
  const apiKey = process.env.API_KEY;

  router.use(express.json({ limit: "1mb" }));

  // CORS, so a browser front end can call this directly if you want it to.
  const origins = (process.env.API_CORS_ORIGIN ?? "")
    .split(",").map((o) => o.trim()).filter(Boolean);
  router.use((req, res, next) => {
    const origin = req.headers.origin;
    if (origin && (origins.includes("*") || origins.includes(origin))) {
      res.set("Access-Control-Allow-Origin", origin);
      res.set("Vary", "Origin");
      res.set("Access-Control-Allow-Headers", "content-type, x-api-key");
      res.set("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
    }
    if (req.method === "OPTIONS") return res.sendStatus(204);
    next();
  });

  router.use((req, res, next) => {
    if (!apiKey) {
      return res.status(503).json({
        error: "The API is disabled. Set API_KEY to enable it.",
      });
    }
    const given = req.get("x-api-key") ?? "";
    if (!secretMatches(given, apiKey)) {
      return res.status(401).json({ error: "Invalid or missing x-api-key." });
    }
    next();
  });

  // Blocking: send URLs, get people back.
  router.post("/group", async (req, res) => {
    let urls;
    try {
      urls = readList(req.body);
    } catch (error) {
      return res.status(400).json({ error: error.message });
    }
    if (urls.length > MAX_SYNC_IMAGES) {
      return res.status(413).json({
        error: `At most ${MAX_SYNC_IMAGES} images per request. Use POST /api/jobs for more.`,
      });
    }

    try {
      res.json(await group(urls, thresholdFrom(req.body)));
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });

  // Non-blocking: hand over URLs, poll for the result.
  router.post("/jobs", (req, res) => {
    let urls;
    try {
      urls = readList(req.body);
    } catch (error) {
      return res.status(400).json({ error: error.message });
    }
    if (urls.length > MAX_JOB_IMAGES) {
      return res.status(413).json({ error: `At most ${MAX_JOB_IMAGES} images per job.` });
    }

    sweep(jobs);
    const id = crypto.randomUUID();
    const job = { id, status: "queued", images: urls.length, updatedAt: Date.now() };
    jobs.set(id, job);

    group(urls, thresholdFrom(req.body)).then(
      (result) => Object.assign(job, { status: "done", result, updatedAt: Date.now() }),
      (error) => Object.assign(job, { status: "error", error: error.message, updatedAt: Date.now() }),
    );
    job.status = "running";

    res.status(202).json({ jobId: id, status: job.status, images: job.images, poll: `/api/jobs/${id}` });
  });

  router.get("/jobs/:id", (req, res) => {
    const job = jobs.get(req.params.id);
    if (!job) {
      return res.status(404).json({
        error: "No such job. Results are kept for 30 minutes, and are lost if the service restarts.",
      });
    }
    res.json(job);
  });

  /**
   * Are these two photos the same person? Returns the descriptor distance so
   * the caller can apply its own judgement rather than trusting a boolean.
   *
   * This is a similarity measurement, not an identity check. See the README:
   * it must not gate access to anything.
   */
  router.post("/compare", async (req, res) => {
    const { a, b } = req.body ?? {};
    if (typeof a !== "string" || !a.trim() || typeof b !== "string" || !b.trim()) {
      return res.status(400).json({ error: "Send two image URLs, as `a` and `b`." });
    }

    const threshold = thresholdFrom(req.body);
    try {
      const result = await detect({
        urls: a === b ? [a] : [a, b],
        hosts: allowedHosts(),
        maxDim: Number(process.env.API_MAX_DIM ?? 1600),
      });

      const { status, body } = compareResult(result, a, b, threshold);
      res.status(status).json(body);
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });

  router.get("/status", (_req, res) => {
    res.json({
      ok: true,
      queued: queueDepth(),
      maxSyncImages: MAX_SYNC_IMAGES,
      maxJobImages: MAX_JOB_IMAGES,
      threshold: MATCH_THRESHOLD,
      routes: ["POST /api/group", "POST /api/jobs", "GET /api/jobs/:id", "POST /api/compare", "GET /api/status"],
      allowedHosts: allowedHosts() ?? "any public host",
    });
  });

  return router;
}

/**
 * The shape the existing cloudbox integration expects: an array of groups,
 * each with the URLs of the photos that person appears in.
 */
export function legacyHandler() {
  const apiKey = process.env.API_KEY;

  return async (req, res) => {
    if (!apiKey) return res.status(503).json({ error: "The API is disabled. Set API_KEY to enable it." });
    if (!secretMatches(req.get("x-api-key") ?? "", apiKey)) {
      return res.status(401).json({ error: "Invalid or missing x-api-key." });
    }

    let urls;
    try {
      urls = readList(req.body);
    } catch (error) {
      return res.status(400).json({ error: error.message });
    }
    if (urls.length > MAX_SYNC_IMAGES) {
      return res.status(413).json({ error: `At most ${MAX_SYNC_IMAGES} images per request.` });
    }

    try {
      const result = await group(urls, thresholdFrom(req.body));
      res.json(result.people.map((person) => ({ urls: person.photos })));
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  };
}
