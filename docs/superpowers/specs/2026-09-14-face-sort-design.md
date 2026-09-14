# Sort-Image-by-Face — Design

Date: 2026-09-14
Status: Approved

## Problem

The current service is a single Express endpoint that accepts an array of image
URLs, downloads each one, runs face-api.js on the server, and returns groups.
It has four problems that together make it unfixable in place:

1. **The grouping is wrong.** `group-by-face.js` keeps faces whose descriptor
   distance is *greater* than 0.6, which groups photos of different people and
   separates photos of the same person. Groups also overlap, because a face is
   never marked processed against itself.
2. **Only the first face in each photo is used.** Group photos are represented
   by one arbitrary person.
3. **It cannot run on the Render free tier.** 512MB RAM and 0.1 CPU have to hold
   the Node process, `node-canvas`, TensorFlow.js, and 13MB of models. Cold
   starts are dominated by model loading.
4. **It is an open SSRF.** An unauthenticated POST makes the server fetch any
   URL supplied by the caller.

## Approach

Move all face work into the browser. The server keeps no models, decodes no
images, and fetches no URLs. It serves a gated static app and nothing else.

This removes the memory ceiling, the cold-start cost, and the SSRF in one move,
and it means photos never leave the user's device.

## Architecture

```
BROWSER  (all compute)
  index.html    login gate and app shell
  app.js        folder selection, orchestration, rendering
  detect.js     face-api wrapper: File -> [descriptor]
  cluster.js    greedy clustering of descriptors
  worker.js     runs detect.js off the main thread
  vendor/       face-api dist, jszip
  models/       6.7MB, cached by the browser after first load

RENDER FREE  (Express, ~50MB RAM)
  POST /api/auth    passphrase -> signed cookie
  GET  /*           static files, cookie-gated; login page exempt
```

### Server dependencies

`express` only. Removed: `canvas`, `face-api.js`, `axios`, `multer`,
`node-fetch`, `body-parser`, `nodemon`, `jest`, `supertest`. Dropping `canvas`
also removes a native build from the Render deploy.

### Models

Reduced from 13MB to 6.7MB by deleting what is never used:

| Model | Size | Kept |
|---|---|---|
| `tiny_face_detector` | 193KB | yes |
| `face_landmark_68_tiny` | 77KB | yes |
| `face_recognition` | 6.4MB | yes — this is the descriptor net |
| `ssd_mobilenetv1` | 5.6MB | no |
| `age_gender` | 430KB | no |
| `face_expression` | 330KB | no |

### Library

`face-api.js` is replaced with `@vladmandic/face-api`, a maintained fork with a
compatible API and current TensorFlow.js. The original's last release was 2020
and pins tfjs 1.x.

Backend preference at runtime: WebGPU, then WebGL, then WASM. This choice is
worth roughly 10x end to end and is the only performance lever that matters —
the pipeline is CNN inference, which already runs as compiled code regardless of
the language orchestrating it.

## Data flow

1. The user picks a folder with
   `<input type="file" webkitdirectory multiple accept="image/*">`. Nothing is
   uploaded; the browser holds `File` objects.
2. Each file is decoded with `createImageBitmap(file, { resizeWidth: 416 })`,
   detected, and the bitmap is released immediately. Downscaling before
   detection is what keeps a large album from exhausting the tab's memory, and
   detection cost scales with pixel count.
3. Up to 3 files are in flight at once, to keep the GPU busy during decode
   without holding many full-size bitmaps.
4. Every detected face contributes `{ file, faceIndex, descriptor, box }`.
5. Faces are clustered (below).
6. Each cluster renders as a person: a representative face crop with its photos
   beneath. A photo containing three faces appears under three people.
7. A person can be renamed inline and downloaded as a zip.

## Clustering

Greedy single-pass assignment against cluster centroids:

```
for each face:
  find the cluster whose centroid is nearest
  if that distance < 0.55 -> join it, update the centroid
  else                    -> start a new cluster
```

O(n·k), pure, and roughly twenty lines. The 0.55 threshold is face-api's usual
operating point and is a named constant so it can be tuned against real photos.

Greedy assignment is order-dependent and will occasionally split one person into
two clusters. A merge-two-people control fixes that case for far less work than
a hierarchical or density-based algorithm, and is deferred until real albums
show it is needed.

## Gating

Uses the `node:crypto` module; no dependency is added.

- `POST /api/auth { passphrase }` compares against the `APP_SECRET` environment
  variable with `timingSafeEqual`, then sets an httpOnly cookie of the form
  `expiry.hmac(expiry)` signed with `COOKIE_KEY`. Valid for 30 days.
- Middleware verifies the cookie on every route except the login page and
  `/api/auth`. A missing, tampered, or expired cookie returns the login page.
- An in-memory `Map` of IP to attempt timestamps limits `/api/auth` to 5 attempts
  per 15 minutes. The service runs as a single instance, so a Map is sufficient.

Gating a static app is usually theatre, but here the application bundle and the
models are genuinely not served without a valid cookie.

## Build and deploy

No bundler. The app is plain ES modules; face-api and jszip are vendored as dist
files. Render runs `npm install && node server.js` with nothing to build.

The one thing that could change this: if the WebGPU backend turns out to need a
TensorFlow.js backend package that the face-api dist does not bundle, adding
Vite may be cheaper than vendoring it by hand. This is checked during phase 1
rather than assumed.

Environment variables: `APP_SECRET`, `COOKIE_KEY`.

### Keep-alive cron

The GitHub Actions workflow that pings the service every minute is removed.
Render's free tier grants 750 instance-hours per month and a month is 720–744
hours, so staying awake consumes essentially the whole quota in order to avoid a
cold start. With no models to load, a bare Express cold start is roughly 15–30
seconds. The app shows a waking-up state instead.

## Error handling

A run never aborts partway. Specifically:

- A corrupt or undecodable file is skipped and counted in a "N skipped" notice.
- A photo with no detected faces goes to a "No faces" bucket.
- A detection that throws skips that file and the run continues.
- An invalid or expired cookie returns the login page rather than an error.

## Testing

`node --test`, no framework.

- `cluster.test.js` — identical descriptors cluster together; distant ones do
  not; a photo with two different faces appears in two clusters. This is where
  the shipped bug lived, so this is where the check goes.
- `auth.test.js` — a tampered cookie is rejected; an expired cookie is rejected.

## Phases

1. Server gate, static serving, detection, clustering, and a plain result grid.
   Working end to end.
2. Person naming, per-person zip download, no-faces bucket, skipped counter.
3. Deferred: remembered people in IndexedDB, a merge-two-people control, an
   opt-in `ssd_mobilenetv1` accuracy toggle, and splitting static assets onto a
   Render static site to avoid the cold start.

## Explicitly out of scope

A Rust implementation. The pipeline is CNN inference, which is already compiled
code; clustering is about five milliseconds of arithmetic for a thousand faces.
Rust would optimise the orchestration layer, which is not the bottleneck. A Rust
command-line tool for sorting a large local archive across all cores is a
genuinely different product and is not part of this work.
