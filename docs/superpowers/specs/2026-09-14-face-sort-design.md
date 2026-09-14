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

Reduced from 13MB to 12MB by deleting what is never used:

| Model | Size | Kept |
|---|---|---|
| `ssd_mobilenetv1` | 5.6MB | yes — the detector |
| `face_landmark_68_tiny` | 77KB | yes |
| `face_recognition` | 6.4MB | yes — the descriptor net |
| `tiny_face_detector` | 193KB | no — see below |
| `age_gender` | 430KB | no |
| `face_expression` | 330KB | no |

An earlier version of this design kept `tiny_face_detector` and dropped
`ssd_mobilenetv1`, cutting the download to 6.7MB. That was wrong. On a real
folder of phone photos the tiny detector found **1 face where SSD finds 12**;
every test until then used the library's demo photos, which have large frontal
faces and hide the difference entirely. A face sorter that misses eleven faces
in twelve is not worth 5.6MB of saved download, so SSD is the detector and the
size saving is given back.

### Library

`face-api.js` is replaced with `@vladmandic/face-api`, a maintained fork with a
compatible API and current TensorFlow.js. The original's last release was 2020
and pins tfjs 1.x.

Backend preference at runtime: WebGL, then WASM, then CPU. This choice is
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

This was checked rather than assumed: the `@vladmandic/face-api` bundle ships
TensorFlow.js 4.22.0 with the CPU, WebGL and WASM backends, and no WebGPU. Using
WebGPU would mean adding a bundler and the no-bundle build, so it is deferred
until a real album shows WebGL is too slow.

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

## Implementation notes

Three things only surfaced once the code ran in a real browser.

**face-api rejects an `ImageBitmap`.** Its `toNetInput` accepts an image, video,
canvas or tensor. Worse, in the chained
`.withFaceLandmarks().withFaceDescriptors()` form the rejection is swallowed and
the promise never settles, so passing a bitmap hangs forever instead of
throwing — the user would have seen a frozen progress bar. Photos are therefore
decoded to an `ImageBitmap` for cheap downscaling and then drawn into a canvas,
which is what detection receives.

**The first inference costs seconds, not milliseconds.** Compiling the WebGL
shaders took roughly six seconds on first use. `init` now runs one throwaway
detection so that cost is paid while the loading message is on screen; after
that, a photo takes 40–130ms.

**The login page needs its stylesheet before the gate.** `style.css` and
`favicon.svg` are served ahead of the session check; everything else, including
the models and the app code, stays behind it.

No Web Worker was built. Detection awaits between photos, which yields to the
event loop often enough for the progress bar to paint, and face-api's browser
environment detection assumes a DOM. Revisit only if the UI actually janks.

## Phases

1. Server gate, static serving, detection, clustering, and a plain result grid.
   Working end to end.
2. Person naming, per-person zip download, no-faces bucket, skipped counter.
3. Deferred: remembered people in IndexedDB, a merge-two-people control, an
   opt-in `ssd_mobilenetv1` accuracy toggle, a Web Worker if the UI janks, the
   WebGPU backend, and splitting static assets onto a Render static site to
   avoid the cold start.

## Addendum: the machine-facing API

Added after the browser app, so that cloudbox — a Next.js app whose
`pages/api/smartgroup.ts` already posts `{ imageUrls }` and expects
`[{urls:[...]}]` — can use this service without a browser in the loop.

This deliberately reintroduces server-side inference, which the main design
removed. The constraint that made that decision still applies, so it is
contained rather than ignored.

### What the measurements said

Running face-api in Node through the WebAssembly backend, with sharp decoding
straight to a tensor, avoids `node-canvas` and its native build entirely.
Measured on the demo photos:

| Input size | Faces found | Peak RSS |
|---|---|---|
| 1280px | 29 | 462MB |
| 800px | 29 | 435MB |
| 640px | 29 | 415MB |

Face counts are identical at all three, and match the browser exactly, so 640px
is the default: it finds the same faces for the least memory. Even so, 415MB
against a 512MB instance leaves too little headroom to hold across requests.

### Consequences

**Detection runs in a forked child process that exits when the job ends**, so
its memory returns to the operating system and the web server stays at roughly
50MB. A cold worker costs 130–200ms locally, which is negligible next to the
inference itself. **Only one job runs at a time**: two TensorFlow processes do
not fit in 512MB.

**Two endpoints, not one.** `POST /api/group` blocks and takes up to 40 images;
`POST /api/jobs` returns a job id and takes up to 250. On a tenth of a CPU a
photo costs on the order of a second, so a blocking call for a large album would
hold a request open for minutes — which is what the existing comment in
cloudbox's `smartgroup.ts`, "Response is coming but taking too much time",
already describes. Jobs live in memory for 30 minutes and are lost on restart;
on the free tier that means whenever the service sleeps.

**Fetching caller-supplied URLs is the old SSRF**, so it is guarded rather than
trusted: http and https only, public addresses only (loopback, RFC1918, carrier
NAT, multicast and link-local — including cloud metadata at 169.254.169.254 —
are refused), each redirect hop revalidated, the resolved address pinned for the
connection so the name cannot change between check and use, a 15MB ceiling, a 15
second timeout, and an `image/*` content type. `ALLOWED_IMAGE_HOSTS` narrows it
to named hosts; for cloudbox that is `res.cloudinary.com`.

**The API is disabled unless `API_KEY` is set**, so it can never be accidentally
open. Keys are compared with a constant-time hash comparison, reusing the helper
written for the browser login.

Clustering is shared: the API imports the same `public/cluster.js` the browser
uses, so both paths group identically and there is one implementation to test.

## Addendum: what tuning could and could not fix

A real album of eight photos containing four people was grouped into six. The
investigation that followed is worth recording, because it ends in a ceiling
rather than a fix.

**What was wrong and got fixed.** The detector reported a patterned jacket as a
face at confidence 0.50, which became a person of its own. Faces as small as
24px were being kept, and their descriptors sit close to every other descriptor,
so they do not merely form their own group — they pull unrelated people
together. Confidence below 0.55 and faces below 30px are now dropped, and the
full 68-point landmark net replaces the tiny one, since descriptors are taken
from a crop aligned by those points and many real photos have faces at an angle.

**What did not help.** Raising the working resolution to 2048 left descriptor
separation unchanged (nearest-neighbour distance 0.418 against 0.403 at 1280)
and crashed the tab with three photos in flight. Average-linkage agglomerative
clustering returned counts identical to the greedy pass at every threshold, so
the clustering algorithm is not the limiting factor. Full landmarks moved the
minimum pair distance from 0.420 to 0.407 — real, but small.

**The ceiling.** Across that album the pairwise descriptor distances ran from
0.40 to 0.93 with a median of 0.65, and same-person pairs were not separated
from different-person pairs by any margin. Photos with sideways, dark or blurry
faces are simply outside what this recognition model resolves. Every parameter
set that reported exactly four people did so by discarding most of the album:
confidence 0.75 with a 50px floor reached four, from five faces across three of
the eight photos.

**What actually fixed it.** Two faces in the same photo are two different
people. Nobody appears twice in one frame, so a cluster never accepts a second
face from a photo it already holds. This is knowledge the descriptors do not
carry, and it changes the character of the result: without it, raising the
threshold collapsed everyone in a group shot into one person, so the count fell
off a cliff and only a knife-edge value looked correct. With it, the same album
returns four people for every threshold from 0.64 to 0.72 — a plateau that wide
is evidence the answer is right rather than lucky — and a photo of three people
appears under all three names.

The quality gates were also loosened back to 0.5 confidence and 24px, having
been set too tight: at 0.55 and 30px they discarded real faces in dark photos,
leaving four faces where twelve were available. An occasional false positive is
easier to live with than a missing person, and the merge control exists for the
former.

The default threshold is 0.65, in the middle of that plateau. It is not
universal: an album full of strangers over-merges there and wants nearer 0.55.
Personal albums of a few friends are the case this serves.

**What was done alongside.** The threshold became a slider over cached
descriptors, so moving it regroups instantly without re-reading a photo, and two
groups can be merged by hand. After the gates and the landmark change the curve
is at least monotonic and legible — 0.50 and 0.55 give six people, 0.58 five,
0.60 four, 0.62 three — where before it jumped from seven to two. The default is
0.6, which is face-api's own default and recovers exactly the four people in
that album.

Names are stored against a face's stable identity rather than a group number, so
renaming survives both re-grouping and merging.

## Explicitly out of scope

A Rust implementation. The pipeline is CNN inference, which is already compiled
code; clustering is about five milliseconds of arithmetic for a thousand faces.
Rust would optimise the orchestration layer, which is not the bottleneck. A Rust
command-line tool for sorting a large local archive across all cores is a
genuinely different product and is not part of this work.
