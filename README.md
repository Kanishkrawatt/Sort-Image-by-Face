# Sort Image by Face

Point it at a folder of photos and it groups them by who appears in them.
Face detection runs in your browser — the photos are never uploaded.

## How it works

The server does almost nothing. It checks a passphrase, sets a signed cookie,
and serves static files. All the face work happens on the client:

```
BROWSER                                   SERVER (Render free tier)
  pick a folder                             POST /api/auth  passphrase -> cookie
  decode + downscale each photo             GET  /*         static, cookie-gated
  detect faces, take a descriptor
  cluster descriptors into people
  zip a person's photos on demand
```

Keeping inference in the browser is what makes the free tier viable: the server
holds no models, so it needs roughly 50MB of RAM and starts cold in seconds
rather than loading 12MB of weights on every wake.

| File | What it does |
|---|---|
| `server.js` | Auth route, session gate, static serving |
| `lib/auth.js` | HMAC session cookies and the login rate limiter |
| `public/detect.js` | Backend selection, model loading, per-photo detection |
| `public/cluster.js` | Groups descriptors into people |
| `public/app.js` | Orchestration, rendering, zip download |
| `lib/api.js` | The machine-facing API routes |
| `lib/image-source.js` | Guarded downloading of caller-supplied URLs |
| `lib/worker.js` | Child process that does server-side detection |
| `lib/runner.js` | Queues jobs, one child process at a time |

## Running it locally

```sh
npm install
APP_SECRET=some-passphrase COOKIE_KEY=$(node -e "console.log(crypto.randomUUID())") npm run dev
```

Then open http://localhost:3000 and enter the passphrase.

Both variables are required and the server refuses to start without them.

- `APP_SECRET` — the passphrase that unlocks the app. Required.
- `COOKIE_KEY` — signs session cookies. Required. Changing it signs everyone out.
- `API_KEY` — enables the machine-facing API. Optional; the API is off without it.
- `ALLOWED_IMAGE_HOSTS` — comma-separated hosts the API may download from.
  Optional; any public host is allowed if unset.
- `API_CORS_ORIGIN` — comma-separated origins allowed to call the API from a
  browser. Optional.
- `API_MAX_DIM` — longest side images are scaled to server-side. Defaults to 1024.
- `API_CHUNK_SIZE` — images per child process before it is replaced. Defaults to 4.

## Tests

```sh
npm test
```

Thirty tests covering the clustering logic, session cookies, and the URL guard
that stops the API being used to reach private addresses.

Detection quality itself is not unit tested; it is a property of the models, and
the way to check it is to run a real album through the app. As a cross-check,
the browser and the server agree exactly on the demo photos: both find 20 faces
across the six samples.

## Deploying to Render

`render.yaml` describes a free web service. Set `APP_SECRET` in the dashboard;
`COOKIE_KEY` is generated for you.

**A service created before that file exists does not use it.** It keeps whatever
build and start commands were typed into its dashboard, and a first deploy of
this code onto such a service fails in a way that does not look like a
misconfiguration. Check all four of these on the service itself:

| Setting | Must be | Why |
|---|---|---|
| Build command | `npm ci --omit=dev` | `yarn` ignores `package-lock.json` and resolves fresh, so the versions that run are not the versions that were tested. |
| Start command | `npm start` | `yarn dev` runs `node --watch`, which keeps the process alive after a fatal startup error instead of exiting. Render then reports "no open ports detected" rather than the actual error. |
| Environment | `APP_SECRET`, `API_KEY`, `ALLOWED_IMAGE_HOSTS` | The server exits deliberately if `APP_SECRET` or `COOKIE_KEY` is missing. |
| Node version | from `.node-version` | `engines` alone says `>=20`, which Render reads as "newest available". That has already meant Node 26, whose ABI may have no prebuilt `sharp` binary. |

The `node --watch` point is the one that wastes time. The process prints its
error, declines to exit, binds nothing, and the deploy times out on a port scan
several minutes later with the real cause scrolled far above.

There is deliberately no keep-alive ping. The free tier grants 750 instance
hours a month and a month is 720–744 hours, so pinging to stay awake spends
essentially the whole budget to avoid a cold start of a few seconds.

## Using it as an API

Other services can post image URLs and get people back. This path does the face
work on the server, so it is far slower than the browser app — but it needs no
browser at all.

The API is disabled until `API_KEY` is set, and every request must carry it as
`x-api-key`.

```sh
curl -X POST https://your-service.onrender.com/api/group \
  -H 'content-type: application/json' \
  -H 'x-api-key: YOUR_KEY' \
  -d '{"imageUrls": ["https://.../a.jpg", "https://.../b.jpg"]}'
```

```json
{
  "people": [
    { "id": 0, "photos": ["https://.../a.jpg"], "faceCount": 2,
      "faces": [{ "url": "https://.../a.jpg", "score": 0.97, "box": {"x":1,"y":2,"width":3,"height":4} }] }
  ],
  "noFaces": [],
  "failed": [{ "url": "https://.../c.jpg", "error": "responded 404" }],
  "stats": { "images": 2, "processed": 2, "faces": 3, "people": 1, "threshold": 0.55, "ms": 1820 }
}
```

| Route | Purpose |
|---|---|
| `POST /api/group` | Blocking. Up to 8 images. |
| `POST /api/jobs` | Returns `202 {jobId}`. Up to 250 images. |
| `GET /api/jobs/:id` | Job status; the result is nested under `result`. |
| `POST /api/compare` | Are these two photos the same person? See the warning below. |
| `GET /api/status` | Limits, queue depth, configured threshold, route list. |
| `POST /` | The older `[{urls:[...]}]` shape, kept for existing callers. |

Optional `threshold` in the body overrides the clustering distance for that
request (default 0.9). Lower splits people apart, higher merges them together;
it is the quickest way to tune grouping without redeploying.

A photo that fails to download appears in `failed`; the rest of the batch still
returns. A photo with no faces appears in `noFaces`.

### Comparing two photos

```sh
curl -X POST .../api/compare -H 'x-api-key: KEY' -H 'content-type: application/json' \
  -d '{"a": "https://.../reference.jpg", "b": "https://.../capture.jpg"}'
```

```json
{
  "match": false, "distance": 0.8601, "threshold": 0.65,
  "a": {"faces": 3, "score": 0.989, "box": {"x": 1206, "y": 128, "width": 161, "height": 203}},
  "b": {"faces": 5, "score": 0.9923, "box": {"x": 656, "y": 91, "width": 130, "height": 177}}
}
```

The clearest face in each photo is compared. `faces` tells you how many were
found: anything above one means the answer is ambiguous and you should send a
cropped, single-face image instead. A photo with no face returns 422 rather
than a verdict, so "no face" is never silently reported as "no match".

Use `distance`, not `match`. The boolean is only `distance < threshold`, and
the right threshold depends on your photos.

**This is a similarity measurement, not an identity check, and it must not gate
access to anything.** It compares two images and cannot tell a person from a
photograph of that person, so anyone holding a picture of the subject passes.
Grouping accuracy is good enough to sort an album and wrong often enough to
matter for anything else: a face turned far away, in deep shadow, or motion
blurred can still land on the wrong side of any threshold. Treat a match as a
hint for sorting and labelling. Anything that must actually be enforced needs
real authentication and authorisation on the server holding the data.

A finished job wraps the result rather than returning it directly:

```json
{"id": "…", "status": "done", "images": 3, "updatedAt": 1789416919203,
 "result": {"people": [], "noFaces": [], "failed": [], "stats": {}}}
```

So read `body.result`, not `body.people`. `status` is `queued`, `running`,
`done` or `error`.

### Batch size, and why it is small

Detection runs in a child process that handles four images and is then
replaced. TensorFlow's WebAssembly heap grows as images are processed and never
gives memory back, so a long run creeps towards the instance limit and is
killed. On a 512MB instance, eight images at 1600px was enough to take the whole
service down — a hard restart, which also lost whatever was queued. Replacing
the process keeps peak memory tied to the chunk size instead of the batch size.

A chunk that dies now costs only its own images: they come back in `failed` and
the remaining chunks still run. The request succeeds with a partial result
rather than failing outright.

`API_MAX_DIM` defaults to 1024 for the same reason. Detection quality is flat
across this range — the same faces are found at 1024 as at 1600 — so the larger
size bought nothing but memory pressure. `API_CHUNK_SIZE` (default 4) tunes the
rest.

Downscale on your side too if you can. A Cloudinary `w_1024` derivative is less
to download, less to decode, and measurably faster.

### Speed, and why there are two endpoints

Render's free tier gives 0.1 of a CPU. Locally a photo takes roughly 100–250ms
server-side; on the free tier expect something closer to one to three seconds
each. Use `POST /api/group` for small batches and `POST /api/jobs` beyond
four or five images, so the request does not sit open past a caller's own
timeout — eight photos measured around 45 seconds on the free tier, and many
serverless callers cap out at 60.

Jobs are held in memory for 30 minutes and are lost if the service restarts,
which on the free tier happens whenever it sleeps. Poll promptly.

Detection runs in a child process that exits when the job finishes. TensorFlow
settles at around 400MB resident and the instance only has 512MB, so the memory
has to go back to the operating system between jobs. Only one job runs at a
time for the same reason.

### Calling it from cloudbox

`pages/api/smartgroup.ts` already posts `{ imageUrls }` and expects
`[{urls:[...]}]`, so the only change is the key:

```ts
axios.post(`${url}`, body, {
  headers: { "x-api-key": process.env.FACE_API_KEY },
});
```

Set `FACE_API_KEY` in cloudbox and the matching `API_KEY` on this service. Since
cloudbox keeps its bytes in Cloudinary, restrict downloads to it:

```
ALLOWED_IMAGE_HOSTS=res.cloudinary.com
```

Prefer `/api/jobs` there: the existing comment in that file says the response
"[takes] too much time", which is what a blocking call on a tenth of a CPU feels
like.

### Fetching is guarded

The service this replaced would fetch any URL it was given, which made it usable
as a way to reach private infrastructure. Now every URL is checked before and
after each redirect: only `http` and `https`, only public addresses — loopback,
private ranges, carrier NAT, multicast and link-local (including cloud metadata
at `169.254.169.254`) are all refused — with the resolved address pinned for the
connection so the name cannot change under us, a 15MB ceiling, a 15 second
timeout, and an `image/*` content type. `ALLOWED_IMAGE_HOSTS` narrows it further.

## Notes on the design

**Clustering.** Each face joins the nearest cluster whose centroid is within
0.9 — provided that cluster holds no other face from the same photo.

Embeddings are unit length, so distances run from 0 to 2. The default sits in
the middle of the measured gap: 0.579 was the furthest two faces of one person
ever fell apart, and 1.220 the closest two faces of different people came
together.

That last rule matters more than any tuning. Nobody appears twice in one frame,
so two faces in a photo are two people. It is knowledge the descriptors do not
have, and without it everyone in a group shot tends to collapse into a single
person as the threshold rises. With it, a photo of three people appears under
all three names, and the result stops being sensitive to the exact threshold:
on a real album of four people, every value from 0.64 to 0.72 recovers exactly
four, where before only a knife-edge did.

No single threshold is right for every album. The default suits a personal
album of a few friends; a set full of strangers over-merges at 0.65 and wants
something nearer 0.55.

Descriptor quality is still the ceiling. Faces that are sideways, dark or
blurry produce descriptors that are not cleanly separable, and no threshold
fixes that. Average-linkage agglomerative clustering produced counts identical
to this greedy pass, so the algorithm is not the limit, and raising the working
resolution from 1280 to 2048 did not improve separation either.

So the app exposes the threshold as a slider and lets you merge two groups by
hand, and the API takes `threshold` per request. Those two controls are more
reliable than any default.

**Quality gates.** Faces below 0.5 detector confidence or 24px are dropped.
Tiny faces are worse than useless: their descriptors sit close to everything and
pull unrelated people together. The gates are deliberately loose — tightening
them to 0.55 and 30px cost real faces in dark photos, and an occasional false
positive is easier to live with than a missing person.

**Backend.** WebAssembly with SIMD, multi-threaded. The server sets
`Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` so the page is
cross-origin isolated, which is what permits threads; everything the app loads
is same-origin, so nothing is lost by it. Responses are gzipped, which takes the
runtime from 13.3MB to 3.4MB on the wire. WebGPU is not bundled
with the face-api build in use and would require adding a bundler, which is not
worth it until a real album shows WebGL is too slow.

**Photo size.** Photos are scaled so the longest side is 1280px before
detection. Detection cost scales with pixel count and the recognition network
only sees a small aligned crop, so a full-size original costs memory and buys
nothing.

See `docs/superpowers/specs/2026-09-14-face-sort-design.md` for the full design.
