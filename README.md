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
- `API_MAX_DIM` — longest side images are scaled to server-side. Defaults to 640.

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
| `POST /api/group` | Blocking. Up to 40 images. |
| `POST /api/jobs` | Returns `202 {jobId}`. Up to 250 images. |
| `GET /api/jobs/:id` | Job status, then the same body as `/api/group`. |
| `GET /api/status` | Limits, queue depth, configured threshold. |
| `POST /` | The older `[{urls:[...]}]` shape, kept for existing callers. |

Optional `threshold` in the body overrides the clustering distance for that
request (default 0.6). Lower splits people apart, higher merges them together;
it is the quickest way to tune grouping without redeploying.

A photo that fails to download appears in `failed`; the rest of the batch still
returns. A photo with no faces appears in `noFaces`.

### Speed, and why there are two endpoints

Render's free tier gives 0.1 of a CPU. Locally a photo takes roughly 100–250ms
server-side; on the free tier expect something closer to one to three seconds
each. Use `POST /api/group` for small batches and `POST /api/jobs` beyond
roughly fifteen images, so the request does not sit open for a minute.

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
0.6, or starts its own. A photo with three people therefore appears under three
names.

No single threshold is right for every album, and this is not a tuning problem
that can be solved once. On a set of eight photos holding four people, the
descriptors for faces that are sideways, dark or blurry are simply not
separable: every threshold from 0.5 to 0.55 gave six people, 0.6 gave four,
0.62 gave three. Average-linkage agglomerative clustering produced identical
counts, so the algorithm is not the limit — the descriptors are. Raising the
working resolution from 1280 to 2048 did not improve separation either.

So the app exposes the threshold as a slider and lets you merge two groups by
hand, and the API takes `threshold` per request. Those two controls are more
reliable than any default.

**Quality gates.** Faces below 0.55 detector confidence or 30px are dropped. At
0.5 the detector reported a patterned jacket as a face, which then appeared as a
person of its own; tiny faces are worse than useless, because their descriptors
sit close to everything and pull unrelated people together. Both gates are kept
deliberately low so real faces survive.

**Detector.** `ssdMobilenetv1`. An earlier version shipped `tinyFaceDetector`
instead, which is a tenth of the size and three times faster — and on real phone
photos it found 1 face in a folder where this one finds 12. The demo photos have
large frontal faces and hid the difference completely. Accuracy wins.

**Backends.** WebGL, falling back to WASM and then CPU. WebGPU is not bundled
with the face-api build in use and would require adding a bundler, which is not
worth it until a real album shows WebGL is too slow.

**Photo size.** Photos are scaled so the longest side is 1280px before
detection. Detection cost scales with pixel count and the recognition network
only sees a small aligned crop, so a full-size original costs memory and buys
nothing.

See `docs/superpowers/specs/2026-09-14-face-sort-design.md` for the full design.
