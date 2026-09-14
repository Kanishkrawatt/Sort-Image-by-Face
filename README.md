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
rather than loading 6.4MB of weights on every wake.

| File | What it does |
|---|---|
| `server.js` | Auth route, session gate, static serving |
| `lib/auth.js` | HMAC session cookies and the login rate limiter |
| `public/detect.js` | Backend selection, model loading, per-photo detection |
| `public/cluster.js` | Groups descriptors into people |
| `public/app.js` | Orchestration, rendering, zip download |

## Running it locally

```sh
npm install
APP_SECRET=some-passphrase COOKIE_KEY=$(node -e "console.log(crypto.randomUUID())") npm run dev
```

Then open http://localhost:3000 and enter the passphrase.

Both variables are required and the server refuses to start without them.

- `APP_SECRET` — the passphrase that unlocks the app.
- `COOKIE_KEY` — signs session cookies. Changing it signs everyone out.

## Tests

```sh
npm test
```

Covers the clustering logic and the session cookie. Detection quality itself is
not unit tested; it is a property of the models, and the way to check it is to
run a real album through the app.

## Deploying to Render

`render.yaml` describes a free web service. Set `APP_SECRET` in the dashboard;
`COOKIE_KEY` is generated for you.

There is deliberately no keep-alive ping. The free tier grants 750 instance
hours a month and a month is 720–744 hours, so pinging to stay awake spends
essentially the whole budget to avoid a cold start of a few seconds.

## Notes on the design

**Clustering.** Each face joins the nearest cluster whose centroid is within
0.55, or starts its own. A photo with three people therefore appears under three
names. Greedy assignment is order-dependent and occasionally splits one person
in two; merging two people by hand is a smaller fix than a heavier algorithm.

**Backends.** WebGL, falling back to WASM and then CPU. WebGPU is not bundled
with the face-api build in use and would require adding a bundler, which is not
worth it until a real album shows WebGL is too slow.

**Photo size.** Photos are scaled so the longest side is 1280px before
detection. Detection cost scales with pixel count and the recognition network
only sees a small aligned crop, so a full-size original costs memory and buys
nothing.

See `docs/superpowers/specs/2026-09-14-face-sort-design.md` for the full design.
