import path from "node:path";
import { fileURLToPath } from "node:url";
import compression from "compression";
import express from "express";
import { createApiRouter, legacyHandler } from "./lib/api.js";
import {
  COOKIE_NAME,
  SESSION_MS,
  allowAttempt,
  makeToken,
  readCookie,
  secretMatches,
  verifyToken,
} from "./lib/auth.js";

const root = path.dirname(fileURLToPath(import.meta.url));
const port = process.env.PORT || 3000;

const APP_SECRET = process.env.APP_SECRET;
const COOKIE_KEY = process.env.COOKIE_KEY;

if (!APP_SECRET || !COOKIE_KEY) {
  console.error(
    "Missing configuration. Set both before starting:\n" +
      "  APP_SECRET  the passphrase that unlocks the app\n" +
      "  COOKIE_KEY  a random string used to sign session cookies\n\n" +
      "Generate a key with:  node -e \"console.log(crypto.randomUUID())\"",
  );
  process.exit(1);
}

const app = express();
app.disable("x-powered-by");

// The WebAssembly runtime is 13MB uncompressed and under 4MB gzipped, so this
// is the difference between a tolerable first visit and an unusable one.
app.use(compression());

// Cross-origin isolation, which is what lets the runtime use more than one
// thread. Everything this app loads is same-origin, so nothing is lost.
app.use((_req, res, next) => {
  res.set("Cross-Origin-Opener-Policy", "same-origin");
  res.set("Cross-Origin-Embedder-Policy", "require-corp");
  next();
});
app.set("trust proxy", 1); // Render terminates TLS in front of us.

app.get("/healthz", (_req, res) => res.type("text").send("ok"));

const publicDir = path.join(root, "public");

app.get("/login", (_req, res) => {
  res.sendFile(path.join(publicDir, "login.html"));
});

// The login page's own assets, served before the gate so it is not unstyled.
// Only these exact paths match, so there is nothing to traverse.
app.get(["/style.css", "/favicon.svg"], (req, res) => {
  res.sendFile(path.join(publicDir, req.path));
});

app.post("/api/auth", express.json({ limit: "1kb" }), (req, res) => {
  if (!allowAttempt(req.ip)) {
    return res.status(429).json({ error: "Too many attempts. Try again later." });
  }
  if (!secretMatches(req.body?.passphrase, APP_SECRET)) {
    return res.status(401).json({ error: "Wrong passphrase." });
  }

  res.cookie(COOKIE_NAME, makeToken(COOKIE_KEY), {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    maxAge: SESSION_MS,
  });
  res.json({ ok: true });
});

// The machine-facing API. It authenticates with x-api-key rather than the
// session cookie, so it is mounted ahead of the gate below.
app.use("/api", createApiRouter());

// The shape cloudbox already posts to. Kept so that integration needs no
// change beyond adding its API key.
app.post("/", express.json({ limit: "1mb" }), legacyHandler());

// Everything past this point requires a valid session.
app.use((req, res, next) => {
  const token = readCookie(req.headers.cookie, COOKIE_NAME);
  if (verifyToken(token, COOKIE_KEY)) return next();

  res.clearCookie(COOKIE_NAME);
  if (req.accepts("html")) return res.redirect(302, "/login");
  res.status(401).json({ error: "Not authenticated." });
});

app.use(express.static(publicDir));

// Models are content-addressed by their filename and never change in place.
app.use(
  "/models",
  express.static(path.join(root, "models"), {
    maxAge: "30d",
    immutable: true,
  }),
);

app.listen(port, () => {
  console.log(`Listening on http://localhost:${port}`);
});

export default app;
