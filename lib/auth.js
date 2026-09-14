import crypto from "node:crypto";

export const COOKIE_NAME = "sif_session";
export const SESSION_MS = 30 * 24 * 60 * 60 * 1000;

const ATTEMPT_MAX = 5;
const ATTEMPT_WINDOW_MS = 15 * 60 * 1000;

function sign(expiry, key) {
  return crypto.createHmac("sha256", key).update(String(expiry)).digest("hex");
}

/** Build a session token of the form `<expiry>.<hmac>`. */
export function makeToken(key, now = Date.now()) {
  const expiry = now + SESSION_MS;
  return `${expiry}.${sign(expiry, key)}`;
}

/** True only for a token that is well-formed, correctly signed, and unexpired. */
export function verifyToken(token, key, now = Date.now()) {
  if (typeof token !== "string") return false;

  const dot = token.indexOf(".");
  if (dot < 1) return false;

  const expiry = token.slice(0, dot);
  const signature = token.slice(dot + 1);
  const expected = sign(expiry, key);

  // timingSafeEqual throws on a length mismatch, so check that first.
  if (signature.length !== expected.length) return false;
  if (!crypto.timingSafeEqual(Buffer.from(signature), Buffer.from(expected))) {
    return false;
  }

  const expiresAt = Number(expiry);
  return Number.isFinite(expiresAt) && expiresAt > now;
}

/**
 * Compare a supplied passphrase against the configured one in constant time.
 * Both sides are hashed first so that differing lengths do not leak.
 */
export function secretMatches(given, expected) {
  if (typeof given !== "string" || typeof expected !== "string") return false;
  const a = crypto.createHash("sha256").update(given).digest();
  const b = crypto.createHash("sha256").update(expected).digest();
  return crypto.timingSafeEqual(a, b);
}

// ponytail: in-memory attempts, fine for one Render instance. Needs a shared
// store only if this ever runs on more than one.
const attempts = new Map();

/** Record an attempt for `ip`. False once the window's budget is spent. */
export function allowAttempt(ip, now = Date.now()) {
  const recent = (attempts.get(ip) ?? []).filter(
    (at) => now - at < ATTEMPT_WINDOW_MS,
  );

  if (recent.length >= ATTEMPT_MAX) {
    attempts.set(ip, recent);
    return false;
  }

  recent.push(now);
  attempts.set(ip, recent);

  // Keep the map from growing without bound across many distinct IPs.
  if (attempts.size > 5000) {
    for (const [key, times] of attempts) {
      if (times.every((at) => now - at >= ATTEMPT_WINDOW_MS)) attempts.delete(key);
    }
  }
  return true;
}

/** Read one cookie out of a raw Cookie header. */
export function readCookie(header, name) {
  if (!header) return null;
  for (const part of header.split(";")) {
    const eq = part.indexOf("=");
    if (eq < 0) continue;
    if (part.slice(0, eq).trim() !== name) continue;
    try {
      return decodeURIComponent(part.slice(eq + 1).trim());
    } catch {
      return null;
    }
  }
  return null;
}
