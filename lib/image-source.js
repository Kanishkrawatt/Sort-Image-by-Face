// Fetching caller-supplied image URLs safely.
//
// The service this replaced would fetch any URL it was handed, which made it a
// server-side request forgery tool: a caller could point it at an internal
// address and read the response. Every guard here exists for that reason.

import dnsPromises from "node:dns/promises";
import http from "node:http";
import https from "node:https";
import net from "node:net";

export const MAX_BYTES = 15 * 1024 * 1024;
export const FETCH_TIMEOUT_MS = 15000;
const MAX_REDIRECTS = 3;

/** Hosts the caller is allowed to reach, or null for "any public address". */
export function allowedHosts(raw = process.env.ALLOWED_IMAGE_HOSTS) {
  if (!raw) return null;
  const hosts = raw.split(",").map((h) => h.trim().toLowerCase()).filter(Boolean);
  return hosts.length ? hosts : null;
}

function ipv4IsPrivate(ip) {
  const [a, b] = ip.split(".").map(Number);
  if (a === 0 || a === 10 || a === 127) return true;
  if (a === 169 && b === 254) return true; // link-local, incl. cloud metadata
  if (a === 172 && b >= 16 && b <= 31) return true;
  if (a === 192 && b === 168) return true;
  if (a === 100 && b >= 64 && b <= 127) return true; // carrier NAT
  if (a === 198 && (b === 18 || b === 19)) return true; // benchmarking
  if (a >= 224) return true; // multicast, reserved, broadcast
  return false;
}

/** True for any address a caller must not be able to reach through us. */
export function isPrivateAddress(ip) {
  const version = net.isIP(ip);
  if (version === 4) return ipv4IsPrivate(ip);
  if (version !== 6) return true; // not an address at all

  const lower = ip.toLowerCase();
  if (lower === "::" || lower === "::1") return true;
  // IPv4 mapped or compatible: judge the embedded address.
  const mapped = lower.match(/^::(?:ffff:)?(\d+\.\d+\.\d+\.\d+)$/);
  if (mapped) return ipv4IsPrivate(mapped[1]);

  const head = parseInt(lower.split(":")[0] || "0", 16);
  if ((head & 0xfe00) === 0xfc00) return true; // fc00::/7 unique local
  if ((head & 0xffc0) === 0xfe80) return true; // fe80::/10 link local
  return false;
}

/**
 * Check a URL is one we are willing to fetch, and resolve it to a pinned
 * address so the host cannot re-resolve to something private afterwards.
 */
export async function resolveTarget(rawUrl, { hosts = allowedHosts(), resolver = dnsPromises } = {}) {
  let url;
  try {
    url = new URL(rawUrl);
  } catch {
    throw new Error("not a valid URL");
  }

  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error(`unsupported protocol ${url.protocol}`);
  }

  // URL keeps IPv6 literals in brackets; strip them so they parse as addresses.
  const hostname = url.hostname.toLowerCase().replace(/^\[(.+)\]$/, "$1");
  if (hosts && !hosts.includes(hostname)) {
    throw new Error(`host ${hostname} is not in ALLOWED_IMAGE_HOSTS`);
  }

  // A literal address needs no lookup, but still needs checking.
  if (net.isIP(hostname)) {
    if (isPrivateAddress(hostname)) throw new Error(`address ${hostname} is not public`);
    return { url, address: hostname, family: net.isIP(hostname) };
  }

  let records;
  try {
    records = await resolver.lookup(hostname, { all: true });
  } catch {
    throw new Error(`could not resolve ${hostname}`);
  }

  const usable = records.find((r) => !isPrivateAddress(r.address));
  if (!usable) throw new Error(`${hostname} resolves only to non-public addresses`);
  return { url, address: usable.address, family: usable.family };
}

function requestOnce({ url, address, family }) {
  const client = url.protocol === "https:" ? https : http;

  return new Promise((resolve, reject) => {
    const request = client.request(
      url,
      {
        // Pin the address we already validated. The hostname is still used for
        // TLS and the Host header, so certificates verify normally. Node calls
        // this with `all` set in some paths, and then wants an array back.
        lookup: (_host, options, cb) =>
          options?.all ? cb(null, [{ address, family }]) : cb(null, address, family),
        headers: { accept: "image/*" },
        timeout: FETCH_TIMEOUT_MS,
      },
      (response) => resolve(response),
    );

    request.on("timeout", () => request.destroy(new Error("timed out")));
    request.on("error", reject);
    request.end();
  });
}

function readCapped(response) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let total = 0;
    response.on("data", (chunk) => {
      total += chunk.length;
      if (total > MAX_BYTES) {
        response.destroy();
        reject(new Error(`image exceeds ${MAX_BYTES} bytes`));
        return;
      }
      chunks.push(chunk);
    });
    response.on("end", () => resolve(Buffer.concat(chunks)));
    response.on("error", reject);
  });
}

/**
 * Download one image. Redirects are followed by hand so that each hop is
 * validated too — otherwise a redirect to an internal address would walk
 * straight past the checks above.
 */
export async function fetchImage(rawUrl, options = {}) {
  let target = await resolveTarget(rawUrl, options);

  for (let hop = 0; hop <= MAX_REDIRECTS; hop++) {
    const response = await requestOnce(target);
    const status = response.statusCode ?? 0;

    if (status >= 300 && status < 400 && response.headers.location) {
      response.resume(); // discard the body
      if (hop === MAX_REDIRECTS) throw new Error("too many redirects");
      const next = new URL(response.headers.location, target.url).toString();
      target = await resolveTarget(next, options);
      continue;
    }

    if (status !== 200) {
      response.resume();
      throw new Error(`responded ${status}`);
    }

    const type = String(response.headers["content-type"] ?? "");
    if (type && !type.startsWith("image/")) {
      response.resume();
      throw new Error(`not an image (${type})`);
    }

    const declared = Number(response.headers["content-length"] ?? 0);
    if (declared > MAX_BYTES) {
      response.resume();
      throw new Error(`image exceeds ${MAX_BYTES} bytes`);
    }

    return await readCapped(response);
  }

  throw new Error("too many redirects");
}
