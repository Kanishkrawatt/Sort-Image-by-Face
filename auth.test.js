import assert from "node:assert/strict";
import test from "node:test";
import {
  COOKIE_NAME,
  SESSION_MS,
  allowAttempt,
  makeToken,
  readCookie,
  secretMatches,
  verifyToken,
} from "./lib/auth.js";

const KEY = "test-cookie-key";

test("a freshly minted token verifies", () => {
  assert.ok(verifyToken(makeToken(KEY), KEY));
});

test("a token signed with another key is rejected", () => {
  assert.equal(verifyToken(makeToken("other-key"), KEY), false);
});

test("a tampered expiry is rejected", () => {
  const token = makeToken(KEY);
  const signature = token.slice(token.indexOf(".") + 1);
  const farFuture = Date.now() + 10 * SESSION_MS;
  assert.equal(verifyToken(`${farFuture}.${signature}`, KEY), false);
});

test("a tampered signature is rejected", () => {
  const token = makeToken(KEY);
  const [expiry, signature] = token.split(".");
  const flipped = (signature[0] === "a" ? "b" : "a") + signature.slice(1);
  assert.equal(verifyToken(`${expiry}.${flipped}`, KEY), false);
});

test("an expired token is rejected", () => {
  const issuedAt = Date.now() - SESSION_MS - 1000;
  assert.equal(verifyToken(makeToken(KEY, issuedAt), KEY), false);
});

test("a token valid yesterday still works today", () => {
  const issuedAt = Date.now() - 24 * 60 * 60 * 1000;
  assert.ok(verifyToken(makeToken(KEY, issuedAt), KEY));
});

test("malformed tokens are rejected rather than throwing", () => {
  for (const bad of ["", ".", "nodot", ".sig", "123.", null, undefined, 42, {}]) {
    assert.equal(verifyToken(bad, KEY), false, `should reject ${JSON.stringify(bad)}`);
  }
});

test("passphrase comparison accepts only an exact match", () => {
  assert.ok(secretMatches("hunter2", "hunter2"));
  assert.equal(secretMatches("hunter3", "hunter2"), false);
  assert.equal(secretMatches("hunter", "hunter2"), false, "prefix must not pass");
  assert.equal(secretMatches("", "hunter2"), false);
  assert.equal(secretMatches(undefined, "hunter2"), false);
});

test("login attempts are capped, and the window expires", () => {
  const ip = "198.51.100.7";
  const start = Date.now();

  for (let i = 0; i < 5; i++) {
    assert.ok(allowAttempt(ip, start), `attempt ${i + 1} should be allowed`);
  }
  assert.equal(allowAttempt(ip, start), false, "sixth attempt is refused");

  const afterWindow = start + 16 * 60 * 1000;
  assert.ok(allowAttempt(ip, afterWindow), "allowed again once the window passes");
});

test("attempt budgets are tracked per IP", () => {
  const now = Date.now();
  for (let i = 0; i < 5; i++) allowAttempt("203.0.113.1", now);
  assert.equal(allowAttempt("203.0.113.1", now), false);
  assert.ok(allowAttempt("203.0.113.2", now), "a different IP is unaffected");
});

test("cookies are read out of a header, and absent ones give null", () => {
  const header = `other=1; ${COOKIE_NAME}=abc.def; trailing=2`;
  assert.equal(readCookie(header, COOKIE_NAME), "abc.def");
  assert.equal(readCookie("other=1", COOKIE_NAME), null);
  assert.equal(readCookie(undefined, COOKIE_NAME), null);
  assert.equal(readCookie(`${COOKIE_NAME}x=nope`, COOKIE_NAME), null, "no prefix match");
});
