import assert from "node:assert/strict";
import test from "node:test";
import { allowedHosts, isPrivateAddress, resolveTarget } from "./lib/image-source.js";

const publicResolver = { lookup: async () => [{ address: "93.184.216.34", family: 4 }] };
const privateResolver = { lookup: async () => [{ address: "127.0.0.1", family: 4 }] };
const metadataResolver = { lookup: async () => [{ address: "169.254.169.254", family: 4 }] };

test("loopback, private and link-local addresses are refused", () => {
  const blocked = [
    "127.0.0.1", "127.9.9.9", "0.0.0.0", "10.0.0.5", "10.255.255.255",
    "172.16.0.1", "172.31.255.255", "192.168.1.1", "100.64.0.1",
    "169.254.169.254", // cloud instance metadata
    "198.18.0.1", "224.0.0.1", "255.255.255.255",
    "::1", "::", "fc00::1", "fd12:3456::1", "fe80::1",
    "::ffff:127.0.0.1", "::ffff:10.0.0.1",
  ];
  for (const ip of blocked) {
    assert.equal(isPrivateAddress(ip), true, `${ip} must be refused`);
  }
});

test("ordinary public addresses are allowed", () => {
  for (const ip of ["8.8.8.8", "1.1.1.1", "93.184.216.34", "172.32.0.1", "192.167.1.1", "2606:4700::1111"]) {
    assert.equal(isPrivateAddress(ip), false, `${ip} should be allowed`);
  }
});

test("anything that is not an address is refused", () => {
  for (const junk of ["", "not-an-ip", "999.1.1.1", "localhost"]) {
    assert.equal(isPrivateAddress(junk), true);
  }
});

test("only http and https are fetched", async () => {
  for (const url of ["file:///etc/passwd", "ftp://example.com/x.jpg", "data:image/png;base64,AAAA"]) {
    await assert.rejects(
      () => resolveTarget(url, { hosts: null, resolver: publicResolver }),
      /unsupported protocol|not a valid URL/,
    );
  }
});

test("a malformed URL is refused", async () => {
  await assert.rejects(() => resolveTarget("http://", { resolver: publicResolver }), /not a valid URL/);
});

test("a host resolving to a private address is refused", async () => {
  await assert.rejects(
    () => resolveTarget("http://evil.test/x.jpg", { hosts: null, resolver: privateResolver }),
    /only to non-public/,
  );
});

test("a host resolving to cloud metadata is refused", async () => {
  await assert.rejects(
    () => resolveTarget("http://metadata.test/x.jpg", { hosts: null, resolver: metadataResolver }),
    /only to non-public/,
  );
});

test("a literal private address is refused without any lookup", async () => {
  const exploding = { lookup: async () => { throw new Error("should not be called"); } };
  await assert.rejects(
    () => resolveTarget("http://169.254.169.254/latest/meta-data/", { hosts: null, resolver: exploding }),
    /not public/,
  );
});

test("bracketed IPv6 literals are parsed, not sent to DNS", async () => {
  const exploding = { lookup: async () => { throw new Error("should not be called"); } };
  await assert.rejects(
    () => resolveTarget("http://[::1]:8080/x.jpg", { hosts: null, resolver: exploding }),
    /not public/,
  );
  await assert.rejects(
    () => resolveTarget("http://[fd00::1]/x.jpg", { hosts: null, resolver: exploding }),
    /not public/,
  );
  const ok = await resolveTarget("http://[2606:4700::1111]/x.jpg", { hosts: null, resolver: exploding });
  assert.equal(ok.address, "2606:4700::1111");
});

test("a public host resolves and pins its address", async () => {
  const target = await resolveTarget("https://example.test/a.jpg", { hosts: null, resolver: publicResolver });
  assert.equal(target.address, "93.184.216.34");
  assert.equal(target.url.hostname, "example.test");
});

test("the allowlist keeps other hosts out", async () => {
  const hosts = ["res.cloudinary.com"];
  await assert.rejects(
    () => resolveTarget("https://elsewhere.test/a.jpg", { hosts, resolver: publicResolver }),
    /not in ALLOWED_IMAGE_HOSTS/,
  );
  const ok = await resolveTarget("https://res.cloudinary.com/a.jpg", { hosts, resolver: publicResolver });
  assert.equal(ok.url.hostname, "res.cloudinary.com");
});

test("the allowlist is parsed from the environment, and blank means unrestricted", () => {
  assert.deepEqual(allowedHosts("a.com, B.COM ,"), ["a.com", "b.com"]);
  assert.equal(allowedHosts(""), null);
  assert.equal(allowedHosts(undefined), null);
});
