// Canonical JSON text (S4) and the sha256 of it (S15 document hash). Web standards only.
import { SdfmError } from './errors.js';

/**
 * Canonical text of a JSON value: `JSON.stringify(value, null, 2)` plus LF.
 * Object keys keep their insertion order; numbers use JS Number-to-string and -0 is 0.
 * JS itself lists integer-like keys first, which asset keys never are.
 *
 * @param {unknown} value
 * @returns {string}
 * @throws SdfmError `sdfm.not_finite` (NaN or an infinity, with its JSON pointer),
 *   `sdfm.schema` (a value that is not JSON, with its JSON pointer)
 */
export function canonicalJson(value) {
  checkJson(value, '');
  return JSON.stringify(value, null, 2) + '\n';
}

/**
 * SHA-256 of a text as lowercase hex, over its UTF-8 bytes.
 *
 * @param {string} text
 * @returns {Promise<string>}
 */
export async function sha256Hex(text) {
  const bytes = new TextEncoder().encode(text);
  // crypto.subtle exists only in secure contexts; the GPU client's opaque-origin iframe and its
  // blob: worker are not one, so the same digest is computed in JS there.
  const subtle = globalThis.crypto && globalThis.crypto.subtle;
  const digest = subtle ? new Uint8Array(await subtle.digest('SHA-256', bytes)) : sha256Bytes(bytes);
  return Array.from(digest, (byte) => byte.toString(16).padStart(2, '0')).join('');
}

const K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

const rotr = (x, n) => (x >>> n) | (x << (32 - n));

/** SHA-256 (FIPS 180-4) of a byte array, used where crypto.subtle is unavailable. */
export function sha256Bytes(bytes) {
  const bitLength = bytes.length * 8;
  const padded = new Uint8Array(Math.ceil((bytes.length + 9) / 64) * 64);
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(padded.length - 8, Math.floor(bitLength / 0x100000000));
  view.setUint32(padded.length - 4, bitLength >>> 0);
  const h = new Uint32Array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ]);
  const w = new Uint32Array(64);
  for (let block = 0; block < padded.length; block += 64) {
    for (let i = 0; i < 16; i++) w[i] = view.getUint32(block + i * 4);
    for (let i = 16; i < 64; i++) {
      const s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3);
      const s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    let [a, b, c, d, e, f, g, hh] = h;
    for (let i = 0; i < 64; i++) {
      const t1 = (hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i]) >>> 0;
      const t2 = ((rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) >>> 0;
      hh = g;
      g = f;
      f = e;
      e = (d + t1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (t1 + t2) >>> 0;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  const out = new Uint8Array(32);
  const outView = new DataView(out.buffer);
  for (let i = 0; i < 8; i++) outView.setUint32(i * 4, h[i]);
  return out;
}

/**
 * SHA-256 of the canonical text of a document (the asset `hash` field).
 *
 * @param {unknown} document
 * @returns {Promise<string>}
 * @throws SdfmError as {@link canonicalJson}
 */
export async function documentHash(document) {
  return sha256Hex(canonicalJson(document));
}

function checkJson(value, pointer) {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return;
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new SdfmError('sdfm.not_finite', `non-finite number ${value}`, pointer);
    }
    return;
  }
  if (Array.isArray(value)) {
    for (let i = 0; i < value.length; i++) checkJson(value[i], `${pointer}/${i}`);
    return;
  }
  if (typeof value === 'object' && isPlainObject(value)) {
    for (const key of Object.keys(value)) {
      checkJson(value[key], `${pointer}/${escapePointerToken(key)}`);
    }
    return;
  }
  throw new SdfmError('sdfm.schema', 'not a JSON value', pointer);
}

function isPlainObject(value) {
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

function escapePointerToken(key) {
  return key.replaceAll('~', '~0').replaceAll('/', '~1');
}
