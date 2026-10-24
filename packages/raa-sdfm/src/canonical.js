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
  const digest = await globalThis.crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
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
