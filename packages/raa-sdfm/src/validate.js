// Validation of the shared `sdfm.game-asset` v1 file (S15). Synchronous, no MAMA.
import { SdfmError } from './errors.js';
import { canonicalJson, documentHash } from './canonical.js';

/** Canonical key order of an sdfm.game-asset v1 file. */
export const ASSET_KEYS = Object.freeze([
  'format', 'version', 'game', 'id', 'name', 'category', 'world',
  'mesher', 'document', 'collision', 'anchors', 'bounds', 'hash',
]);

/** Largest canonical document text, in UTF-8 bytes. */
export const DOCUMENT_CAP_BYTES = 2 * 1024 * 1024;

const ID_RE = /^[a-z][a-z0-9]*(\.[a-z0-9-]+)+$/;
const HASH_RE = /^[0-9a-f]{64}$/;

/**
 * Validates an asset against S15 and collects every error.
 *
 * @param {unknown} asset parsed JSON
 * @returns {{ ok: boolean, errors: { code: string, path: string, message: string }[] }}
 */
export function validateAsset(asset) {
  const errors = [];
  const add = (code, path, message) => errors.push({ code, path, message });

  if (!isObject(asset)) {
    add('sdfm.schema', '', 'asset root must be an object');
    return { ok: false, errors };
  }
  if (asset.format !== 'sdfm.game-asset') {
    add('sdfm.format', '/format', `format must be 'sdfm.game-asset', got ${JSON.stringify(asset.format)}`);
  }
  if (asset.version !== 1) {
    add('sdfm.version', '/version', `version must be 1, got ${JSON.stringify(asset.version)}`);
  }

  const keys = Object.keys(asset);
  for (const key of ASSET_KEYS) {
    if (!(key in asset)) add('sdfm.schema', `/${key}`, `missing required key '${key}'`);
  }
  for (const key of keys) {
    if (!ASSET_KEYS.includes(key)) add('sdfm.schema', `/${escape(key)}`, `unknown key '${key}'`);
  }
  const known = keys.filter((k) => ASSET_KEYS.includes(k));
  const expected = ASSET_KEYS.filter((k) => known.includes(k));
  const at = known.findIndex((k, i) => k !== expected[i]);
  if (at >= 0) {
    add('sdfm.schema', `/${known[at]}`, `key '${known[at]}' out of canonical order (expected '${expected[at]}' here)`);
  }

  for (const key of ['game', 'id', 'name', 'category', 'world', 'hash']) {
    if (key in asset && typeof asset[key] !== 'string') add('sdfm.schema', `/${key}`, `'${key}' must be a string`);
  }
  for (const key of ['mesher', 'document', 'anchors', 'bounds']) {
    if (key in asset && !isObject(asset[key])) add('sdfm.schema', `/${key}`, `'${key}' must be an object`);
  }
  if ('collision' in asset && !Array.isArray(asset.collision)) {
    add('sdfm.schema', '/collision', "'collision' must be an array");
  }

  const nonFinite = [];
  findNonFinite(asset, '', nonFinite);
  for (const path of nonFinite) add('sdfm.not_finite', path, 'number must be finite');

  if (typeof asset.id === 'string' && !ID_RE.test(asset.id)) {
    add('sdfm.id', '/id', `id ${JSON.stringify(asset.id)} does not match ${ID_RE.source}`);
  }

  if (isObject(asset.document) && !nonFinite.some((p) => p.startsWith('/document'))) {
    try {
      const bytes = new TextEncoder().encode(canonicalJson(asset.document)).length;
      if (bytes > DOCUMENT_CAP_BYTES) {
        add('sdfm.cap_document', '/document', `canonical document is ${bytes} bytes, cap ${DOCUMENT_CAP_BYTES}`);
      }
    } catch (e) {
      if (!(e instanceof SdfmError)) throw e;
      add(e.code, `/document${e.path ?? ''}`, e.message);
    }
  }

  if (isObject(asset.anchors)) {
    for (const [name, a] of Object.entries(asset.anchors)) {
      const p = `/anchors/${escape(name)}`;
      if (!isObject(a)) { add('sdfm.schema', p, `anchor '${name}' must be an object {t, r}`); continue; }
      for (const k of Object.keys(a)) if (k !== 't' && k !== 'r') add('sdfm.schema', `${p}/${escape(k)}`, `unknown anchor key '${k}'`);
      if (!isNumbers(a.t, 3)) add('sdfm.schema', `${p}/t`, 'anchor t must be 3 numbers');
      if (!isNumbers(a.r, 4)) add('sdfm.schema', `${p}/r`, 'anchor r must be 4 numbers');
    }
  }
  if (isObject(asset.bounds)) {
    for (const k of Object.keys(asset.bounds)) if (k !== 'min' && k !== 'max') add('sdfm.schema', `/bounds/${escape(k)}`, `unknown bounds key '${k}'`);
    for (const k of ['min', 'max']) {
      if (!isNumbers(asset.bounds[k], 3)) add('sdfm.schema', `/bounds/${k}`, `bounds ${k} must be 3 numbers`);
    }
  }
  if (typeof asset.hash === 'string' && !HASH_RE.test(asset.hash)) {
    add('sdfm.schema', '/hash', 'hash must be 64 lowercase hex characters');
  }

  return { ok: errors.length === 0, errors };
}

/**
 * Recomputes the document hash and compares it with `asset.hash`.
 *
 * @param {{ document: unknown, hash: string }} asset a validated asset
 * @returns {Promise<void>}
 * @throws SdfmError `sdfm.hash_mismatch` at `/hash`
 */
export async function checkHash(asset) {
  const actual = await documentHash(asset.document);
  if (actual !== asset.hash) {
    throw new SdfmError('sdfm.hash_mismatch', `hash ${asset.hash} does not match the document (sha256 ${actual})`, '/hash');
  }
}

function isObject(v) {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function isNumbers(v, n) {
  return Array.isArray(v) && v.length === n && v.every((x) => typeof x === 'number');
}

function escape(key) {
  return key.replaceAll('~', '~0').replaceAll('/', '~1');
}

function findNonFinite(v, path, out) {
  if (typeof v === 'number') { if (!Number.isFinite(v)) out.push(path); return; }
  if (Array.isArray(v)) { v.forEach((x, i) => findNonFinite(x, `${path}/${i}`, out)); return; }
  if (isObject(v)) for (const k of Object.keys(v)) findNonFinite(v[k], `${path}/${escape(k)}`, out);
}
