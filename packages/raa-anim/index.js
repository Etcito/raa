// @etcito/raa-anim entry point. The brick itself is the wasm-bindgen output of the
// raa-anim-js crate in ./pkg/. The public API is brickInfo(), selftest(), the clip
// loaders and the visual ik namespace; types live in index.d.ts and types/*.d.ts.
import init, { brickInfo, ik, loadClip, loadRig, sampleClip, selftest } from './pkg/raa_anim_js.js';

const WASM_FILE = 'raa_anim_js_bg.wasm';
const DEFAULT_WASM_URL = new URL(`./pkg/${WASM_FILE}`, import.meta.url);

function isNodeRuntime() {
  const proc = globalThis.process;
  return (
    typeof globalThis.window === 'undefined' &&
    typeof proc !== 'undefined' &&
    Object.prototype.toString.call(proc) === '[object process]' &&
    proc.versions != null &&
    proc.versions.node != null
  );
}

async function fetchBytes(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`@etcito/raa-anim: cannot load ${url} (HTTP ${response.status})`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

function isBytes(value) {
  return value instanceof Uint8Array || value instanceof ArrayBuffer;
}

// Node: a string is a filesystem path (absolute, or relative to process.cwd()) unless
// it names a file:, http: or https: URL. The wasm is read with fs, never fetched from
// a file URL.
async function loadNode(source) {
  const { readFile, stat } = await import('node:fs/promises');
  const { fileURLToPath, pathToFileURL } = await import('node:url');
  const { isAbsolute, resolve } = await import('node:path');

  let url;
  if (source === undefined) {
    url = DEFAULT_WASM_URL;
  } else if (source instanceof URL) {
    url = source;
  } else if (/^[A-Za-z]:[\\/]/.test(source)) {
    url = pathToFileURL(source);
  } else if (/^[A-Za-z][A-Za-z0-9+.-]*:/.test(source)) {
    url = new URL(source);
  } else {
    url = pathToFileURL(isAbsolute(source) ? source : resolve(source));
  }

  if (url.protocol === 'file:') {
    let file = fileURLToPath(url);
    if ((await stat(file)).isDirectory()) {
      const direct = resolve(file, WASM_FILE);
      const nested = resolve(file, 'pkg', WASM_FILE);
      file = (await stat(direct).catch(() => null)) ? direct : nested;
    }
    return readFile(file);
  }
  if (url.protocol === 'http:' || url.protocol === 'https:') return fetchBytes(url);
  throw new TypeError(`@etcito/raa-anim: unsupported Node source URL ${url.href}`);
}

// Browser: the wasm is fetched. A relative source is resolved against this module,
// like the default ./pkg/raa_anim_js_bg.wasm next to index.js.
function loadBrowser(source) {
  if (source === undefined) return fetchBytes(DEFAULT_WASM_URL);
  if (source instanceof URL) return fetchBytes(source);
  return fetchBytes(new URL(source, import.meta.url));
}

async function resolveModule(source) {
  if (isBytes(source)) return source;
  if (source !== undefined && typeof source !== 'string' && !(source instanceof URL)) {
    throw new TypeError(
      '@etcito/raa-anim: source must be a URL, a path string, an ArrayBuffer or a Uint8Array',
    );
  }
  return isNodeRuntime() ? loadNode(source) : loadBrowser(source);
}

/**
 * Initialises the generated raa-anim wasm package and returns its brick API.
 *
 * In Node the wasm is read with fs; in a browser it is fetched. On repeated calls the
 * wasm-bindgen glue reuses the instantiated module, so the returned object is a fresh,
 * frozen wrapper around the same brick functions.
 *
 * @param {string | URL | ArrayBuffer | Uint8Array} [source] wasm location or preloaded bytes
 * @returns {Promise<{ brickInfo: () => { name: 'raa-anim', version: '0.5.0' },
 *   selftest: (testCase: string, seed: number, ticks: number) =>
 *     { case: string, seed: number, ticks: number, hash: string },
 *   loadRig: (json: string) => object, loadClip: (rig: object, json: string) => object,
 *   sampleClip: (clip: object, tick: number) => object, ik: object }>}
 */
export async function loadRaaAnim(source) {
  const moduleOrPath = await resolveModule(source);
  await init({ module_or_path: moduleOrPath });
  return Object.freeze({
    brickInfo: () => brickInfo(),
    selftest: (testCase, seed, ticks) => selftest(testCase, seed, ticks),
    loadRig,
    loadClip,
    sampleClip,
    ik,
  });
}
