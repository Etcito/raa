// @etcito/raa entry point. The engine itself is the wasm-bindgen output of the raa-js
// crate in ./pkg/ (built by scripts/build-package.sh). Types live in index.d.ts.
import init, { Engine } from './pkg/raa_js.js';

const VERSION = '0.4.0';

// Editor controls are native Engine methods: updateMesh, setWireframe,
// setSelectionOutline and setCamera (perspective or orthographic). Returning
// the native object preserves the synchronous, validated setter contract.

/**
 * Initialises the wasm module and returns a RaaModule.
 *
 * index.js keeps no module-level mutable state: the only state is inside the
 * wasm-bindgen glue, whose init is idempotent (later calls reuse the instantiated
 * module). Each call returns a new, independent, frozen RaaModule object.
 *
 * @param {{ wasmUrl?: string | URL }} [options]
 * @returns {Promise<import('./index.d.ts').RaaModule>}
 */
export async function loadRaa(options = {}) {
  await init({
    module_or_path: options.wasmUrl ?? new URL('./pkg/raa_js_bg.wasm', import.meta.url),
  });
  return Object.freeze({
    version: VERSION,
    createEngine: (canvas, opts) => Engine.create(canvas, opts ?? {}),
  });
}
