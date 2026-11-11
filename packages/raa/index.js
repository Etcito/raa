// @etcito/raa entry point. The engine itself is the wasm-bindgen output of the raa-js
// crate in ./pkg/ (built by scripts/build-package.sh). Types live in index.d.ts.
import init, { Engine } from './pkg/raa_js.js';

// Pure weather text parsing/formatting (no Engine, GPU or service); declared in types/m08-sky-ibl.d.ts.
export { parseWeatherText, formatWeatherText } from './sky-weather.js';

const VERSION = '0.5.0';

// Editor controls are native Engine methods: updateMesh, setWireframe,
// setSelectionOutline and setCamera (perspective or orthographic). Returning
// the native object preserves the synchronous, validated setter contract.
//
// Material textures are native Engine methods too: addTexture({ width, height, data })
// returns the id that material functions sample with raa_texture(id, uv), and
// removeTexture(id) frees it (docs/api/m22-textures.md).
//
// Trace scene methods are native Engine methods (docs/api/m23-sdf-trace.md).
//
// Material data words as well: addMaterialData(words) returns the base a surface material
// carries as `data`, read in WGSL by raa_material_word / raa_material_f32
// (docs/api/m25-material-data.md).

// The one instantiation of the wasm module. wasm-bindgen's init only skips instantiating once a
// previous call has FINISHED: two overlapping calls each instantiate the module and the second
// replaces the glue's exports, so engines made by the first then call into the other instance's
// memory and closure table ("memory access out of bounds", "table index is out of bounds",
// "closure invoked recursively or after being dropped": sdfm card 7126, runs 12681-12791).
let ready = null;

/**
 * Initialises the wasm module and returns a RaaModule.
 *
 * Every call, overlapping or not, shares the first call's instantiation; a failed one is
 * forgotten so a later call can try again. Each call returns a new, independent, frozen
 * RaaModule object.
 *
 * @param {{ wasmUrl?: string | URL }} [options]
 * @returns {Promise<import('./index.d.ts').RaaModule>}
 */
export async function loadRaa(options = {}) {
  ready ??= init({
    module_or_path: options.wasmUrl ?? new URL('./pkg/raa_js_bg.wasm', import.meta.url),
  }).catch((error) => {
    ready = null;
    throw error;
  });
  await ready;
  return Object.freeze({
    version: VERSION,
    createEngine: (canvas, opts) => Engine.create(canvas, opts ?? {}),
  });
}
