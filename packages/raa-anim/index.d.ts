// Public types of @etcito/raa-anim 0.5.0 (BRICK_VERSION). The wasm-bindgen declarations of
// the generated package are aggregated into types/all.d.ts by the brick build; this file
// adds the hand-written loading entry point and re-exports the brick API shapes.
/// <reference path="./types/all.d.ts" />

import type { RaaAnimApi } from './types/brick.js';
import type { RaaAnimClipsApi } from './types/clips.js';
import type { RaaAnimGraphApi } from './types/graph.js';
import type { RaaAnimIkApi } from './types/ik.js';

export type { BrickInfo, RaaAnimApi, SelftestCase, SelftestResult } from './types/brick.js';
export type { RaaAnimClipsApi } from './types/clips.js';
export type { AnimationGraph, AnimationGraphState, GraphStep, RaaAnimGraphApi } from './types/graph.js';
export type { RaaAnimIkApi } from './types/ik.js';

/** The initialised brick: metadata, selftest, clip loaders, the S19 graph loader and the visual IK namespace. */
export interface RaaAnimLoaded extends RaaAnimApi, Pick<RaaAnimClipsApi, 'loadRig' | 'loadClip' | 'sampleClip'>, Pick<RaaAnimGraphApi, 'loadGraph'> {
  readonly ik: RaaAnimIkApi;
}

/**
 * What {@link loadRaaAnim} accepts: a wasm file path or URL, or preloaded wasm bytes.
 *
 * In Node a string is a filesystem path to the wasm file or its package directory,
 * unless it starts with a `file:`, `http:` or `https:` scheme; `loadRaaAnim` reads it
 * with `fs`. In a browser a string or URL is fetched, and relative values resolve
 * against this module. The default is `./pkg/raa_anim_js_bg.wasm` next to index.js.
 */
export type RaaAnimSource = string | URL | ArrayBuffer | Uint8Array;

/**
 * Initialises the generated raa-anim wasm package and resolves to its brick API.
 *
 * Repeated calls reuse the instantiated wasm module and return a fresh frozen object.
 * Loading failures reject with the underlying filesystem, fetch or wasm error; the brick
 * methods themselves throw a `TypeError` with `api.invalid_input` or `api.unknown_case`
 * as described by {@link RaaAnimApi.selftest}.
 *
 * @param source wasm location or preloaded wasm bytes; default `pkg/raa_anim_js_bg.wasm`
 */
export function loadRaaAnim(source?: RaaAnimSource): Promise<RaaAnimLoaded>;
