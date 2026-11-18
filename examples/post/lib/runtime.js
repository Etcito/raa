// Page runtime of the post page (S49, module post-page). createPostPageRunner wraps createPageRunner of page-runner
// with the post facts, the inventory check of lib/inventory.js, the automatic exposure settle gate and the capture at
// 1280 x 720. Each scene's build is followed by its variant's settings, so M2 and M3 add scenes without editing this file.
// Consumed upstream, not re-implemented here: createPageRunner (web/common/page-runner.js:265; its Handoff doc comment
// is at :256), describeError (web/common/page-runner.js:113), browserClock (web/common/page-runner.js:110), renderAt
// (web/common/page-runner.js:125), variantTimeMs (web/common/page-runner.js:119) and captureCurrent
// (examples/sky-weather/lib/capture.js:54). validateRequiredFacts (examples/post/lib/inventory.js:177; its Handoff doc
// comment is at :172) and SETTING_ORDER come from ./inventory.js.
import { browserClock, createPageRunner, describeError, renderAt, variantTimeMs } from '../../../web/common/page-runner.js';
import { captureCurrent } from '../../sky-weather/lib/capture.js';
import { SETTING_ORDER, validateRequiredFacts } from './inventory.js';

// The frame limit of the automatic exposure settle (S49): the row fails when the exposure has not settled by then.
export const POST_SETTLE_MAX_FRAMES = 600;
// One frame of a 60 Hz clock; settle frame k renders at the variant's time plus (k + 1) of these.
export const SETTLE_FRAME_MS = 1000 / 60;
// Settled reads in a row that end the gate (S46: settled is decided over the two latest results).
const SETTLED_READS = 2;

// The engine method of each settings key; SETTING_ORDER gives the order of the calls.
const SETTERS = Object.freeze({
  toneMapping: 'setToneMapping',
  passes: 'setPassEnabled',
  renderScale: 'setRenderScale',
  autoExposure: 'setAutoExposure',
  sharpening: 'setSharpening',
});

const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const verdict = (status, state, frames, error) => ({ status, state, frames, error });

// Renders settle frames until autoExposureState() reports settled on SETTLED_READS reads in a row. It reports ready
// only then; a spent budget, the frame limit, or a thrown render or read fails it, with the state of the last read.
export async function waitExposureSettled(engine, render, budget, maxFrames = POST_SETTLE_MAX_FRAMES) {
  let frames = 0;
  let reads = 0;
  let state = null;
  for (;;) {
    if (budget.remainingMs() <= 0) return verdict('failed', state, frames, 'exposure: not settled before the operation deadline');
    if (frames >= maxFrames) return verdict('failed', state, frames, `exposure did not settle in ${maxFrames} frames`);
    try {
      await render(frames);
      frames += 1;
      state = engine.autoExposureState();
    } catch (error) {
      return verdict('failed', state, frames, describeError(error));
    }
    reads = state?.settled === true ? reads + 1 : 0;
    if (reads >= SETTLED_READS) return verdict('ready', state, frames, null);
  }
}

// The render of settle frame k (0-based): it waits for the clock's next frame, then renders at the variant's time plus
// (k + 1) SETTLE_FRAME_MS. The page runner's own render holds the variant's time, and the pipeline gives dt 0 for a
// repeated or earlier time (crates/raa-pipeline/src/pipeline.rs:445-455), so exposure would keep its first value. The
// advancing time lets it adapt; the measured and captured frames, back at the variant's time, keep the settled value.
export function advancingRender({ engine, scene, variant, state, clock }) {
  return k => new Promise((resolve, reject) => {
    clock.requestFrame(() => {
      try {
        renderAt(engine, scene, variant, variantTimeMs(variant) + (k + 1) * SETTLE_FRAME_MS, state);
        resolve();
      } catch (error) {
        reject(error);
      }
    });
  });
}

// Applies the settings keys present, in SETTING_ORDER: toneMapping, passes (setPassEnabled per entry), renderScale,
// autoExposure, sharpening. Every key and method is checked before the first call, so a refused key changes nothing;
// an engine refusal propagates.
export function applyVariantSettings(engine, settings = {}) {
  const keys = Object.keys(settings);
  for (const key of keys) {
    if (!SETTING_ORDER.includes(key)) throw new Error(`post runtime: unknown setting ${key}`);
  }
  const applied = SETTING_ORDER.filter(key => keys.includes(key));
  for (const key of applied) {
    if (typeof engine?.[SETTERS[key]] !== 'function') throw new Error(`post runtime: engine has no ${SETTERS[key]}`);
  }
  for (const key of applied) {
    if (key === 'passes') {
      for (const [name, enabled] of Object.entries(settings.passes)) engine.setPassEnabled(name, enabled);
    } else {
      engine[SETTERS[key]](settings[key]);
    }
  }
}

// The capture of the canvas as captureCurrent makes it (a base64 PNG). A canvas of another size, a row drawn below
// render scale 1, is first drawn smoothed into a 2D canvas of width x height, and that copy is captured.
export async function captureDisplayed(canvas, status, { width = 1280, height = 720, capture = captureCurrent } = {}) {
  if (canvas.width === width && canvas.height === height) return capture(canvas, status);
  const owner = canvas.ownerDocument ?? globalThis.document;
  if (!owner || typeof owner.createElement !== 'function') throw new Error('post runtime: a document is required for the capture');
  const proxy = owner.createElement('canvas');
  proxy.width = width;
  proxy.height = height;
  const context = proxy.getContext('2d');
  if (!context) throw new Error('post runtime: 2D canvas context is unavailable');
  context.imageSmoothingEnabled = true;
  context.drawImage(canvas, 0, 0, width, height);
  return capture(proxy, status);
}

// The settle phase of createPageRunner: a variant with settle draws until automatic exposure settles, on the advancing
// frames; any other variant is ready at once, with no frame rendered.
function postSettle(clock) {
  return (engine, { budget, scene, variant, state }) => {
    if (variant.settle !== true) return { status: 'ready', state: null, frames: 0, error: null };
    return waitExposureSettled(engine, advancingRender({ engine, scene, variant, state, clock }), budget);
  };
}

// Each scene's build is followed by applyVariantSettings(engine, variant.settings), and its state is returned. A
// namespace without a default scene object is passed on unchanged, so createPageRunner refuses it.
function wrapScenes(scenes) {
  if (!(scenes instanceof Map)) return scenes;
  return new Map([...scenes].map(([id, namespace]) => [id, wrapNamespace(namespace)]));
}

function wrapNamespace(namespace) {
  if (!record(namespace) || !record(namespace.default)) return namespace;
  const scene = namespace.default;
  const build = async (engine, lib, variant) => {
    const state = await scene.build(engine, lib, variant);
    applyVariantSettings(engine, variant.settings);
    return state;
  };
  return { ...namespace, default: { ...scene, build } };
}

// The clock when options.clock is absent: browserClock(), looked up on first use, so that a refused option (a missing
// canvas, say) is reported before any requestAnimationFrame lookup.
function lazyBrowserClock() {
  let clock = null;
  const current = () => (clock ??= browserClock());
  return {
    now: () => current().now(),
    requestFrame: callback => current().requestFrame(callback),
    cancelFrame: id => current().cancelFrame(id),
  };
}

// The facts of an observation; the page runner adds passes, viewSizes and readiness (page-runner.js:414-419).
const collectPostFacts = (engine, observation) => ({ values: observation.values, assertions: observation.assertions });

/**
 * Handoff: post-page: page entry, build script and p5 commands calls `createPostPageRunner(options: { canvas: object, status: object, load: () => Promise<object>, scenes: Map<string, object>, lib: { raa: object, content: object } }) -> { run(input: object): Promise<object>, engine(): object | null, dispose(): void }`
 * exported by examples/post/lib/runtime.js:154 createPostPageRunner; the page entry (main.js) attaches it as attachPostPage's default createRunner and registers runner.run with registerNmcGpu({ run }). State/keys: the scenes Map (scene id to namespace) and lib, passed on to createPageRunner with topic 'post'.
 * Today it returns createPageRunner's runner with topic 'post', collectPostFacts, validateRequiredFacts, the post settle phase and captureDisplayed as its capture; its option and run refusals carry the 'post runtime: ' prefix.
 */
export function createPostPageRunner(options) {
  const clock = options?.clock ?? lazyBrowserClock();
  return createPageRunner({
    ...options,
    topic: 'post',
    scenes: wrapScenes(options?.scenes),
    collectFacts: collectPostFacts,
    validateRequiredFacts,
    settle: postSettle(clock),
    capture: (canvas, status) => captureDisplayed(canvas, status),
    clock,
  });
}
