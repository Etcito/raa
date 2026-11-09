// Runs one lookdev (scene, variant) and returns the S14 page op result.
//
// Every run creates a fresh engine (variant engine options merged over the scene's),
// builds the scene, applies the variant's pass settings, then renders LOOKDEV_CAPTURE:
// 30 warm-up frames and 1 frame, all at the variant's fixed time (default 1000 ms), so a
// capture never depends on wall-clock time. The page takes the screenshot after the run
// resolves (see renderAt, used by main.js for it).

export const WARMUP = 30;
export const FRAMES = 1;
export const DEFAULT_TIME_MS = 1000;
export const DEFAULT_SIZE = [1280, 720];

/** The fixed time of a variant in milliseconds. */
export function variantTimeMs(variant) {
  return variant && variant.timeMs !== undefined ? variant.timeMs : DEFAULT_TIME_MS;
}

/** Renders one frame at `timeMs`, letting the scene animate first when it can. */
export function renderAt(engine, scene, variant, timeMs) {
  if (typeof scene.frame === 'function') scene.frame(engine, timeMs, variant);
  engine.render(timeMs);
}

// Lets the browser run its own tasks (GPU callbacks, presentation) between frames.
// rAF fires rarely in the capture client, so a zero timeout steps the loop instead.
const yieldTask = () => new Promise((resolve) => setTimeout(resolve, 0));

/** Nearest-rank percentile ceil(p*n)-1 of an ascending array. */
export function percentile(sorted, p) {
  return sorted[Math.max(0, Math.ceil(p * sorted.length) - 1)];
}

/**
 * Turns measured animation-frame intervals (milliseconds) into the S20 timing keys.
 * Every interval must be a positive finite number: missing or nonpositive timestamps
 * reject the whole timing result, and no timer fallback ever stands in for them.
 */
export function summarizeFrameIntervals(intervals) {
  if (!Array.isArray(intervals) || intervals.length === 0) {
    throw new Error('runScene: measured mode recorded no frame intervals');
  }
  for (const interval of intervals) {
    if (typeof interval !== 'number' || !Number.isFinite(interval) || interval <= 0) {
      throw new Error('runScene: measured mode requires positive finite frame intervals');
    }
  }
  const ascending = [...intervals].sort((a, b) => a - b);
  const fps = intervals.map((interval) => 1000 / interval).sort((a, b) => a - b);
  const totalMs = intervals.reduce((sum, interval) => sum + interval, 0);
  return {
    frameIntervalMsMedian: percentile(ascending, 0.5),
    frameIntervalMsP95: percentile(ascending, 0.95),
    fpsMean: intervals.length / (totalMs / 1000),
    fpsMedian: percentile(fps, 0.5),
    fpsP95: percentile(fps, 0.95),
  };
}

function nextAnimationFrame() {
  return new Promise((resolve, reject) => {
    if (typeof globalThis.requestAnimationFrame !== 'function') {
      reject(new Error('runScene: measured mode requires requestAnimationFrame'));
      return;
    }
    globalThis.requestAnimationFrame((timestamp) => resolve(timestamp));
  });
}

// Boundary after warm-up, then exactly `frames` rendered frames whose consecutive
// animation-frame timestamps give the measured intervals.
export async function measureAnimationFrames(engine, scene, variant, timeMs, frames) {
  measureCount(frames, 'frames');
  let previous = await nextAnimationFrame();
  const intervals = [];
  for (let i = 0; i < frames; i++) {
    const timestamp = await nextAnimationFrame();
    if (typeof previous !== 'number' || !Number.isFinite(previous) ||
        typeof timestamp !== 'number' || !Number.isFinite(timestamp)) {
      throw new Error('runScene: measured mode requires positive finite frame intervals from finite timestamps');
    }
    renderAt(engine, scene, variant, timeMs);
    intervals.push(timestamp - previous);
    previous = timestamp;
  }
  return { ...summarizeFrameIntervals(intervals), frameIntervalsMs: intervals };
}

function measureCount(value, name) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`runScene: measure.${name} must be a positive integer`);
  }
  return value;
}

function setCanvasSize(canvas, width, height) {
  // devicePixelRatio 1: the backing store equals the CSS size.
  canvas.width = width;
  canvas.height = height;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
}

function applyPasses(engine, passes) {
  if (!passes) return;
  for (const name of passes.disable ?? []) engine.setPassEnabled(name, false);
  if (passes.order) engine.setPassOrder(passes.order);
}

/**
 * Runs `variant` of `scene` on `canvas` with the package module `raa` (from loadRaa).
 * `args` = { width?, height?, lib?, measure?, onEngine?(engine) }: onEngine receives
 * the new engine as soon as it exists, so the caller owns it (screenshot, dispose) even
 * when a later step throws. Resolves to the S14 op result without gpuErrors (the page
 * adds them).
 *
 * Omitting `args.measure` keeps the default capture: 30 warm-up frames, one capture
 * frame and exactly the existing result keys. Passing `args.measure =
 * { warmup: number, frames: number }` (both positive integers) runs the opt-in measured
 * mode: `warmup` frames, then a requestAnimationFrame boundary, then exactly `frames`
 * scheduled frames whose positive consecutive timestamps add `frameIntervalMsMedian`,
 * `frameIntervalMsP95`, `fpsMean`, `fpsMedian` and `fpsP95`. Missing or nonpositive
 * intervals reject; no timer fallback contributes to FPS. `cpuMsMedian` and `cpuMsP95`
 * keep their CPU submission meaning in both modes.
 *
 * Handoff: `runScene(raa, canvas, scene, variant, args = {})` in
 * `examples/lookdev/lib/runner.js`. The many-lights operation (examples/lights/main.js,
 * card "Ship the 64-light preset with integrated on/off functional results") attaches
 * this runner by importing it and calling it once per variant with
 * `args.measure = { warmup: 60, frames: 300 }`, then returns both results keyed by
 * `scene`/`variant` and carrying `canvas`, `adapter`, `hardware`, `nonEmpty`,
 * `gpuErrors` plus the measured timing keys above.
 */
export async function runScene(raa, canvas, scene, variant, args = {}) {
  const width = args.width ?? DEFAULT_SIZE[0];
  const height = args.height ?? DEFAULT_SIZE[1];
  setCanvasSize(canvas, width, height);

  const options = { ...(scene.engine ?? {}), ...(variant.engine ?? {}) };
  const engine = await raa.createEngine(canvas, options);
  if (typeof args.onEngine === 'function') args.onEngine(engine);
  engine.resize(width, height);

  await scene.build(engine, args.lib, variant);
  applyPasses(engine, variant.passes);

  const measure = args.measure;
  const warmup = measure ? measureCount(measure.warmup, 'warmup') : WARMUP;
  const frames = measure ? measureCount(measure.frames, 'frames') : FRAMES;

  const timeMs = variantTimeMs(variant);
  for (let i = 0; i < warmup; i++) {
    renderAt(engine, scene, variant, timeMs);
    await yieldTask();
  }

  let timing = null;
  if (measure) {
    timing = await measureAnimationFrames(engine, scene, variant, timeMs, frames);
  } else {
    for (let i = 0; i < frames; i++) {
      renderAt(engine, scene, variant, timeMs);
      await yieldTask();
    }
  }

  if (typeof scene.frame === 'function') scene.frame(engine, timeMs, variant);
  const nonEmpty = await engine.probeNonEmpty(timeMs);
  const stats = engine.stats();

  const result = {
    adapter: stats.adapter,
    hardware: stats.hardware,
    canvas: [width, height],
    nonEmpty: nonEmpty === true,
    firstFrameMs: stats.firstFrameMs,
    warmup,
    frames,
    cpuMsMedian: stats.cpuMsMedian,
    cpuMsP95: stats.cpuMsP95,
    instances: stats.instances,
    drawCalls: stats.drawCalls,
    triangles: stats.triangles,
    allocsPerFrameTotal: null,
    resize: null,
    scene: scene.id,
    variant: variant.id,
    passes: engine.passes(),
  };
  if (timing) Object.assign(result, timing);
  return result;
}
