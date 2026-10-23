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
 * `args` = { width?, height?, onEngine?(engine) }: onEngine receives the new engine as
 * soon as it exists, so the caller owns it (screenshot, dispose) even when a later step
 * throws. Resolves to the S14 op result without gpuErrors (the page adds them).
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

  const timeMs = variantTimeMs(variant);
  for (let i = 0; i < WARMUP + FRAMES; i++) {
    renderAt(engine, scene, variant, timeMs);
    await yieldTask();
  }

  if (typeof scene.frame === 'function') scene.frame(engine, timeMs, variant);
  const nonEmpty = await engine.probeNonEmpty(timeMs);
  const stats = engine.stats();

  return {
    adapter: stats.adapter,
    hardware: stats.hardware,
    canvas: [width, height],
    nonEmpty: nonEmpty === true,
    firstFrameMs: stats.firstFrameMs,
    warmup: WARMUP,
    frames: FRAMES,
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
}
