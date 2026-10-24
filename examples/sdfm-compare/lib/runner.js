// Runs one side-by-side comparison (S14 page op): the same sdfm.game-asset fixture is shown
// by the MAMA reference viewport on the left canvas (MAMA's own meshes and base colours, plain
// WebGPU: MAMA 864ebc3 has no viewport host) and by RAA on the right, with the same camera and size.
//
// Every run creates a fresh RAA engine, loads and spawns the fixture through
// @etcito/raa-sdfm, sets the camera from the variant settings (`settings.camera`) on both
// sides, then renders LOOKDEV_CAPTURE: 30 warm-up frames and 1 frame, all at the variant's
// fixed time. The page takes the screenshot after the run resolves (see renderCapture).
//
// Scene variant contract: `variant.fixture` (or `settings.fixture` / `scene.fixture`, a path
// relative to the page, e.g. 'fixtures/crate.asset.json') and `settings.camera` ({ position |
// eye, target, fovYDeg } or a name in `scene.cameras`); the asset's bounds frame the default.

import { createSdfmLoader } from '../raa-sdfm/index.js';
import { createMamaView } from './mama-view.js';

export const WARMUP = 30;
export const FRAMES = 1;
export const DEFAULT_TIME_MS = 1000;
export const DEFAULT_SIZE = [1280, 720];
export const DEFAULT_FOV_Y_DEG = 45;

// Standard lit setup of the lookdev scenes (m08-sky-ibl): a warm daytime sun, a dim flat
// ambient so shadow sides stay readable and the procedural day sky with its image-based
// lighting. The asset keeps its own materials and colours.
const SUN = { direction: [-0.2955, -0.2924, 0.9096], color: [1, 0.93, 0.82], intensity: 5.5 };
const AMBIENT = { color: [0.55, 0.6, 0.7], intensity: 0.35 };
const SKY = { preset: 'day' };
const BACKGROUND = [0.3, 0.34, 0.4];

// The page root: lib/runner.js lives one level below dist/sdfm-compare/.
const PAGE_ROOT = new URL('../', import.meta.url);
// MAMA's staged browser bundle, same paths as MAMA's dist/ (README, "Browser build").
const MAMA_ENTRY = 'mama/index.js';
// The worker of the staged raa-sdfm package (build-sdfm-compare.sh copies it next to lib/).
const WORKER_URL = new URL('raa-sdfm/worker.js', PAGE_ROOT);

/** The fixed time of a variant in milliseconds (same rule as S14). */
export function variantTimeMs(variant) {
  return variant && variant.timeMs !== undefined ? variant.timeMs : DEFAULT_TIME_MS;
}

const finite = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : undefined);

function vec3(v) {
  if (!v || typeof v.length !== 'number' || v.length !== 3) return null;
  const out = [v[0], v[1], v[2]];
  return out.every((x) => Number.isFinite(x)) ? out : null;
}

function setCanvasSize(canvas, width, height) {
  // devicePixelRatio 1: the backing store equals the CSS size.
  canvas.width = width;
  canvas.height = height;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
}

function applyLitSetup(engine) {
  engine.setBackground(BACKGROUND);
  engine.setLight(SUN);
  engine.setAmbient(AMBIENT);
  engine.setSky(SKY);
}

// A camera that frames `bounds` from the front right, used when the variant names none.
function framingCamera(bounds) {
  const min = vec3(bounds && bounds.min) ?? [-1, -1, -1];
  const max = vec3(bounds && bounds.max) ?? [1, 1, 1];
  const target = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
  const radius = Math.max(1e-3, 0.5 * Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]));
  const fov = (DEFAULT_FOV_Y_DEG * Math.PI) / 180;
  const distance = (radius / Math.sin(fov / 2)) * 1.15;
  const dir = [0.55, 0.35, 1];
  const len = Math.hypot(...dir);
  return {
    eye: [target[0] + (distance * dir[0]) / len, target[1] + (distance * dir[1]) / len, target[2] + (distance * dir[2]) / len],
    target,
  };
}

/**
 * The camera of a variant: `settings.camera` (or a name in `scene.cameras`), with the frame
 * of the asset as the fallback. Both the eye and the target are shared by MAMA and RAA.
 */
function cameraFromSettings(settings, scene, asset) {
  let desc = settings && settings.camera;
  if (typeof desc === 'string') desc = scene && scene.cameras ? scene.cameras[desc] : undefined;
  if (!desc || typeof desc !== 'object') desc = {};
  const fallback = framingCamera(asset && asset.bounds);
  const eye = vec3(desc.position) ?? vec3(desc.eye) ?? fallback.eye;
  const target = vec3(desc.target) ?? fallback.target;
  return {
    eye,
    target,
    up: vec3(desc.up) ?? [0, 1, 0],
    fovYDeg: finite(desc.fovYDeg) ?? DEFAULT_FOV_Y_DEG,
    near: finite(desc.near) ?? 0.01,
    far: finite(desc.far) ?? 200,
  };
}

async function fetchFixture(scene, variant, settings) {
  const fixture = variant.fixture ?? settings.fixture ?? scene.fixture;
  if (typeof fixture !== 'string' || fixture === '') {
    throw new Error(`variant ${variant.id}: no fixture (variant.fixture, settings.fixture or scene.fixture)`);
  }
  const response = await fetch(new URL(fixture, PAGE_ROOT));
  if (!response.ok) throw new Error(`fixture ${fixture}: HTTP ${response.status}`);
  return response.json();
}

const yieldTask = () => new Promise((resolve) => setTimeout(resolve, 0));

async function renderFrame(state) {
  if (state.mamaView) await state.mamaView.render();
  state.engine.render(state.timeMs);
}

/** Renders the capture frame again, at the variant's fixed time (screenshot path). */
export async function renderCapture(state) {
  await renderFrame(state);
}

function disposeView(view) {
  if (view && typeof view.dispose === 'function') {
    try {
      view.dispose();
    } catch {
      // a refused view may have nothing to release
    }
  }
}

/**
 * Runs `variant` of `scene` with the module `raa` and the page canvases
 * `{ mama, raa }`. Resolves to the S14 op result without gpuErrors (the page adds them).
 * `args` = { width?, height?, onState?(state) }: onState receives the state of a completed
 * run (engine, MAMA view, fixed time, canvas size) so the caller can render the capture
 * frame and dispose it.
 */
export async function runCompare(raa, canvases, scene, variant, args = {}) {
  const width = args.width ?? DEFAULT_SIZE[0];
  const height = args.height ?? DEFAULT_SIZE[1];
  const paneWidth = Math.max(1, Math.round(width / 2));
  setCanvasSize(canvases.mama, paneWidth, height);
  setCanvasSize(canvases.raa, paneWidth, height);

  const settings = variant.settings ?? {};
  const timeMs = variantTimeMs(variant);
  const options = { ...(scene.engine ?? {}), ...(variant.engine ?? {}) };
  let engine = null;
  let loader = null;
  let mamaView = null;
  try {
    engine = await raa.createEngine(canvases.raa, options);
    engine.resize(paneWidth, height);
    applyLitSetup(engine);

    const mamaUrl = new URL(MAMA_ENTRY, PAGE_ROOT).href;
    loader = createSdfmLoader({
      engine,
      mama: async () => mamaUrl,
      workerUrl: WORKER_URL,
    });

    const fixture = await fetchFixture(scene, variant, settings);
    const asset = await loader.load(fixture);
    loader.spawn(asset);
    loader.dispose();
    loader = null;

    mamaView = await createMamaView(canvases.mama, { mamaUrl, background: BACKGROUND });
    await mamaView.load(fixture);

    const camera = cameraFromSettings(settings, scene, asset);
    engine.setCamera(camera);
    mamaView.setCamera({
      position: camera.eye,
      target: camera.target,
      up: camera.up,
      fovYDeg: camera.fovYDeg,
      near: camera.near,
      far: camera.far,
    });

    const state = { engine, mamaView, timeMs, width, height, scene: scene.id, variant: variant.id };
    for (let i = 0; i < WARMUP + FRAMES; i++) {
      await renderFrame(state);
      await yieldTask();
    }

    const raaProbe = (await engine.probeNonEmpty(timeMs)) === true;
    const mamaProbe = (await mamaView.probeNonEmpty()) === true;
    const stats = engine.stats();
    const result = {
      adapter: stats.adapter,
      hardware: stats.hardware,
      canvas: [width, height],
      nonEmpty: raaProbe && mamaProbe,
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
    if (typeof args.onState === 'function') args.onState(state);
    return result;
  } catch (error) {
    disposeView(mamaView);
    if (loader) loader.dispose();
    if (engine) {
      try {
        engine.dispose();
      } catch {
        // already disposed by the engine itself
      }
    }
    throw error;
  }
}
