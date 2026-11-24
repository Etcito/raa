import areaStochastic from '../scenes/area-stochastic.js';
import { CORNELL_LIGHT_INTENSITY, CORNELL_SELECTION } from '../scenes/lighting.js';

export const BASELINE_FIXTURE = 'cornell-stratified-v1';
export const BASELINE_FEATURES = Object.freeze({ denoise: false, upscale: false, predict: false });
export const BASELINE_FRAME_MS = 1000 / 60;
export const BASELINE_WARMUP_FRAMES = 240;
export const BASELINE_FRAMES = 96;

export const BASELINE_EVENTS = Object.freeze([
  Object.freeze({ frame: 16, eventId: 'blocker-step' }),
  Object.freeze({ frame: 32, eventId: 'flash-on' }),
  Object.freeze({ frame: 48, eventId: 'flash-off' }),
  Object.freeze({ frame: 64, eventId: 'camera-edge' }),
  Object.freeze({ frame: 80, eventId: 'camera-return' }),
]);

const stochasticStill = areaStochastic.variants.find(row => row.id === 'stochastic-2x2-still');
const variant = {
  ...stochasticStill,
  movingBlocker: true,
  areaShadows: { ...stochasticStill.areaShadows, stochasticMode: 'stratified' },
};
const camera = CORNELL_SELECTION.camera;
const edgeCamera = { ...camera, target: [camera.target[0] + 2.7, camera.target[1], camera.target[2]] };

export async function buildNeuralBaseline(engine) {
  const state = await areaStochastic.build(engine, null, variant);
  engine.setWeather({ seed: 7, changing: false });
  engine.setTimeOfDay({ playing: false });
  engine.setToneMapping({ dither: false });
  engine.setAutoExposure({ enabled: false });
  engine.setPassEnabled('taa', false);
  engine.setRenderScale(1);
  return state;
}

export function assertNeuralBaselineReady(engine) {
  const gi = engine.giStats();
  const area = engine.areaLightShadows();
  const lights = engine.lightStats();
  if (gi.active !== true || gi.pendingRebuild !== false || area.tier !== 'auto' ||
      area.activeHeroes !== 0 || area.activeStochastic !== 1 || area.stochasticCells !== 4 ||
      area.stochasticMode !== 'stratified' ||
      lights.activeLights !== 1 || !engine.passes().some(pass => pass.name === 'gi.area-shadow' && pass.enabled)) {
    throw new Error('neural baseline: expected one active 2×2 stratified area light and a ready GI volume');
  }
  return area;
}

export function applyBaselineEvent(engine, state, eventId) {
  switch (eventId) {
    case 'blocker-step':
      engine.setTransform(state.blockNode, [0.85, 0, 0], [0, 0, 0, 1], [1, 1, 1]);
      break;
    case 'flash-on':
      engine.updateLight(state.areaLight, { intensity: CORNELL_LIGHT_INTENSITY * 2 });
      break;
    case 'flash-off':
      engine.updateLight(state.areaLight, { intensity: CORNELL_LIGHT_INTENSITY });
      break;
    case 'camera-edge':
      engine.setCamera(edgeCamera);
      break;
    case 'camera-return':
      engine.setCamera(camera);
      break;
    default:
      throw new Error(`neural baseline: unknown event ${eventId}`);
  }
}
