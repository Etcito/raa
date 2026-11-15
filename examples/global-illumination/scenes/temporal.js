// Temporal anti-aliasing over a swaying camera (M5, contract 'gi-taa-rows') on the Cornell box of scenes/lighting.js.
// Two rows show the same box at msaa 1 while the camera sways about its orbit target: TAA off (stair-stepped edges)
// and TAA on (smooth edges, no ghost trail behind the orange sphere sliding along the floor). The sphere is excluded
// from GI and the sway stays inside one clipmap snap cell, so the GI volume never rebuilds during the sway.
// Importing this module has no side effects.
import { CORNELL_HELD, DEFAULT_QUALITY, VOLUME_BYTES, cornellBox } from './lighting.js';
import { sphere } from '../lib/content.js';

const ORBIT_TARGET = [2.78, 2.73, 0];
const ORBIT_DISTANCE = 7.9;
const YAW_STEP = 0.002;
const TAA_ORDER = 100;
const SPHERE_RADIUS = 0.25;
const SPHERE_Z = 0.35;
const SHADOW = { heroDistance: 25, fadeDistance: 80, band: 5 };
const SELECTION = {
  camera: { eye: [2.78, 2.73, -7.9], target: [2.78, 2.73, 0], fovYDeg: 39.3, near: 0.05, far: 200 },
  subjects: [
    { name: 'cornell-room', min: [0, 0, 0], max: [5.56, 5.488, 5.592] },
    { name: 'light', min: [2.13, 5.4, 2.27], max: [3.43, 5.49, 3.32] },
    { name: 'sliding-sphere', min: [0.75, 0, 0.1], max: [4.75, 0.5, 0.6] },
  ],
};
// mode, variant.taa, filename, criterionName
const ROWS = [
  { mode: 'taa-off', taa: false, filename: 'gi-taa-off-swaying-camera-jagged-stair-step-edges.png', criterionName: 'M5 temporal accumulation: TAA off' },
  { mode: 'taa-on', taa: true, filename: 'gi-taa-on-swaying-camera-smooth-edges-no-ghost-trail.png', criterionName: 'M5 temporal accumulation: TAA on' },
];
const DESCRIPTIONS = {
  'taa-off': 'The Cornell box under a swaying camera with temporal anti-aliasing off.',
  'taa-on': 'The Cornell box under a swaying camera with temporal anti-aliasing on.',
};
const BEHAVIOUR = {
  'taa-off': 'Temporal anti-aliasing is off and the frame has no multisampling: the outlines of the two white blocks, the edges of the ceiling light fixture, the wall corners and the outline of the orange sphere are jagged, with visible stair-stepped pixel edges along every slanted line.',
  'taa-on': 'Temporal anti-aliasing is on: the outlines of the two white blocks, the ceiling light fixture, the wall corners and the orange sphere are smooth anti-aliased lines with no visible stair steps; the orange sphere is crisp with no smeared trail, doubled copy or dark ghost outline beside it; walls and blocks stay sharp, not blurred.',
};
const FACTS = ['snapshot:settings', 'snapshot:stats', 'runtime:ready', 'assertion:gi-active', 'assertion:taa-pass'];
export const factNames = ['gi-active', 'taa-pass'];

const variants = ROWS.map(row => ({
  id: row.mode,
  description: DESCRIPTIONS[row.mode],
  references: [],
  judge: true,
  arrangement: 'cornell-box',
  engine: { msaa: 1 },
  gi: { enabled: true, debugView: 'none' },
  quality: { ...DEFAULT_QUALITY },
  giExpected: 'active',
  selection: SELECTION,
  areaShadows: { tier: 'hero', heroCount: 1, ...SHADOW },
  taa: row.taa,
}));

export const inventory = ROWS.map(row => ({
  id: `temporal-${row.mode}`,
  criterionName: row.criterionName,
  scenario: 'temporal',
  mode: row.mode,
  filename: row.filename,
  expectedVisibleBehavior: BEHAVIOUR[row.mode],
  requiredFacts: FACTS,
  pair: null,
  heldConstant: [...CORNELL_HELD, 'orange sphere path along the floor, excluded from GI'],
  producer: 'raa-taa',
}));

// The Cornell box, its quality, the hero shadow tier, the orbit camera, GI, the TAA pass and the sliding sphere.
export async function build(engine, lib, variant) {
  const state = { instances: [], moved: false, blockNode: null, frame: 0, sphereNode: null };
  engine.setGiQuality(variant.quality);
  cornellBox(engine, variant, state);
  engine.setAreaLightShadows(variant.areaShadows);
  engine.setCamera(variant.selection.camera);
  engine.setOrbit(ORBIT_TARGET, ORBIT_DISTANCE, Math.PI, 0);
  engine.setGlobalIllumination(variant.gi);
  engine.setPassEnabled('taa', variant.taa);
  state.sphereNode = engine.addNode(null, [1.0, SPHERE_RADIUS, SPHERE_Z], [0, 0, 0, 1], [1, 1, 1]);
  const material = engine.addMaterial({ color: [1, 0.45, 0.05], roughness: 0.5 });
  const desc = sphere(SPHERE_RADIUS, 32);
  const mesh = engine.addMesh({ ...desc, subMeshes: [{ first: 0, count: desc.indices.length, material }] });
  const instance = engine.addMeshInstance(state.sphereNode, mesh);
  engine.setInstanceGlobalIllumination(instance, false);
  state.instances.push({ name: 'sliding-sphere', instance });
  return state;
}

// The yaw is one YAW_STEP per frame-hook call, reversed every 20 calls, so it stays within PI +- 0.02.
const yawSign = frame => (Math.floor((frame + 10) / 20) % 2 === 0 ? 1 : -1);
// The sphere slides between x 1.0 and 4.5 and back, one period per 240 calls.
const sphereX = frame => 1.0 + 3.5 * (1 - Math.abs(((frame / 120) % 2) - 1));

// Called by the page runner before every render: one orbit step (not a camera cut) and the sphere's new place.
export function frame(engine, timeMs, variant, state) {
  engine.orbit(yawSign(state.frame) * YAW_STEP, 0, 1);
  const x = sphereX(state.frame);
  engine.setTransforms(new Uint32Array([state.sphereNode]), new Float32Array([x, SPHERE_RADIUS, SPHERE_Z, 0, 0, 0, 1, 1, 1, 1]));
  state.frame += 1;
}

const check = (name, passed, detail) => ({ name, passed: passed === true, detail });

// `observe` reports gi-active with the rules of lighting.js (the volume at 96/5) and taa-pass from the pass rows.
export async function observe(engine, variant, state) {
  const stats = engine.giStats();
  const rows = engine.passes().filter(pass => pass.name === 'taa');
  const row = rows[0];
  const bytes = VOLUME_BYTES['96/5'];
  const assertions = [
    check('gi-active',
      stats.active === true && stats.pendingRebuild === false && stats.framesSinceRebuild >= 240
        && stats.allocatedBytes >= bytes && stats.voxelizedForwardInstances >= 1,
      `active ${stats.active}, pendingRebuild ${stats.pendingRebuild}, framesSinceRebuild ${stats.framesSinceRebuild}, `
        + `allocatedBytes ${stats.allocatedBytes} (expected at least ${bytes}), voxelizedForwardInstances ${stats.voxelizedForwardInstances}`),
    check('taa-pass',
      rows.length === 1 && row.enabled === variant.taa && row.order === TAA_ORDER && row.role === 'pass',
      `taa rows ${rows.length} (expected 1), enabled ${row?.enabled} (expected ${variant.taa}), order ${row?.order} (expected ${TAA_ORDER})`),
  ];
  return { instances: state.instances, assertions, edits: [] };
}

// Nothing to wait for in this stage: the GI settle gate of the runtime waits for giStats() on GI rows.
function ready() {
  return { ready: true, stage: 'lighting' };
}

export default {
  id: 'temporal',
  milestone: 'lighting',
  title: 'Temporal anti-aliasing over a swaying camera',
  variants,
  build,
  ready,
  frame,
};
