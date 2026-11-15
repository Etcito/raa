// Mid-tier area-light stochastic shadows (M6, contract 'gi-area-stochastic-rows') on the Cornell box of
// scenes/lighting.js. Four rows show the 2x2 and 3x3 stratified mid tier of the ceiling area light: two still
// rows, one with a swaying camera inside one clipmap snap cell and one with an SDF sphere and box as extra
// occluders. Each block must cast ONE smooth rectangular penumbra (never four or nine separate shadows), sharp
// at its base and widening with distance, with no crawling noise in the penumbra. Importing this module has no
// side effects.
import { CORNELL_HELD, CORNELL_SELECTION, DEFAULT_QUALITY, VOLUME_BYTES, cornellBox } from './lighting.js';
import { programBytes } from '../../trace-first-pixel/scene.js';

const ORBIT_TARGET = [2.78, 2.73, 0];
const ORBIT_DISTANCE = 8.0;
const YAW_STEP = 0.002;
const AREA_SHADOW_ORDER = 43;
const SDF_SUBJECTS = [
  { name: 'sdf-sphere', min: [3.65, 0, 0.65], max: [4.75, 1.1, 1.75] },
  { name: 'sdf-box', min: [0.75, 0, 3.85], max: [1.65, 1.0, 4.75] },
];
// The SDF occluders passed to programBytes of examples/trace-first-pixel/scene.js: a sphere at the front
// right and a box at the back left of the Cornell floor, each with its own flat colour.
const SDF_SCENE = {
  box: { center: [1.2, 0.5, 4.3], halfExtents: [0.45, 0.5, 0.45], color: [0.8, 0.8, 0.8] },
  sphere: { center: [4.2, 0.55, 1.2], radius: 0.55, color: [0.8, 0.8, 0.8] },
};
// mode, stochasticCells, motion, sdf, filename, criterionName
const ROWS = [
  { mode: 'stochastic-2x2-still', cells: 4, motion: false, sdf: false,
    filename: 'gi-area-stochastic-2x2-still-one-smooth-rectangular-penumbra.png', criterionName: 'M6 area light stochastic shadows: 2x2 cells, still camera' },
  { mode: 'stochastic-3x3-still', cells: 9, motion: false, sdf: false,
    filename: 'gi-area-stochastic-3x3-still-one-smooth-rectangular-penumbra.png', criterionName: 'M6 area light stochastic shadows: 3x3 cells, still camera' },
  { mode: 'stochastic-2x2-moving', cells: 4, motion: true, sdf: false,
    filename: 'gi-area-stochastic-2x2-moving-camera-no-crawling-noise-in-penumbra.png', criterionName: 'M6 area light stochastic shadows: 2x2 cells, moving camera' },
  { mode: 'stochastic-sdf-occluder', cells: 4, motion: false, sdf: true,
    filename: 'gi-area-stochastic-sdf-sphere-and-box-cast-soft-penumbrae-on-floor.png', criterionName: 'M6 area light stochastic shadows: SDF occluders' },
];
const DESCRIPTIONS = {
  'stochastic-2x2-still': 'The Cornell box with the mid-tier 2x2 stratified area-light shadows and a still camera.',
  'stochastic-3x3-still': 'The Cornell box with the mid-tier 3x3 stratified area-light shadows and a still camera.',
  'stochastic-2x2-moving': 'The Cornell box with the 2x2 mid tier while the camera sways slightly.',
  'stochastic-sdf-occluder': 'The Cornell box with an SDF sphere and box in front of the ceiling area light.',
};
const STILL_BEHAVIOUR = 'The two white blocks cast soft shadows of the ceiling area light on the floor and on the red wall. Each block casts ONE continuous shadow whose penumbra is a single smooth gradient, sharp where the shadow meets the block base and wider and softer with distance from the block, like the shadows in the reference; never four (or nine) separate overlapping copies of the shadow, no banding or stepped edges. The penumbrae are clean: no grain, speckles or blotchy noise. Bounce light lights the ceiling and the red and green walls tint the nearby white surfaces. Brightness differences from the path-traced reference are not faults.';
const BEHAVIOUR = {
  'stochastic-2x2-still': STILL_BEHAVIOUR,
  'stochastic-3x3-still': STILL_BEHAVIOUR,
  'stochastic-2x2-moving': 'The camera is swaying slightly. The shadows of the two blocks are still single smooth penumbrae, sharp at the block bases and widening with distance; there is no grain, no speckle, no shimmering blotches and no smeared, doubled or trailing ghost copy of any shadow edge; walls and blocks stay sharp. Brightness differences from the path-traced reference are not faults.',
  'stochastic-sdf-occluder': 'Besides the two white blocks, a sphere stands on the floor at the front right and a box at the back left. The sphere and the box each cast a soft shadow of the ceiling area light onto the floor: one smooth penumbra per object, darkest at the object base and widening away from it, with no grain or noise, like the block shadows. The colour or brightness of the sphere and box themselves is not judged. Brightness differences from the path-traced reference are not faults.',
};
const BASE_FACTS = ['snapshot:settings', 'snapshot:stats', 'runtime:ready', 'assertion:gi-active', 'assertion:area-stochastic', 'assertion:area-shadow-pass'];
const SDF_FACTS = [...BASE_FACTS, 'assertion:sdf-content'];
export const factNames = ['gi-active', 'area-stochastic', 'area-shadow-pass', 'sdf-content'];

const selectionOf = row => row.sdf
  ? { camera: CORNELL_SELECTION.camera, subjects: [...CORNELL_SELECTION.subjects, ...SDF_SUBJECTS] }
  : CORNELL_SELECTION;
const heldOf = row => {
  const held = [...CORNELL_HELD, `area-light shadow tier auto with stochasticCount 1 and stochasticCells ${row.cells}`];
  if (row.motion) held.push('camera swaying by at most 0.02 rad about the box centre');
  if (row.sdf) held.push('an SDF sphere at the front right and an SDF box at the back left, drawn by the SDF trace pass with their own shading');
  return held;
};
const variants = ROWS.map(row => ({
  id: row.mode,
  description: DESCRIPTIONS[row.mode],
  references: [],
  judge: true,
  arrangement: 'cornell-box',
  gi: { enabled: true, debugView: 'none' },
  quality: { ...DEFAULT_QUALITY },
  giExpected: 'active',
  selection: selectionOf(row),
  areaShadows: {
    tier: 'auto', heroCount: 0, heroDistance: 25, fadeDistance: 80, band: 5,
    stochasticCount: 1, stochasticDistance: 40, stochasticCells: row.cells,
  },
  motion: row.motion,
  sdf: row.sdf,
}));

export const inventory = ROWS.map(row => ({
  id: `area-stochastic-${row.mode}`,
  criterionName: row.criterionName,
  scenario: 'area-stochastic',
  mode: row.mode,
  filename: row.filename,
  expectedVisibleBehavior: BEHAVIOUR[row.mode],
  requiredFacts: row.sdf ? SDF_FACTS : BASE_FACTS,
  pair: null,
  heldConstant: heldOf(row),
  producer: 'raa-gi-area-stochastic',
  judgeReference: '/refs/cornell-box-reference.png',
  judgeAspect: 'functional',
}));

// The Cornell box, its quality, the mid-tier settings, the camera; then the orbit camera of the moving row,
// the SDF trace scene of the occluder row and the variant's global illumination.
export async function build(engine, lib, variant) {
  const state = { instances: [], moved: false, blockNode: null, frame: 0 };
  engine.setGiQuality(variant.quality);
  cornellBox(engine, variant, state);
  engine.setAreaLightShadows(variant.areaShadows);
  engine.setCamera(variant.selection.camera);
  if (variant.motion) engine.setOrbit(ORBIT_TARGET, ORBIT_DISTANCE, Math.PI, 0);
  if (variant.sdf) engine.setTraceScene(programBytes(SDF_SCENE));
  engine.setGlobalIllumination(variant.gi);
  return state;
}

// The yaw is one YAW_STEP per frame-hook call, reversed every 20 calls, so it stays within PI +- 0.02
// rad, inside one clipmap snap cell.
const yawSign = frame => (Math.floor((frame + 10) / 20) % 2 === 0 ? 1 : -1);

// Called by the page runner before every render: only the moving row sways the camera.
export function frame(engine, timeMs, variant, state) {
  if (!variant.motion) return;
  engine.orbit(yawSign(state.frame) * YAW_STEP, 0, 1);
  state.frame += 1;
}

const check = (name, passed, detail) => ({ name, passed: passed === true, detail });

// `observe` reports the four assertions of factNames, in order: gi-active with the rules of lighting.js (the
// volume at 96/5), area-stochastic from areaLightShadows() against the variant, area-shadow-pass from the
// gi.area-shadow pass row and sdf-content from traceStats().
export async function observe(engine, variant, state) {
  const stats = engine.giStats();
  const area = engine.areaLightShadows();
  const wanted = variant.areaShadows;
  const rows = engine.passes().filter(pass => pass.name === 'gi.area-shadow');
  const row = rows[0];
  const trace = engine.traceStats();
  const bytes = VOLUME_BYTES['96/5'];
  const assertions = [
    check('gi-active',
      stats.active === true && stats.pendingRebuild === false && stats.framesSinceRebuild >= 240
        && stats.allocatedBytes >= bytes && stats.voxelizedForwardInstances >= 1,
      `active ${stats.active}, pendingRebuild ${stats.pendingRebuild}, framesSinceRebuild ${stats.framesSinceRebuild}, `
        + `allocatedBytes ${stats.allocatedBytes} (expected at least ${bytes}), voxelizedForwardInstances ${stats.voxelizedForwardInstances}`),
    check('area-stochastic',
      area.tier === 'auto' && area.heroCount === 0 && area.stochasticCount === 1
        && area.stochasticCells === wanted.stochasticCells && area.activeStochastic === 1,
      `tier ${area.tier} (expected auto), heroCount ${area.heroCount} (expected 0), stochasticCount ${area.stochasticCount} (expected 1), `
        + `stochasticCells ${area.stochasticCells} (expected ${wanted.stochasticCells}), activeStochastic ${area.activeStochastic} (expected 1)`),
    check('area-shadow-pass',
      rows.length === 1 && row.enabled === true && row.order === AREA_SHADOW_ORDER && row.role === 'pass',
      `gi.area-shadow rows ${rows.length} (expected 1), enabled ${row?.enabled} (expected true), `
        + `order ${row?.order} (expected ${AREA_SHADOW_ORDER}), role ${row?.role} (expected pass)`),
    check('sdf-content', trace.instances === (variant.sdf ? 2 : 0),
      `trace instances ${trace.instances} (expected ${variant.sdf ? 2 : 0})`),
  ];
  return { instances: state.instances, assertions, edits: [] };
}

// Nothing to wait for in this stage: the GI settle gate of the runtime waits for giStats() on GI rows.
function ready() {
  return { ready: true, stage: 'lighting' };
}

export default {
  id: 'area-stochastic',
  milestone: 'lighting',
  title: 'Area light stochastic shadows (mid tier)',
  variants,
  build,
  ready,
  frame,
};
