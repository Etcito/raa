// M7 ReSTIR DI ultra tier rows on the Cornell box of scenes/lighting.js (contract 'gi-area-restir-rows'): the
// ceiling area light's visibility comes from ReSTIR DI in the gi.area-shadow pass instead of stratified rays. Three
// rows: a still camera after convergence, a swaying camera and an SDF sphere and box as extra occluders. Each block
// must cast ONE smooth rectangular penumbra, sharp at its base and widening with distance, with no blotchy
// correlation patterns and no boiling. Colour bleed is the M9 job and is not judged here. Importing this module has
// no side effects.
import { CORNELL_HELD, DEFAULT_QUALITY } from './lighting.js';
import { areaAssertions, build, frame, MOVING_HELD, SDF_HELD, selectionOf } from './area-stochastic.js';

// mode, motion, sdf, filename, criterionName
const ROWS = [
  { mode: 'restir-still', motion: false, sdf: false,
    filename: 'gi-area-restir-still-converged-one-soft-rectangular-penumbra-no-blotches.png',
    criterionName: 'M7 ReSTIR area light shadows: still camera after convergence' },
  { mode: 'restir-moving', motion: true, sdf: false,
    filename: 'gi-area-restir-moving-camera-no-boiling-or-blotchy-noise-in-penumbra.png',
    criterionName: 'M7 ReSTIR area light shadows: moving camera' },
  { mode: 'restir-sdf-occluder', motion: false, sdf: true,
    filename: 'gi-area-restir-sdf-sphere-and-box-cast-soft-penumbrae-on-floor.png',
    criterionName: 'M7 ReSTIR area light shadows: SDF occluders' },
];
const DESCRIPTIONS = {
  'restir-still': 'The Cornell box with the ReSTIR ultra-tier area-light shadows and a still camera after convergence.',
  'restir-moving': 'The Cornell box with the ReSTIR ultra-tier area-light shadows while the camera sways slightly.',
  'restir-sdf-occluder': 'The Cornell box with the ReSTIR ultra tier and an SDF sphere and box in front of the ceiling area light.',
};
const STILL_BEHAVIOUR = 'The two white blocks cast soft shadows of the ceiling area light on the floor and on the red wall. Each block casts ONE continuous shadow whose penumbra is a single smooth gradient, sharp where the shadow meets the block base and wider and softer with distance from the block, like the shadows in the reference. The penumbrae and the lit surfaces are clean: no grain, no speckles, no blotchy or splotchy correlated patches, no bright fireflies and no banding or stepped edges. Bounce light lights the ceiling. Brightness and colour-bleed differences from the path-traced reference are not faults.';
const MOVING_BEHAVIOUR = 'The camera is swaying slightly. The shadows of the two blocks stay single smooth penumbrae, sharp at the block bases and widening with distance; there is no boiling (no flickering bright or dark blotches), no grain or speckle, no blotchy correlated patches and no smeared, doubled or trailing ghost copy of any shadow edge; walls and blocks stay sharp. Brightness and colour-bleed differences from the path-traced reference are not faults.';
const SDF_BEHAVIOUR = 'Besides the two white blocks, a sphere stands on the floor at the front right and a box at the back left. The sphere and the box each cast a soft shadow of the ceiling area light onto the floor: one smooth penumbra per object, darkest at the object base and widening away from it, with no grain, blotches or boiling noise, like the block shadows. The colour or brightness of the sphere and box themselves is not judged. Brightness and colour-bleed differences from the path-traced reference are not faults.';
const BEHAVIOUR = {
  'restir-still': STILL_BEHAVIOUR,
  'restir-moving': MOVING_BEHAVIOUR,
  'restir-sdf-occluder': SDF_BEHAVIOUR,
};
const BASE_FACTS = ['snapshot:settings', 'snapshot:stats', 'runtime:ready', 'assertion:gi-active', 'assertion:area-restir', 'assertion:area-shadow-pass'];
const SDF_FACTS = [...BASE_FACTS, 'assertion:sdf-content'];
export const factNames = ['gi-active', 'area-restir', 'area-shadow-pass', 'sdf-content'];
const heldOf = row => {
  const held = [...CORNELL_HELD, 'area-light shadow tier auto with stochasticCount 1 and stochasticMode restir (ultra tier)'];
  if (row.motion) held.push(MOVING_HELD);
  if (row.sdf) held.push(SDF_HELD);
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
    stochasticCount: 1, stochasticDistance: 40, stochasticCells: 4, stochasticMode: 'restir',
  },
  motion: row.motion,
  sdf: row.sdf,
}));

export const inventory = ROWS.map(row => ({
  id: `area-restir-${row.mode}`,
  criterionName: row.criterionName,
  scenario: 'area-restir',
  mode: row.mode,
  filename: row.filename,
  expectedVisibleBehavior: BEHAVIOUR[row.mode],
  requiredFacts: row.sdf ? SDF_FACTS : BASE_FACTS,
  pair: null,
  heldConstant: heldOf(row),
  producer: 'raa-gi-area-restir',
  judgeReference: '/refs/cornell-box-reference.png',
  judgeAspect: 'functional',
}));

// The ReSTIR tier check is this scene's own; GI, the gi.area-shadow pass and the SDF content come from areaAssertions.
export async function observe(engine, variant, state) {
  const area = engine.areaLightShadows();
  const tierCheck = {
    name: 'area-restir',
    passed: area.tier === 'auto' && area.heroCount === 0 && area.stochasticCount === 1
      && area.stochasticMode === 'restir' && area.activeStochastic === 1,
    detail: `tier ${area.tier} (expected auto), heroCount ${area.heroCount} (expected 0), `
      + `stochasticCount ${area.stochasticCount} (expected 1), stochasticMode ${area.stochasticMode} (expected restir), `
      + `activeStochastic ${area.activeStochastic} (expected 1)`,
  };
  return { instances: state.instances, assertions: areaAssertions(engine, variant, tierCheck), edits: [] };
}

// Nothing to wait for in this stage: the GI settle gate of the runtime waits for giStats() on GI rows.
function ready() {
  return { ready: true, stage: 'lighting' };
}

export default {
  id: 'area-restir',
  milestone: 'lighting',
  title: 'Area light ray-traced shadows (ReSTIR ultra tier)',
  variants,
  build,
  ready,
  frame,
};
