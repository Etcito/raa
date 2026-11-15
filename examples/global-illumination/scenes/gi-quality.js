// GI quality rows (M9, contract 'gi-quality-rows') on the Cornell box of scenes/lighting.js. Row 1 turns every
// lighting part on at once - the LTC ceiling area light, the tiered shadows with the ReSTIR ultra tier, SSGI and
// the voxel GI with screen probes - row 2 turns every part off (direct light only) and row 3 keeps only the far
// cone shadow tier. The rows are judged against /refs/cornell-box-reference.png: row 1 with aspect 'look' over the
// whole image, above all for the GI colour bleed (the bounce-lit ceiling, floor and block fronts and the red and
// green wall tints), which is this milestone's job; rows 2 and 3 for the direct-light distribution and the soft
// cone-tier shadow edges. Importing this module has no side effects.
import { CORNELL_HELD, CORNELL_SELECTION, DEFAULT_QUALITY, VOLUME_BYTES } from './lighting.js';
import { build, frame } from './area-stochastic.js';

export const factNames = ['gi-active', 'gi-off', 'all-parts', 'cone-tier'];
const SHADOW_BASE = { tier: 'auto', heroCount: 0, heroDistance: 25, fadeDistance: 80, band: 5, stochasticDistance: 40, stochasticCells: 4 };
const ALL_ON_SHADOWS = { ...SHADOW_BASE, stochasticCount: 1, stochasticMode: 'restir' };
const CONE_SHADOWS = { ...SHADOW_BASE, stochasticCount: 0 };
const ALL_ON_FACTS = ['snapshot:settings', 'snapshot:stats', 'runtime:ready', 'assertion:gi-active', 'assertion:all-parts'];
const OFF_FACTS = ['snapshot:settings', 'snapshot:stats', 'runtime:ready', 'assertion:gi-off'];
const CONE_FACTS = ['snapshot:settings', 'snapshot:stats', 'runtime:ready', 'assertion:gi-active', 'assertion:cone-tier'];
const VOLUME_HELD = 'global illumination volume and screen probes at quality 96/5';
// mode, filename, criterionName, gi, areaShadows, giExpected, requiredFacts, producer, judgeAspect, held, behaviour
const ROWS = [
  { mode: 'cornell-all-on', filename: 'gi-quality-cornell-box-all-parts-on-bright-bounce-and-strong-colour-bleed.png',
    criterionName: 'M9 GI quality: Cornell box with every lighting part on against the path-traced reference',
    gi: { enabled: true, debugView: 'none', screenSpace: true, screenSpaceQuality: 'high' }, areaShadows: ALL_ON_SHADOWS,
    giExpected: 'active', requiredFacts: ALL_ON_FACTS, producer: 'raa-gi-screen', judgeAspect: 'look', held: [VOLUME_HELD,
      'screen-space indirect light quality high', 'area-light shadow tier auto with stochasticCount 1 and stochasticMode restir (ultra tier)'] },
  { mode: 'cornell-all-off', filename: 'gi-quality-cornell-box-all-parts-off-direct-light-only.png',
    criterionName: 'M9 GI quality: Cornell box with GI off, direct light only',
    gi: { enabled: false, debugView: 'none' }, areaShadows: ALL_ON_SHADOWS,
    giExpected: 'disabled', requiredFacts: OFF_FACTS, producer: 'raa-area-ltc', judgeAspect: 'functional', held: ['global illumination disabled'] },
  { mode: 'cornell-cone-tier-on', filename: 'gi-quality-cornell-box-cone-tier-soft-block-shadows-no-voxel-steps.png',
    criterionName: 'M9 GI quality: far cone shadow tier soft, without voxel steps',
    gi: { enabled: true, debugView: 'none' }, areaShadows: CONE_SHADOWS,
    giExpected: 'active', requiredFacts: CONE_FACTS, producer: 'raa-shader-gi', judgeAspect: 'functional', held: [VOLUME_HELD,
      'area-light shadow tier auto with stochasticCount 0 (far cone tier only)'] },
];
const DESCRIPTIONS = {
  'cornell-all-on': 'The Cornell box with every lighting part on: the area light, the soft shadow tiers, ReSTIR, SSGI and the voxel GI.',
  'cornell-all-off': 'The same Cornell box with every lighting part off: the direct light of the ceiling area light only.',
  'cornell-cone-tier-on': 'The Cornell box with the far cone shadow tier alone over the voxel global illumination.',
};
const BEHAVIOUR = {
  'cornell-all-on': 'Every lighting part is on: the ceiling area light with its soft shadows, global illumination and screen-space indirect light. Compared with the path-traced reference: the ceiling is clearly lit by warm bounce light, a light warm orange like the reference, not a dark brown; the floor is the brightest large surface, a light warm cream; the front faces of the two white blocks, which the light does not reach directly, are clearly lit by warm bounce light from the floor (the short block front is a visible warm tan, not black) and stay crisp, not smeared; the tall block side facing the red wall is strongly tinted red and the short block side facing the green wall clearly green, and the floor and ceiling near each wall take the same tint, about as strongly as in the reference; the back wall is evenly lit with no dark lobe or blotch beside the tall block other than the soft shadow the block itself casts; each block casts one soft shadow whose penumbra widens with distance, with no staircase or voxel-stepped edges, grain or blotches. Small differences in fine shadow shape and slight overall brightness differences are not faults.',
  'cornell-all-off': 'Global illumination is off; the same box, area light and exposure. Compared with the reference only for how the direct light of the ceiling light is distributed: the floor under the light is the brightest region and stays well lit towards the front of the box, the back wall is evenly lit in its middle and falls off towards its top corners, and the tops of the two blocks are bright; the ceiling and the faces turned away from the light stay dark and there is no red or green colour bleeding. Differences from the path-traced reference in overall brightness, missing bounce light, shadows or colour bleeding are not faults here.',
  'cornell-cone-tier-on': 'Only the far cone shadow tier shades the area light: each white block casts one soft shadow of the ceiling light on the floor, darkest at the block base and widening into a smooth gradient with distance; the shadow edges are smooth curves with no staircase, voxel-sized steps or blocky contours, and the open floor and walls away from the blocks show no dark blotches or contour bands. Bounce light lights the ceiling. Brightness and colour-bleed differences from the path-traced reference are not faults.',
};

// Every row is the Cornell box at DEFAULT_QUALITY with its own GI patch and shadow tier; none moves and none adds SDF content.
const variants = ROWS.map(row => ({
  id: row.mode,
  description: DESCRIPTIONS[row.mode],
  references: [],
  judge: true,
  arrangement: 'cornell-box',
  gi: row.gi,
  quality: { ...DEFAULT_QUALITY },
  giExpected: row.giExpected,
  selection: CORNELL_SELECTION,
  areaShadows: row.areaShadows,
  motion: false,
  sdf: false,
}));

export const inventory = ROWS.map(row => ({
  id: `gi-quality-${row.mode}`,
  criterionName: row.criterionName,
  scenario: 'gi-quality',
  mode: row.mode,
  filename: row.filename,
  expectedVisibleBehavior: BEHAVIOUR[row.mode],
  requiredFacts: row.requiredFacts,
  pair: null,
  heldConstant: [...CORNELL_HELD, ...row.held],
  producer: row.producer,
  judgeReference: '/refs/cornell-box-reference.png',
  judgeAspect: row.judgeAspect,
}));

const SSGI_ORDER = 44;
const AREA_SHADOW_ORDER = 43;
const check = (name, passed, detail) => ({ name, passed: passed === true, detail });

// `observe` reports the four assertions of factNames in order, reading giStats(), globalIllumination(),
// areaLightShadows() and passes(); each detail names every value read.
export async function observe(engine, variant, state) {
  const stats = engine.giStats();
  const settings = engine.globalIllumination();
  const area = engine.areaLightShadows();
  const passes = engine.passes();
  const ssgi = passes.filter(pass => pass.name === 'gi.ssgi');
  const ssgiRow = ssgi[0];
  const shadow = passes.filter(pass => pass.name === 'gi.area-shadow');
  const shadowRow = shadow[0];
  const bytes = VOLUME_BYTES['96/5'];
  const assertions = [
    check('gi-active',
      stats.active === true && stats.pendingRebuild === false && stats.framesSinceRebuild >= 240
        && stats.allocatedBytes >= bytes && stats.voxelizedForwardInstances >= 1,
      `active ${stats.active}, pendingRebuild ${stats.pendingRebuild}, framesSinceRebuild ${stats.framesSinceRebuild}, `
        + `allocatedBytes ${stats.allocatedBytes} (expected at least ${bytes}), voxelizedForwardInstances ${stats.voxelizedForwardInstances}`),
    check('gi-off', stats.active === false && stats.inactiveReason === 'disabled' && stats.allocatedBytes === 0,
      `active ${stats.active}, inactiveReason ${stats.inactiveReason}, allocatedBytes ${stats.allocatedBytes}`),
    check('all-parts',
      settings.screenSpace === true && settings.screenSpaceQuality === 'high'
        && area.tier === 'auto' && area.stochasticMode === 'restir' && area.activeStochastic === 1
        && ssgi.length === 1 && ssgiRow.enabled === true && ssgiRow.order === SSGI_ORDER && ssgiRow.role === 'pass'
        && shadow.length === 1 && shadowRow.enabled === true && shadowRow.order === AREA_SHADOW_ORDER && shadowRow.role === 'pass',
      `screenSpace ${settings.screenSpace} (expected true), screenSpaceQuality ${settings.screenSpaceQuality} (expected high), `
        + `tier ${area.tier} (expected auto), stochasticMode ${area.stochasticMode} (expected restir), `
        + `activeStochastic ${area.activeStochastic} (expected 1), gi.ssgi rows ${ssgi.length} (expected 1), `
        + `gi.ssgi enabled ${ssgiRow?.enabled}, order ${ssgiRow?.order}, role ${ssgiRow?.role} (expected true, ${SSGI_ORDER}, pass), `
        + `gi.area-shadow rows ${shadow.length} (expected 1), gi.area-shadow enabled ${shadowRow?.enabled}, `
        + `order ${shadowRow?.order}, role ${shadowRow?.role} (expected true, ${AREA_SHADOW_ORDER}, pass)`),
    check('cone-tier', area.tier === 'auto' && area.stochasticCount === 0 && area.activeStochastic === 0,
      `tier ${area.tier} (expected auto), stochasticCount ${area.stochasticCount} (expected 0), `
        + `activeStochastic ${area.activeStochastic} (expected 0)`),
  ];
  return { instances: state.instances, assertions, edits: [] };
}

// Nothing to wait for in this stage: the GI settle gate of the runtime waits for giStats() on GI rows.
function ready() {
  return { ready: true, stage: 'lighting' };
}

export default {
  id: 'gi-quality',
  milestone: 'lighting',
  title: 'GI quality: the Cornell box with every lighting part',
  variants,
  build,
  ready,
  frame,
};
