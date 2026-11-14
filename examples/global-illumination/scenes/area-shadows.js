// Area light soft shadows (M4, contract 'gi-area-shadow-rows') on the Cornell box of scenes/lighting.js.
// Five rows show the ceiling area light under each shadow tier: hero with GI on (perspective shadow maps
// filtered with PCSS), hero with GI off, cone (soft visibility through the GI distance field), fade (the
// direct term faded out, so light reaches the box through the GI volume only) and automatic selection.
// The block shadows on the floor and on the red wall should widen with distance from the block base and
// stay smooth, without voxel steps. Importing this module has no side effects.
import { CORNELL_HELD, CORNELL_SELECTION, DEFAULT_QUALITY, VOLUME_BYTES, cornellBox } from './lighting.js';

const SHADOW = { heroDistance: 25, fadeDistance: 80, band: 5 };
// mode, gi.enabled, tier, heroCount, giExpected, expectedHeroes, filename, producer
const ROWS = [
  { mode: 'hero-gi-on', gi: true, tier: 'hero', heroCount: 1, giExpected: 'active', expectedHeroes: 1, filename: 'gi-area-shadow-hero-pcss-penumbra-widens-from-block-base.png', producer: 'raa-area-ltc' },
  { mode: 'hero-gi-off', gi: false, tier: 'hero', heroCount: 1, giExpected: 'disabled', expectedHeroes: 1, filename: 'gi-area-shadow-hero-pcss-gi-off-soft-shadows-without-bounce.png', producer: 'raa-area-ltc' },
  { mode: 'cone-gi-on', gi: true, tier: 'cone', heroCount: 0, giExpected: 'active', expectedHeroes: 0, filename: 'gi-area-shadow-cone-trace-smooth-penumbra-no-voxel-steps.png', producer: 'raa-gi-screen' },
  { mode: 'fade-gi-on', gi: true, tier: 'fade', heroCount: 0, giExpected: 'active', expectedHeroes: 0, filename: 'gi-area-shadow-far-fade-light-through-gi-volume-only.png', producer: 'raa-area-ltc' },
  { mode: 'auto-gi-on', gi: true, tier: 'auto', heroCount: 1, giExpected: 'active', expectedHeroes: 1, filename: 'gi-area-shadow-auto-selection-soft-shadows-and-bounce.png', producer: 'raa-area-ltc' },
];
const DESCRIPTIONS = {
  'hero-gi-on': 'Hero tier with one shadow-mapped area light and global illumination on: PCSS-filtered shadow maps with bounce light.',
  'hero-gi-off': 'Hero tier with one shadow-mapped area light and global illumination off: the same shadow maps without bounce light.',
  'cone-gi-on': 'Cone tier with global illumination on: the area light visibility is traced through the GI distance field.',
  'fade-gi-on': 'Fade tier with global illumination on: the direct area light is faded out and light arrives through the GI volume only.',
  'auto-gi-on': 'Automatic tier selection with global illumination on: the nearby ceiling light gets a hero shadow map.',
};
const BEHAVIOUR = {
  'hero-gi-on': 'The two white blocks cast soft shadows of the ceiling area light on the floor: each shadow is sharp where it touches the base of its block and its edge becomes wider and softer with distance from the block, like the shadows in the reference; the shadow edges are smooth gradients with no stair steps or blocky voxel pattern; the ceiling and the block faces turned away from the light are lit by bounce light and the red and green walls tint the nearby white surfaces. The reference is a path-traced Cornell box given as context: differences in overall brightness or bounce smoothness are not faults.',
  'hero-gi-off': 'With global illumination off, the two white blocks still cast soft shadows of the ceiling area light on the floor: sharp at the base of each block and widening with distance from it, smooth with no stair steps; the ceiling and the faces turned away from the light stay dark and there is no red or green colour bleeding. Compared with the reference only for the shape of the shadows: missing bounce light and brightness differences are not faults.',
  'cone-gi-on': 'The shadows of the two blocks on the floor and on the red wall are soft, with smooth gradual penumbrae that widen with distance from the block and are darkest near the block base; no jagged, stair-stepped or blocky voxel edges appear along any shadow; bounce light lights the ceiling and the red and green walls tint the nearby white surfaces. Differences from the path-traced reference in brightness or slightly softer contact shadows are not faults.',
  'fade-gi-on': 'The direct light of the ceiling area light is faded out: there is no bright direct pool on the floor under the light and no sharply outlined shadows; the box is lit only softly by indirect light arriving through the global illumination volume, so the surfaces show low-contrast light with red and green tints near the coloured walls. The image being much darker and flatter than the reference is expected, not a fault.',
  'auto-gi-on': 'With automatic tier selection the nearby ceiling light gets a shadow map: the two blocks cast soft shadows on the floor that are sharp at the block bases and widen with distance, with smooth edges and no stair steps or voxel pattern; bounce light lights the ceiling and the coloured walls tint the nearby white surfaces. Brightness differences from the path-traced reference are not faults.',
};
const ON_FACTS = ['snapshot:settings', 'snapshot:stats', 'runtime:ready', 'assertion:gi-active', 'assertion:area-tier'];
const OFF_FACTS = ['snapshot:settings', 'snapshot:stats', 'runtime:ready', 'assertion:gi-off', 'assertion:area-tier'];
export const factNames = ['gi-active', 'gi-off', 'area-tier'];

const areaShadows = row => ({ tier: row.tier, heroCount: row.heroCount, ...SHADOW });
const variants = ROWS.map(row => ({
  id: row.mode,
  description: DESCRIPTIONS[row.mode],
  references: [],
  judge: true,
  arrangement: 'cornell-box',
  gi: { enabled: row.gi, debugView: 'none' },
  quality: { ...DEFAULT_QUALITY },
  giExpected: row.giExpected,
  selection: CORNELL_SELECTION,
  areaShadows: areaShadows(row),
  expectedHeroes: row.expectedHeroes,
}));

export const inventory = ROWS.map(row => ({
  id: `area-shadows-${row.mode}`,
  criterionName: row.tier === 'auto' ? 'M4 area light soft shadows: automatic selection' : `M4 area light soft shadows: ${row.tier} tier`,
  scenario: 'area-shadows',
  mode: row.mode,
  filename: row.filename,
  expectedVisibleBehavior: BEHAVIOUR[row.mode],
  requiredFacts: row.gi ? ON_FACTS : OFF_FACTS,
  pair: null,
  heldConstant: [...CORNELL_HELD, `area-light shadow tier ${row.tier} with heroCount ${row.heroCount}`],
  producer: row.producer,
  judgeReference: '/refs/cornell-box-reference.png',
  judgeAspect: 'functional',
}));

// The Cornell box, its quality, the tier settings and the camera; then the variant's global illumination.
export async function build(engine, lib, variant) {
  const state = { instances: [], moved: false, blockNode: null };
  engine.setGiQuality(variant.quality);
  cornellBox(engine, variant, state);
  engine.setAreaLightShadows(variant.areaShadows);
  engine.setCamera(variant.selection.camera);
  engine.setGlobalIllumination(variant.gi);
  return state;
}

const check = (name, passed, detail) => ({ name, passed: passed === true, detail });

// `observe` reports the three assertions of factNames, in order, with the rules of lighting.js observe for
// gi-active and gi-off (the volume at 96/5), and area-tier from areaLightShadows() against the variant.
export async function observe(engine, variant, state) {
  const stats = engine.giStats();
  const area = engine.areaLightShadows();
  const wanted = variant.areaShadows;
  const bytes = VOLUME_BYTES['96/5'];
  const assertions = [
    check('gi-active',
      stats.active === true && stats.pendingRebuild === false && stats.framesSinceRebuild >= 240
        && stats.allocatedBytes >= bytes && stats.voxelizedForwardInstances >= 1,
      `active ${stats.active}, pendingRebuild ${stats.pendingRebuild}, framesSinceRebuild ${stats.framesSinceRebuild}, `
        + `allocatedBytes ${stats.allocatedBytes} (expected at least ${bytes}), voxelizedForwardInstances ${stats.voxelizedForwardInstances}`),
    check('gi-off',
      stats.active === false && stats.inactiveReason === 'disabled' && stats.allocatedBytes === 0,
      `active ${stats.active}, inactiveReason ${stats.inactiveReason}, allocatedBytes ${stats.allocatedBytes}`),
    check('area-tier',
      area.tier === wanted.tier && area.heroCount === wanted.heroCount && area.activeHeroes === variant.expectedHeroes,
      `tier ${area.tier} (expected ${wanted.tier}), heroCount ${area.heroCount} (expected ${wanted.heroCount}), `
        + `activeHeroes ${area.activeHeroes} (expected ${variant.expectedHeroes})`),
  ];
  return { instances: state.instances, assertions, edits: [] };
}

// Nothing to wait for in this stage: the GI settle gate of the runtime waits for giStats() on GI rows.
function ready() {
  return { ready: true, stage: 'lighting' };
}

export default {
  id: 'area-shadows',
  milestone: 'lighting',
  title: 'Area light soft shadows',
  variants,
  build,
  ready,
};
