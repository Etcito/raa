// Screen-space GI (M8, contract 'gi-ssgi-rows') on the voxel GI: three SSGI-on rows. cornell-on and cornell-moving-on
// are the Cornell box of scenes/lighting.js (quality high and medium; the second sways its camera), room-on is the
// red-and-green room with a white box and sphere at quality medium. Each row shows the near-field bounce that
// screen-space indirect light adds. Only SSGI-on rows render: the owner-accepted looks gi-cornell-box-on-bounce-lit.png
// and gi-color-bleed-white-box-tinted-red-green.png were judged before M3-M7, so they are comparison looks and not
// current SSGI-off frames, and no judge text here relies on an off frame. Importing this module has no side effects.
import * as lighting from './lighting.js';
import { frame, MOVING_HELD } from './area-stochastic.js';

// The camera and subjects of the colour-bleed room, in metres: the box, the sphere and the floor-to-wall corners.
const ROOM_SELECTION = {
  camera: { eye: [0, 1.2, -1.6], target: [0.4, 0.35, -5.6], fovYDeg: 60, near: 0.05, far: 200 },
  subjects: [
    { name: 'box', min: [0.7, 0, -5.7], max: [1.7, 1, -4.7] },
    { name: 'sphere', min: [-1.8, 0, -5.2], max: [-0.6, 1.2, -4.0] },
    { name: 'floor-wall-corners', min: [-3, 0, -8], max: [3, 0.3, -4] },
  ],
};
const SELECTIONS = { 'cornell-box': lighting.CORNELL_SELECTION, 'color-bleed': ROOM_SELECTION };
// mode, arrangement, quality (the screen-space indirect light quality), motion, filename, criterionName
const ROWS = [
  { mode: 'cornell-on', arrangement: 'cornell-box', quality: 'high', motion: false,
    filename: 'gi-ssgi-cornell-box-on-contact-bleed-and-floor-lit-block-fronts.png',
    criterionName: 'M8 screen-space GI: Cornell box contact bleed and floor-lit block fronts' },
  { mode: 'room-on', arrangement: 'color-bleed', quality: 'medium', motion: false,
    filename: 'gi-ssgi-room-on-box-and-sphere-bases-and-wall-corners-show-contact-bounce.png',
    criterionName: 'M8 screen-space GI: room contact bounce at box, sphere and wall corners' },
  { mode: 'cornell-moving-on', arrangement: 'cornell-box', quality: 'medium', motion: true,
    filename: 'gi-ssgi-cornell-box-moving-camera-no-halos-ghosts-or-crawling-near-blocks.png',
    criterionName: 'M8 screen-space GI: moving camera stability' },
];
const DESCRIPTIONS = {
  'cornell-on': 'The Cornell box with screen-space indirect light (visibility bitmask, quality high) on top of the voxel global illumination.',
  'room-on': 'The red-and-green room with screen-space indirect light (quality medium) on top of the voxel global illumination.',
  'cornell-moving-on': 'The Cornell box with screen-space indirect light (quality medium) while the camera sways slightly.',
};
const BEHAVIOUR = {
  'cornell-on': 'Screen-space indirect light adds near-field bounce detail on top of the voxel global illumination. Where the two white blocks stand on the floor, a soft, narrow contact band darkens the floor right at each block base, with no hard black line. Close to where surfaces meet, the floor and block faces near the red left wall pick up a red tint and those near the green right wall a green tint, and the floor-to-wall and wall-to-wall corners show a soft darkening band tinted by the adjacent wall. The front faces of the two blocks, which the ceiling light does not reach directly, are clearly lit by warm bounce light from the bright floor in front of them, brighter towards their bottom edge. No dark or bright halo outlines the blocks against the walls, there is no grain, speckle or banding, and nothing streaks or smears at the image edges. The reference is a path-traced Cornell box given as context for the scene: overall brightness, the overall strength of colour bleeding across the box, shadow shapes and voxel-scale smoothness of the far bounce are not faults here.',
  'room-on': 'Screen-space indirect light adds near-field bounce detail on top of the voxel global illumination in the red-and-green room. At the bases of the white box and the white sphere the floor shows a soft, narrow contact shadow, and the lower faces of the box and the underside of the sphere pick up soft bounce light from the floor. The floor-to-wall corners show a soft contact darkening tinted by the wall colour: red along the red left wall, green along the green right wall. No dark or bright halo outlines the box or the sphere, there is no grain, speckle or banding, and nothing streaks or smears at the image edges. The overall strength of colour bleeding across the room is not judged.',
  'cornell-moving-on': 'The camera sways slightly while screen-space indirect light is on over the voxel global illumination. The contact darkening at the block bases and the floor bounce on the block fronts stay stable and smooth: there is no crawling or flickering noise, no dark or bright halo outlining the blocks, no smeared, doubled or trailing ghost copy of the block edges or of their contact bands, and no streaks at the image edges where new floor comes into view. Overall brightness and colour-bleed differences from the path-traced reference are not faults.',
};
export const factNames = ['gi-active', 'ssgi-on', 'ssgi-pass'];
const FACTS = ['snapshot:settings', 'snapshot:stats', 'runtime:ready', 'assertion:gi-active', 'assertion:ssgi-on', 'assertion:ssgi-pass'];
const VOLUME_HELD = 'global illumination volume and screen probes at quality 96/5';
const ROOM_HELD = [
  'room shell with red left wall and green right wall, closed at the front behind the camera',
  'white box and sphere', 'sun through the ceiling opening', 'camera',
];
// The judge-held fixtures of a row: its arrangement, the volume, its screen-space quality and, for the moving row, its sway.
const heldOf = row => [
  ...(row.arrangement === 'color-bleed' ? ROOM_HELD : lighting.CORNELL_HELD), VOLUME_HELD,
  `screen-space indirect light quality ${row.quality}`, ...(row.motion ? [MOVING_HELD] : []),
];
const CORNELL_JUDGE = { judgeReference: '/refs/cornell-box-reference.png', judgeAspect: 'functional' };

// The volume is the GI volume at DEFAULT_QUALITY (96/5/8). Every row is an SSGI-on variant without a measurement.
const variants = ROWS.map(row => ({
  id: row.mode,
  description: DESCRIPTIONS[row.mode],
  references: [],
  judge: true,
  arrangement: row.arrangement,
  gi: { enabled: true, debugView: 'none', screenSpace: true, screenSpaceQuality: row.quality },
  quality: { ...lighting.DEFAULT_QUALITY },
  giExpected: 'active',
  selection: SELECTIONS[row.arrangement],
  motion: row.motion,
}));

export const inventory = ROWS.map(row => ({
  id: `ssgi-${row.mode}`,
  criterionName: row.criterionName,
  scenario: 'ssgi',
  mode: row.mode,
  filename: row.filename,
  expectedVisibleBehavior: BEHAVIOUR[row.mode],
  requiredFacts: FACTS,
  pair: null,
  heldConstant: heldOf(row),
  producer: 'raa-gi-ssgi',
  ...(row.arrangement === 'cornell-box' ? CORNELL_JUDGE : {}),
}));

const SSGI_ORDER = 44;
const check = (name, passed, detail) => ({ name, passed: passed === true, detail });

// `observe` reports the three assertions of factNames in order, reading giStats(), globalIllumination() and
// passes(); each detail names every value it reads.
export async function observe(engine, variant, state) {
  const stats = engine.giStats();
  const settings = engine.globalIllumination();
  const rows = engine.passes().filter(pass => pass.name === 'gi.ssgi');
  const row = rows[0];
  const bytes = lighting.VOLUME_BYTES['96/5'];
  const assertions = [
    check('gi-active',
      stats.active === true && stats.pendingRebuild === false && stats.framesSinceRebuild >= 240
        && stats.allocatedBytes >= bytes && stats.voxelizedForwardInstances >= 1,
      `active ${stats.active}, pendingRebuild ${stats.pendingRebuild}, framesSinceRebuild ${stats.framesSinceRebuild}, `
        + `allocatedBytes ${stats.allocatedBytes} (expected at least ${bytes}), `
        + `voxelizedForwardInstances ${stats.voxelizedForwardInstances} (expected at least 1)`),
    check('ssgi-on', settings.screenSpace === true && settings.screenSpaceQuality === variant.gi.screenSpaceQuality,
      `screenSpace ${settings.screenSpace} (expected true), screenSpaceQuality ${settings.screenSpaceQuality} `
        + `(expected ${variant.gi.screenSpaceQuality})`),
    check('ssgi-pass', rows.length === 1 && row.enabled === true && row.order === SSGI_ORDER && row.role === 'pass',
      `gi.ssgi rows ${rows.length} (expected 1), enabled ${row?.enabled} (expected true), order ${row?.order} `
        + `(expected ${SSGI_ORDER}), role ${row?.role} (expected pass)`),
  ];
  return { instances: state.instances, assertions, edits: [] };
}

// The lighting arrangement, quality, camera and GI come from lighting.default.build. The moving row then sways the
// Cornell camera about the box centre, and the frame count of the sway starts at 0.
async function build(engine, lib, variant) {
  const state = await lighting.default.build(engine, lib, variant);
  state.frame = 0;
  if (variant.motion) engine.setOrbit([2.78, 2.73, 0], 8, Math.PI, 0);
  return state;
}

// Nothing to wait for in this stage: the GI settle gate of the runtime waits for giStats() on GI rows.
function ready() {
  return { ready: true, stage: 'lighting' };
}

export default {
  id: 'ssgi',
  milestone: 'lighting',
  title: 'Screen-space GI (visibility bitmask) on the voxel GI',
  variants,
  build,
  ready,
  frame,
};
