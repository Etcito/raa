// Lighting stage (S40) of the global-illumination page, rows 1-23: colour bleeding, interior bounce,
// the emissive panel, local-light shadows, the sky overhang, glossy reflection, quality, forward
// content, the moved block and the Cornell box (ten times the canonical size, metres = mm / 100) with
// its two linear-HDR probe rows. Importing this module has no side effects. Every row is an on/off pair
// on one arrangement, built from lib/content.js builders; observe reports the evidence of contract
// 'gi-lighting-rows'. `ready` holds the moved-instance arrangement until its block has moved.
import { box, panel, plane, roomShell, sphere, volumeRoom } from '../lib/content.js';

const room = volumeRoom();
const DEFAULT_QUALITY = { resolution: 96, levels: 5, probeRays: 8 };
const VOLUME_BYTES = {
  '96/5': 148635648, '96/4': 120324096, '96/3': 92012544,
  '64/5': 44040192, '64/4': 35651584, '64/3': 27262976,
};
// Cornell box (S40 amendment, product-owner directive 2026-11-14). The data is canonical millimetres / 100,
// so the 5.56 m room is 2.8 finest voxels wider than the canonical box would be at mm / 1000.
// The light centre height: 5.088, 0.4 m (two finest voxels) under the 5.488 ceiling, by the lowering rule.
// At the canonical 5.487 (1 mm under the ceiling) the light sat inside the ceiling's voxel layer, so the
// volume visibility of the GI-on row found the ceiling between the light and the upper walls and shaded
// them in streaks. The visible fixture stays at CORNELL_FIXTURE_Y whatever the light height is.
export const CORNELL_LIGHT_Y = 5.088;
export const CORNELL_FIXTURE_Y = 5.4875;
// The shared exposure (a linear multiplier, applied to the image as tone mapping exposureEv = log2 of it).
// Derivation, measured on /refs/cornell-box-reference.png: the mean of the 40 x 40 px patch centred at (640, 230)
// is sRGB (221.4, 173.3, 71.2). Inverting the page's AgX tonemap (sRGB OETF, AgX sigmoid, 2.2 display decode)
// and taking Rec. 709 luminance gives the target scene-linear value after exposure, Y_t = 0.5797. The back-wall
// centre (2.78, 2.74, 5.592) under the S17 area formula, with the light at y 5.487, white albedo
// (0.886, 0.699, 0.666), colour (1, 0.598, 0.149), intensity 18.4 x 1.365 and the Karis representative point,
// has direct luminance Y_d = 0.19819. exposure = Y_t / (1.35 Y_d) = 0.5797 / 0.26756 = 2.1666 (EV +1.1154).
// It is derived for the light at 5.487 and stays fixed when CORNELL_LIGHT_Y changes.
export const CORNELL_EXPOSURE = 2.1666;
// The linear probe rows use a quarter of the Cornell exposure with the linear mapper and no dither, so the
// capture stores sRGB-encoded scene-linear values that the ratio script can decode.
export const CORNELL_LINEAR_EXPOSURE_SCALE = 0.25;
const ON_FACTS = [
  'snapshot:settings', 'snapshot:quality', 'snapshot:stats', 'snapshot:instances',
  'runtime:passes', 'runtime:viewSizes', 'runtime:ready', 'assertion:gi-active',
];
const OFF_FACTS = [
  'snapshot:settings', 'snapshot:quality', 'snapshot:stats',
  'runtime:passes', 'runtime:viewSizes', 'runtime:ready', 'assertion:gi-off',
];

export const factNames = [
  'gi-active', 'gi-off', 'local-lights', 'ssr-enabled', 'quality-applied', 'forward-content', 'block-moved',
];

/**
 * `ROWS` lists the rows in table order (contract gi-lighting-rows). Each entry's `measured` flag
 * gives it a PAIR_RUN measurement, and its `extraFacts` are appended to ON_FACTS or OFF_FACTS.
 * Today ROWS holds rows 1-23 of the lighting stage, including the two unmeasured linear-HDR probe
 * rows, and `ready` (a member of the default export) returns { ready: true, stage: 'lighting' }
 * for every variant except moved-instance.
 * Handoff: gi page: scenes/lighting.js rows 11-19 (glossy, quality, forward content, moved block)
 * appends its rows to `ROWS` (examples/global-illumination/scenes/lighting.js:37), its arrangements
 * to `ARRANGEMENTS` and replaces the moved-block logic of `ready`; state/keys: today `build` returns
 * `moved: false` and `blockNode: null` for every row, and no arrangement moves a block.
 */
// The no-GI baseline (the P1-P3 frame): raa-ibl's sky irradiance at diffuse scale 1, unoccluded by the room.
const NO_GI_AMBIENT = 'flat uncoloured sky ambient fill of the frame without global illumination';

const ROWS = [
  { mode: 'color-bleed-on', arrangement: 'color-bleed', filename: 'gi-color-bleed-white-box-tinted-red-green.png',
    criterionName: 'P4-Q3 colour bleeding', producer: 'raa-gi-screen', extraFacts: [], measured: true },
  { mode: 'color-bleed-off', arrangement: 'color-bleed', filename: 'gi-color-bleed-off-white-box-neutral.png',
    criterionName: 'P4-Q8 GI off frame', producer: 'raa-gi-core', extraFacts: [], measured: true },
  { mode: 'interior-bounce-on', arrangement: 'interior-bounce', filename: 'gi-interior-bounce-lit-from-sunlit-patch.png',
    criterionName: 'P4-Q3 interior bounce', producer: 'raa-gi-screen', extraFacts: [], measured: true },
  { mode: 'interior-bounce-off', arrangement: 'interior-bounce', filename: 'gi-interior-bounce-off-dark-interior.png',
    criterionName: 'P4-Q8 GI off frame', producer: 'raa-gi-core', extraFacts: [], measured: true },
  { mode: 'emissive-panel-on', arrangement: 'emissive-panel', filename: 'gi-emissive-panel-lights-nearby-surfaces.png',
    criterionName: 'P4-Q3 emissive panel light', producer: 'raa-gi-screen', extraFacts: [], measured: true },
  { mode: 'emissive-panel-off', arrangement: 'emissive-panel', filename: 'gi-emissive-panel-off-no-spill.png',
    criterionName: 'P4-Q8 GI off frame', producer: 'raa-gi-core', extraFacts: [], measured: true },
  { mode: 'local-light-shadow-on', arrangement: 'local-light-shadow', filename: 'gi-local-lights-stopped-by-wall.png',
    criterionName: 'P4-Q4 local lights stopped by a wall', producer: 'raa-gi-screen',
    extraFacts: ['assertion:local-lights'], measured: true },
  { mode: 'local-light-shadow-off', arrangement: 'local-light-shadow', filename: 'gi-local-lights-off-leak-through-wall.png',
    criterionName: 'P4-Q8 GI off frame', producer: 'raa-gi-core', extraFacts: ['assertion:local-lights'], measured: true },
  { mode: 'sky-overhang-on', arrangement: 'sky-overhang', filename: 'gi-sky-open-ground-kept-overhang-bounce.png',
    criterionName: 'P4-Q5 sky ground kept and overhang bounce', producer: 'raa-gi-screen', extraFacts: [], measured: true },
  { mode: 'sky-overhang-off', arrangement: 'sky-overhang', filename: 'gi-sky-overhang-off.png',
    criterionName: 'P4-Q8 GI off frame', producer: 'raa-gi-core', extraFacts: [], measured: true },
  { mode: 'glossy-reflection-on', arrangement: 'glossy-reflection', filename: 'gi-glossy-floor-reflects-offscreen-object.png',
    criterionName: 'P4-Q5 glossy floor volume reflection', producer: 'raa-ssr-gi-fallback',
    extraFacts: ['assertion:ssr-enabled'], measured: true },
  { mode: 'glossy-reflection-off', arrangement: 'glossy-reflection', filename: 'gi-glossy-floor-off-env-only.png',
    criterionName: 'P4-Q8 GI off frame', producer: 'raa-gi-core', extraFacts: ['assertion:ssr-enabled'], measured: true },
  { mode: 'quality-64-3-on', arrangement: 'quality', filename: 'gi-quality-64-3-room-lit.png',
    criterionName: 'P4-Q6 quality 64/3', producer: 'raa-gi-screen', extraFacts: ['assertion:quality-applied'], measured: false },
  { mode: 'quality-96-5-on', arrangement: 'quality', filename: 'gi-quality-96-5-room-lit.png',
    criterionName: 'P4-Q6 quality 96/5', producer: 'raa-gi-screen', extraFacts: ['assertion:quality-applied'], measured: true },
  { mode: 'quality-off', arrangement: 'quality', filename: 'gi-quality-off-room.png',
    criterionName: 'P4-Q8 GI off frame', producer: 'raa-gi-core', extraFacts: [], measured: true,
    extraHeld: [NO_GI_AMBIENT] },
  { mode: 'forward-content-on', arrangement: 'forward-content', filename: 'gi-skinned-deformed-transparent-fur-receive-bounce.png',
    criterionName: 'P4-Q6 forward content receives GI', producer: 'raa-gi-screen',
    extraFacts: ['assertion:forward-content'], measured: true },
  { mode: 'forward-content-off', arrangement: 'forward-content', filename: 'gi-forward-content-off.png',
    criterionName: 'P4-Q8 GI off frame', producer: 'raa-gi-core', extraFacts: ['assertion:forward-content'], measured: true },
  { mode: 'moved-instance-on', arrangement: 'moved-instance', filename: 'gi-moved-block-lights-new-place.png',
    criterionName: 'P4-Q6 moved instance relit', producer: 'raa-gi-screen', extraFacts: ['assertion:block-moved'], measured: true },
  { mode: 'moved-instance-off', arrangement: 'moved-instance', filename: 'gi-moved-block-off.png',
    criterionName: 'P4-Q8 GI off frame', producer: 'raa-gi-core', extraFacts: ['assertion:block-moved'], measured: true },
  // Cornell rows: `facts` replaces the on/off fact list; the on row carries the reference image as judge context.
  { mode: 'cornell-box-on', arrangement: 'cornell-box', filename: 'gi-cornell-box-on-bounce-lit.png',
    criterionName: 'P4-Q3 Cornell box GI bounce visible', producer: 'raa-gi-screen', extraFacts: [], measured: true,
    facts: ['snapshot:settings', 'snapshot:stats'], judgeReference: '/refs/cornell-box-reference.png', judgeAspect: 'functional' },
  { mode: 'cornell-box-off', arrangement: 'cornell-box', filename: 'gi-cornell-box-off-direct-only.png',
    criterionName: 'P4-Q8 GI off frame', producer: 'raa-gi-core', extraFacts: [], measured: true,
    facts: ['snapshot:settings', 'snapshot:stats'] },
  // Linear-HDR probe rows: the same Cornell box at a quarter of the exposure with linear, undithered tone mapping,
  // so the capture stores sRGB-encoded scene-linear values that the ratio script can decode. They are unmeasured.
  { mode: 'cornell-box-linear-on', arrangement: 'cornell-box-linear', filename: 'gi-cornell-box-linear-on-hdr-ratio-probe.png',
    criterionName: 'P4-Q3 Cornell linear HDR probe', producer: 'raa-gi-screen', extraFacts: [], measured: false,
    facts: ['snapshot:settings', 'snapshot:stats'] },
  { mode: 'cornell-box-linear-off', arrangement: 'cornell-box-linear', filename: 'gi-cornell-box-linear-off-hdr-ratio-probe.png',
    criterionName: 'P4-Q8 GI off frame', producer: 'raa-gi-core', extraFacts: [], measured: false,
    facts: ['snapshot:settings', 'snapshot:stats'] },
];

// Judge-held fixtures per arrangement: the judge never scores them.
const ROOM_HELD = ['room shell with red left wall and green right wall', 'white box and sphere',
  'sun through the ceiling opening', 'camera'];
const CORNELL_HELD = ['Cornell box geometry and wall colours', 'single ceiling area light and its fixture', 'camera', 'exposure'];
const HELD = {
  'color-bleed': ROOM_HELD,
  'interior-bounce': ROOM_HELD,
  'emissive-panel': ['room shell', 'emissive panel', 'white box and sphere', 'night sky without sun', 'camera'],
  'local-light-shadow': ['floor', '0.6 m wall', 'point and spot light placement', 'night sky without sun', 'camera'],
  'sky-overhang': ['open ground', '3 m overhang on four pillars', 'white box under the overhang',
    'physical day sky and sun', 'camera'],
  'glossy-reflection': ['glossy floor', 'orange glowing sphere above the frame', 'day sky', 'camera'],
  quality: ROOM_HELD,
  'forward-content': ['room shell with red left wall', 'skinned figure, squashed shape, transparent pane and fur ball',
    'camera'],
  'moved-instance': ['room shell', 'orange block and its two positions', 'camera'],
  'cornell-box': CORNELL_HELD,
  'cornell-box-linear': [...CORNELL_HELD, 'linear tone mapping without dither at a quarter of the Cornell exposure'],
};

// Camera and subject bounds per arrangement, in metres (camera near 0.05, far 200).
const lens = (eye, target, fovYDeg) => ({ eye, target, fovYDeg, near: 0.05, far: 200 });
const roomSubject = { name: 'room', min: [-3.4, -0.4, -8.4], max: [3.4, 3.4, 0] };
const CORNELL_SELECTION = {
  camera: lens([2.78, 2.73, -8.0], [2.78, 2.73, 0], 39.3),
  subjects: [
    { name: 'cornell-room', min: [0, 0, 0], max: [5.56, 5.488, 5.592] },
    { name: 'light', min: [2.13, 5.4, 2.27], max: [3.43, 5.49, 3.32] },
  ],
};
const SELECTIONS = {
  'color-bleed': {
    camera: lens([0, 1.4, -1.2], [0, 0.6, -5.4], 60),
    subjects: [roomSubject, { name: 'box', min: [0.7, 0, -5.7], max: [1.7, 1, -4.7] }],
  },
  'interior-bounce': {
    camera: lens([2.4, 1.5, -1.0], [-2.0, 1.0, -7.5], 70),
    subjects: [roomSubject, { name: 'sphere', min: [-1.8, 0, -5.2], max: [-0.6, 1.2, -4.0] }],
  },
  'emissive-panel': {
    camera: lens([0, 1.6, -1.5], [0, 0.8, -7], 65),
    subjects: [
      { name: 'panel', min: [-0.8, 0.4, -7.45], max: [0.8, 1.4, -7.35] },
      { name: 'box', min: [0.7, 0, -6.7], max: [1.7, 1, -5.7] },
    ],
  },
  'local-light-shadow': {
    camera: lens([0, 2.0, 1.8], [0, 0.4, -4], 65),
    subjects: [
      { name: 'floor-front', min: [-3, 0, -3.7], max: [3, 0, 1] },
      { name: 'wall', min: [-3, 0, -4.3], max: [3, 2.5, -3.7] },
    ],
  },
  'sky-overhang': {
    camera: lens([7, 2.2, 8], [0, 1.0, 0], 60),
    subjects: [
      { name: 'ground', min: [-10, 0, -10], max: [10, 0, 10] },
      { name: 'overhang', min: [-2.2, 0, -2.2], max: [2.2, 3.3, 2.2] },
    ],
  },
  'glossy-reflection': {
    camera: lens([0, 2, 2], [0, 0, -3], 70),
    subjects: [{ name: 'floor', min: [-6, 0, -12], max: [6, 0, 1] }],
  },
  quality: { camera: room.camera, subjects: room.subjects },
  'forward-content': {
    camera: lens([0.8, 1.4, -2.2], [-2.6, 0.6, -5.2], 60),
    subjects: [{ name: 'left-wall-area', min: [-3.2, 0, -7.2], max: [-1.8, 1.6, -3.0] }],
  },
  'moved-instance': {
    camera: lens([0, 1.8, -0.5], [0, 0.6, -5.5], 70),
    subjects: [
      roomSubject,
      { name: 'block-new', min: [1.6, 0, -3.6], max: [2.4, 0.8, -2.8] },
      { name: 'block-old', min: [-2.4, 0, -7.4], max: [-1.6, 0.8, -6.6] },
    ],
  },
  'cornell-box': CORNELL_SELECTION,
  'cornell-box-linear': CORNELL_SELECTION,
};

// Quality per mode; a mode without an entry uses DEFAULT_QUALITY.
const QUALITY = {
  'quality-64-3-on': { resolution: 64, levels: 3, probeRays: 6 },
};

const BEHAVIOURS = {
  'color-bleed-on': 'The white box and the white floor beside the coloured walls take a visible red tint '
    + 'on the side facing the red left wall and a green tint on the side facing the green right wall.',
  'color-bleed-off': 'The white box and the white floor stay neutral white-grey, with no red or green tint '
    + 'picked up from the coloured walls.',
  'interior-bounce-on': 'The walls, ceiling and objects away from the bright sunlit floor patch are lit by soft '
    + 'bounce light coming from that patch, so the shadowed interior is clearly visible.',
  'interior-bounce-off': 'Away from the bright sunlit floor patch the interior stays dark, lit only by a faint '
    + 'flat ambient term.',
  'emissive-panel-on': 'The floor, the back wall and the white box and sphere next to the glowing panel are lit '
    + 'by warm light spilling from it, brightest close to the panel and fading with distance.',
  'emissive-panel-off': 'Only the panel itself glows; the floor, walls, box and sphere around it stay dark, with '
    + 'no light spilling from the panel.',
  'local-light-shadow-on': 'The floor on the camera side of the wall stays dark in the wall shadow: the point and '
    + 'spot lights behind the wall light only the far side, whose glow shows past the edges of the wall.',
  'local-light-shadow-off': 'Light from the point and spot lights behind the wall leaks through it: the floor on '
    + 'the camera side of the wall is lit as if the wall were not there.',
  'sky-overhang-on': 'The sunlit open ground keeps its normal daylight brightness, while the ground and the white '
    + 'box under the overhang receive visible soft bounce light and are clearly brighter than plain shadow.',
  'sky-overhang-off': 'The ground and the white box under the overhang sit in flat dark shadow lit only by the '
    + 'ambient light of the sky.',
  'glossy-reflection-on': 'The glossy floor clearly reflects a bright orange glowing object that is itself outside the '
    + 'picture, above its top edge.',
  'glossy-reflection-off': 'The glossy floor reflects only the sky environment; no reflection of an orange glowing '
    + 'object appears on it.',
  'quality-64-3-on': 'The red-and-green room is lit by global illumination at the low quality: the walls, floor, box '
    + 'and sphere away from the sunlit patch show soft bounce light tinted by the coloured walls.',
  'quality-96-5-on': 'The red-and-green room is lit by global illumination at the high quality: the walls, floor, box '
    + 'and sphere away from the sunlit patch show soft bounce light tinted by the coloured walls.',
  'quality-off': 'The same red-and-green room without global illumination: no coloured bounce tint appears anywhere; '
    + 'the box, the sphere and the floor beside the red and green walls stay neutral, lit away from the sunlit patch '
    + 'only by a flat, uncoloured ambient fill.',
  'forward-content-on': 'The skinned figure, the squashed shape, the transparent pane and the fur ball beside the red '
    + 'wall all show a reddish bounce tint and soft indirect light on their sides facing the wall, like the static '
    + 'surfaces around them.',
  'forward-content-off': 'The skinned figure, the squashed shape, the transparent pane and the fur ball beside the red '
    + 'wall show no red bounce tint; their shadowed sides stay dark.',
  'moved-instance-on': 'The bright orange block stands in its new place near the right wall, and the floor and wall '
    + 'around it there glow orange from its bounce light; no orange glow remains at its old place in the far left corner.',
  'moved-instance-off': 'The bright orange block stands in its new place near the right wall with no orange light '
    + 'spilling onto the floor or wall around it.',
  'cornell-box-on': 'Global illumination is visibly at work in the Cornell box, compared with the same box with GI off: the '
    + 'ceiling, dark without GI, is lit by bounce light; the red and green walls bleed their colour onto the '
    + 'white blocks and the floor (a red tint on faces near the red left wall, a green tint on faces near the '
    + 'green right wall); the front faces of the two blocks, which the ceiling light does not reach directly, '
    + 'are lit by bounce light instead of staying black; no bright blotch or ring around the ceiling light and '
    + 'no light leaking through the walls or ceiling. The reference image is a path-traced Cornell box given '
    + 'only as context for the scene: hard shadow edges, a different overall brightness, a different shape of '
    + 'the lit area under the light, visible voxel-scale smoothness in the bounce, or any other difference from '
    + 'the reference are not faults here.',
  'cornell-box-off': 'The same Cornell box with global illumination off and the same exposure: only surfaces the ceiling '
    + 'light reaches directly are lit; the ceiling and the surfaces facing away from the light stay dark; there is no red '
    + 'or green colour bleeding on the white surfaces, and the blocks cast no shadows because RAA local lights are '
    + 'unshadowed without GI.',
  'cornell-box-linear-on': 'Global illumination on, shown with linear tone mapping at a quarter of the usual exposure '
    + '(a darker image with flat contrast): the floor under the ceiling light is the brightest surface, the back wall is '
    + 'lit and darkens towards its top corners, the ceiling is lit only by bounce light, and the red and green walls tint '
    + 'the nearby white surfaces; nothing is clipped to pure white except the light fixture itself.',
  'cornell-box-linear-off': 'Global illumination off, shown with linear tone mapping at a quarter of the usual exposure: '
    + 'only surfaces the ceiling light reaches directly are lit; the floor under the light is the brightest surface and '
    + 'the back wall darkens towards its top corners; the ceiling stays black and there is no red or green colour '
    + 'bleeding; nothing is clipped to pure white except the light fixture itself.',
};
const DESCRIPTIONS = {
  'color-bleed-on': 'Global illumination on: the white box and floor take bounce light from the red and green walls.',
  'color-bleed-off': 'The same red and green room with global illumination disabled: no coloured bounce light.',
  'interior-bounce-on': 'Global illumination on: the shadowed interior is lit by bounce light from the sunlit patch.',
  'interior-bounce-off': 'The same interior with global illumination disabled: only the flat ambient term lights it.',
  'emissive-panel-on': 'Global illumination on: the emissive panel spills warm light onto the nearby surfaces.',
  'emissive-panel-off': 'The same night room with global illumination disabled: the panel glows and lights nothing.',
  'local-light-shadow-on': 'Global illumination on: a point and a spot light behind a 0.6 m wall are stopped by it.',
  'local-light-shadow-off': 'The same lights with global illumination disabled: they leak through the wall.',
  'sky-overhang-on': 'Global illumination on under the physical day sky: the overhang region gains soft bounce light.',
  'sky-overhang-off': 'The same sky and overhang with global illumination disabled: flat shadow under the overhang.',
  'glossy-reflection-on': 'Global illumination on under the day sky: the glossy floor reflects the orange object through the volume.',
  'glossy-reflection-off': 'The same glossy floor with global illumination disabled: it reflects the sky environment only.',
  'quality-64-3-on': 'Global illumination on at the low quality: 64 resolution, 3 levels and 6 probe rays.',
  'quality-96-5-on': 'Global illumination on at the high quality: 96 resolution, 5 levels and 8 probe rays.',
  'quality-off': 'The same red-and-green room with global illumination disabled.',
  'forward-content-on': 'Global illumination on: the skinned, deformed, transparent and fur instances beside the red wall receive bounce light.',
  'forward-content-off': 'The same forward content with global illumination disabled: no bounce light reaches any of it.',
  'moved-instance-on': 'Global illumination on after the orange block moves to a new place near the right wall.',
  'moved-instance-off': 'The same moved block with global illumination disabled: no orange light spills from it.',
  'cornell-box-on': 'Global illumination on: the Cornell box is lit by bounce light from its red and green walls and its ceiling light.',
  'cornell-box-off': 'The same Cornell box with global illumination disabled: only the direct light of the ceiling light.',
  'cornell-box-linear-on': 'Global illumination on, linear tone mapping at a quarter exposure: the Cornell box as scene-linear values.',
  'cornell-box-linear-off': 'The same Cornell box with global illumination disabled, linear tone mapping at a quarter exposure.',
};

const nodeAt = (engine, position = [0, 0, 0]) => engine.addNode(null, position, [0, 0, 0, 1], [1, 1, 1]);
const meshWith = (engine, desc, material) => engine.addMesh({
  ...desc,
  subMeshes: [{ first: 0, count: desc.indices.length, material }],
});
const matte = (engine, color) => engine.addMaterial({ color, roughness: 0.85 });
const pairOf = row => (row.measured ? `gi-lighting-${row.arrangement}` : null);

// Joins MeshDesc parts into one MeshDesc, so that one instance group holds all of them.
function joinMeshes(parts) {
  const total = key => parts.reduce((sum, part) => sum + part[key].length, 0);
  const out = {
    positions: new Float32Array(total('positions')),
    normals: new Float32Array(total('normals')),
    indices: new Uint32Array(total('indices')),
  };
  let vertex = 0, index = 0;
  for (const part of parts) {
    out.positions.set(part.positions, vertex * 3);
    out.normals.set(part.normals, vertex * 3);
    out.indices.set(part.indices.map(i => i + vertex), index);
    vertex += part.positions.length / 3;
    index += part.indices.length;
  }
  return out;
}

// Clear weather, the room's sun and ambient; the night variants have no sun and a dim ambient.
function environment(engine, night) {
  engine.setSky({ preset: night ? 'night' : 'day' });
  engine.setWeather({ preset: 'clear' });
  engine.setLight({ direction: room.sun.direction, color: room.sun.color, intensity: night ? 0 : room.sun.intensity });
  engine.setAmbient({ color: [0.6, 0.65, 0.75], intensity: night ? 0.02 : 0.05 });
}

/**
 * The shared room of volume.js: sky, sun and ambient, the walls (sub-meshes floor, ceiling, back, left,
 * right with materials white, white, white, red, green) on roomShell(room.room) and the 'room' instance.
 * Returns the white material for the objects placed in the room.
 */
function roomScene(engine, state, { night = false, white: whiteColor = room.colors.white } = {}) {
  environment(engine, night);
  const red = matte(engine, room.colors.red), green = matte(engine, room.colors.green);
  const white = matte(engine, whiteColor);
  const shell = roomShell(room.room);
  const wallMaterials = [white, white, white, red, green];
  const roomMesh = engine.addMesh({
    ...shell,
    subMeshes: shell.subMeshes.map((part, index) => ({ ...part, material: wallMaterials[index] })),
  });
  state.instances.push({ name: 'room', instance: engine.addMeshInstance(nodeAt(engine), roomMesh) });
  return white;
}

// The room with a white box and a white sphere: the colour-bleed and interior-bounce arrangements.
function roomWithObjects(engine, variant, state) {
  const white = roomScene(engine, state);
  state.instances.push(
    { name: 'box', instance: engine.addMeshInstance(nodeAt(engine), meshWith(engine, box(room.box.center, room.box.half), white)) },
    { name: 'sphere', instance: engine.addMeshInstance(nodeAt(engine, room.sphere.center), meshWith(engine, sphere(room.sphere.radius, room.sphere.segments), white)) },
  );
}

// The forward-content sun (travel direction). With the shared room sun the red wall was lit only by bounce
// light, much dimmer than the sky through the open front and the white floor, so the light the objects beside it
// received was neutral grey. This direction passes the 1.2 m ceiling opening (x -0.6..0.6, z -4.6..-3.4, 0.4 m
// thick) and lands on the red wall's inner face (x -3) at y 0.4..1.0 m, z -5.7..-4.4 m, right behind the
// skinned figure and between the squashed shape and the pane.
const FORWARD_CONTENT_SUN = [-1.2, -1, -0.43];

const BLOCK_START = [-2.0, 0.4, -7.0];
const BLOCK_TARGET = [2.0, 0.4, -3.2];
const BEND = Math.cos(Math.PI / 6);
// Joint 1 is T(p) Rz(30 degrees) T(-p) with p = (0, 0.7, 0), column-major; joint 0 is the identity.
const SKIN_POSE = new Float32Array([
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
  BEND, 0.5, 0, 0, -0.5, BEND, 0, 0, 0, 0, 1, 0, 0.35, 0.7 * (1 - BEND), 0, 1,
]);

// Two boxes merged into one figure: joint 0 below y 0.7, joint 1 above it, weight 1 on the joint.
function figureMesh(engine, white) {
  const body = joinMeshes([box([0, 0.35, 0], [0.15, 0.35, 0.15]), box([0, 1.05, 0], [0.12, 0.35, 0.12])]);
  const count = body.positions.length / 3;
  const joints = new Uint16Array(count * 4);
  const weights = new Float32Array(count * 4);
  for (let v = 0; v < count; v++) {
    joints[v * 4] = body.positions[v * 3 + 1] < 0.7 ? 0 : 1;
    weights[v * 4] = 1;
  }
  return meshWith(engine, { ...body, joints, weights }, white);
}

// Cornell box (metres): the canonical surfaces as the directive gives them, each quad as four corners in order.
const CORNELL = {
  floor: [[5.528, 0, 0], [0, 0, 0], [0, 0, 5.592], [5.496, 0, 5.592]],
  ceiling: [[5.56, 5.488, 0], [5.56, 5.488, 5.592], [0, 5.488, 5.592], [0, 5.488, 0]],
  back: [[5.496, 0, 5.592], [0, 0, 5.592], [0, 5.488, 5.592], [5.56, 5.488, 5.592]],
  red: [[5.528, 0, 0], [5.496, 0, 5.592], [5.56, 5.488, 5.592], [5.56, 5.488, 0]],
  green: [[0, 0, 5.592], [0, 0, 0], [0, 5.488, 0], [0, 5.488, 5.592]],
  shortTop: [[1.30, 1.65, 0.65], [0.82, 1.65, 2.25], [2.40, 1.65, 2.72], [2.90, 1.65, 1.14]],
  tallTop: [[4.23, 3.30, 2.47], [2.65, 3.30, 2.96], [3.14, 3.30, 4.56], [4.72, 3.30, 4.06]],
};
const CORNELL_CENTRE = [2.78, 2.744, 2.796];
// Linear albedos (the engine takes sRGB colours, so they are converted by srgbOf below).
const CORNELL_ALBEDO = { white: [0.886, 0.699, 0.666], red: [0.570, 0.043, 0.044], green: [0.105, 0.378, 0.076] };
// Linear light colour and radiance ratio 18.4 : 11.0 : 2.75; the intensity is radiance x area (1.30 x 1.05 m).
const CORNELL_LIGHT_COLOUR = [1, 0.598, 0.149];
const CORNELL_LIGHT_INTENSITY = 18.4 * 1.365;
export const CORNELL_LIGHT_X = 2.78, CORNELL_LIGHT_Z = 2.795;
// The rectangle basis of a downward light puts size[0] along -z and size[1] along x, so [1.05, 1.30]
// gives the specified 1.30 m along x and 1.05 m along z.
export const CORNELL_LIGHT_SIZE = [1.05, 1.30];

const srgbOf = v => (v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055);
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

// One quad as two triangles. The normal follows the winding of the first corners, and both are flipped
// when that normal points away from `towards`, so every face points into the room or away from a block.
function quad(corners, towards) {
  const [a, b, c, d] = corners;
  const mid = [0, 1, 2].map(i => (a[i] + b[i] + c[i] + d[i]) / 4);
  let n = cross3(sub3(b, a), sub3(c, a));
  const length = Math.hypot(...n);
  n = n.map(v => v / length);
  const flip = dot3(n, sub3(towards, mid)) < 0;
  if (flip) n = n.map(v => -v);
  return {
    positions: new Float32Array([...a, ...b, ...c, ...d]),
    normals: new Float32Array([...n, ...n, ...n, ...n]),
    indices: new Uint32Array(flip ? [0, 2, 1, 0, 3, 2] : [0, 1, 2, 0, 2, 3]),
  };
}

// A block: its top quad and four sides that drop each top edge to y 0 (the bottom is on the floor).
function block(top, height) {
  const cx = top.reduce((sum, p) => sum + p[0], 0) / 4;
  const cz = top.reduce((sum, p) => sum + p[2], 0) / 4;
  const raised = top.map(p => [p[0], height, p[2]]);
  const parts = [quad(raised, [cx, height + 1, cz])];
  for (let i = 0; i < 4; i++) {
    const p = raised[i], q = raised[(i + 1) % 4];
    const midX = (p[0] + q[0]) / 2, midZ = (p[2] + q[2]) / 2;
    parts.push(quad([p, q, [q[0], 0, q[2]], [p[0], 0, p[2]]], [2 * midX - cx, height / 2, 2 * midZ - cz]));
  }
  return joinMeshes(parts);
}

// The walls' backing: a 0.8 m slab outside the floor, ceiling, back, red and green walls. The front stays open,
// so the slabs' front faces are unlit and read black. Every slab has the colour of the surface it backs: a
// slab's inner face is coplanar with that surface (the green wall at x 0 entirely, the red wall along its top
// edge), so a white slab behind a coloured wall won the depth test there and shared its voxels, which showed
// the green wall cream and removed its green bounce.
const CORNELL_SLABS = {
  white: [
    [[2.78, -0.4, 2.796], [2.78, 0.4, 2.796]],
    [[2.78, 5.888, 2.796], [2.78, 0.4, 2.796]],
    [[2.78, 2.744, 5.992], [2.78, 2.744, 0.4]],
  ],
  red: [[[5.96, 2.744, 2.796], [0.4, 2.744, 2.796]]],
  green: [[[-0.4, 2.744, 2.796], [0.4, 2.744, 2.796]]],
};
const slabsOf = colour => CORNELL_SLABS[colour].map(([center, half]) => box(center, half));

// The Cornell box: the sky, sun and ambient are black, the only light is the ceiling area light, and an unlit
// fixture of the light's rectangle at CORNELL_FIXTURE_Y shows it. The fixture is kept out of the GI volume.
function cornellBox(engine, variant, state) {
  engine.setSky({ model: 'legacy', zenith: [0, 0, 0], horizon: [0, 0, 0], ground: [0, 0, 0], intensity: 0, sunDiscIntensity: 0 });
  engine.setWeather({ preset: 'clear' });
  engine.setEnvironmentLighting({ diffuse: 0, specular: 0 });
  engine.setAmbient({ color: [0, 0, 0], intensity: 0 });
  engine.setLight({ direction: [0, -1, 0], color: [1, 1, 1], intensity: 0 });
  engine.setToneMapping({ exposureEv: Math.log2(CORNELL_EXPOSURE) });
  engine.setGlobalIllumination({ localLightShadows: true, environmentDiffuse: 0, intensity: 1 });
  const material = albedo => engine.addMaterial({ color: albedo.map(srgbOf), roughness: 1, metallic: 0 });
  const white = material(CORNELL_ALBEDO.white);
  const red = material(CORNELL_ALBEDO.red);
  const green = material(CORNELL_ALBEDO.green);
  const whiteParts = [
    quad(CORNELL.floor, CORNELL_CENTRE), quad(CORNELL.ceiling, CORNELL_CENTRE), quad(CORNELL.back, CORNELL_CENTRE),
    block(CORNELL.shortTop, 1.65), block(CORNELL.tallTop, 3.30),
    ...slabsOf('white'),
  ];
  engine.addLight({
    kind: 'area', position: [CORNELL_LIGHT_X, CORNELL_LIGHT_Y, CORNELL_LIGHT_Z], direction: [0, -1, 0],
    color: CORNELL_LIGHT_COLOUR.map(srgbOf), intensity: CORNELL_LIGHT_INTENSITY, range: 30, size: CORNELL_LIGHT_SIZE,
  });
  // The fixture is the light's rectangle: x spans size[1], z spans size[0] (see CORNELL_LIGHT_SIZE).
  const x0 = CORNELL_LIGHT_X - CORNELL_LIGHT_SIZE[1] / 2, x1 = CORNELL_LIGHT_X + CORNELL_LIGHT_SIZE[1] / 2;
  const z0 = CORNELL_LIGHT_Z - CORNELL_LIGHT_SIZE[0] / 2, z1 = CORNELL_LIGHT_Z + CORNELL_LIGHT_SIZE[0] / 2;
  const fixtureMaterial = engine.addMaterial({ color: [0, 0, 0], emissive: [4 / CORNELL_EXPOSURE, 4 / CORNELL_EXPOSURE, 4 / CORNELL_EXPOSURE], model: 'flat' });
  const fixtureQuad = quad([[x0, CORNELL_FIXTURE_Y, z0], [x1, CORNELL_FIXTURE_Y, z0], [x1, CORNELL_FIXTURE_Y, z1], [x0, CORNELL_FIXTURE_Y, z1]],
    [CORNELL_LIGHT_X, 0, CORNELL_LIGHT_Z]);
  const fixture = engine.addMeshInstance(nodeAt(engine), meshWith(engine, fixtureQuad, fixtureMaterial));
  engine.setInstanceGlobalIllumination(fixture, false);
  state.instances.push(
    { name: 'cornell-white', instance: engine.addMeshInstance(nodeAt(engine), meshWith(engine, joinMeshes(whiteParts), white)) },
    { name: 'cornell-red', instance: engine.addMeshInstance(nodeAt(engine), meshWith(engine, quad(CORNELL.red, CORNELL_CENTRE), red)) },
    { name: 'cornell-green', instance: engine.addMeshInstance(nodeAt(engine), meshWith(engine, quad(CORNELL.green, CORNELL_CENTRE), green)) },
    { name: 'fixture', instance: fixture },
    { name: 'cornell-red-backing', instance: engine.addMeshInstance(nodeAt(engine), meshWith(engine, joinMeshes(slabsOf('red')), red)) },
    { name: 'cornell-green-backing', instance: engine.addMeshInstance(nodeAt(engine), meshWith(engine, joinMeshes(slabsOf('green')), green)) },
  );
}

// The linear-HDR probe rows: the same Cornell box, then linear tone mapping without dither at a quarter of the
// Cornell exposure, so the capture stores sRGB-encoded scene-linear values that the ratio script can decode.
function cornellLinear(engine, variant, state) {
  cornellBox(engine, variant, state);
  engine.setToneMapping({
    mapper: 'linear', dither: false, exposureEv: Math.log2(CORNELL_EXPOSURE * CORNELL_LINEAR_EXPOSURE_SCALE),
  });
}

// Each arrangement adds its content and pushes { name, instance } entries into state.instances.
const ARRANGEMENTS = {
  'color-bleed': roomWithObjects,
  'interior-bounce': roomWithObjects,
  'emissive-panel': (engine, variant, state) => {
    const white = roomScene(engine, state, { night: true });
    const glow = engine.addSurfaceMaterial({ color: [1, 0.92, 0.8], emissive: [8, 7.2, 6], roughness: 0.9 });
    state.instances.push(
      { name: 'panel', instance: engine.addMeshInstance(nodeAt(engine, [0, 0.9, -7.4]), meshWith(engine, panel(1.6, 1.0), glow)) },
      { name: 'box', instance: engine.addMeshInstance(nodeAt(engine), meshWith(engine, box([1.2, 0.5, -6.2], [0.5, 0.5, 0.5]), white)) },
      { name: 'sphere', instance: engine.addMeshInstance(nodeAt(engine, [-1.3, 0.6, -6.4]), meshWith(engine, sphere(0.6, 32), white)) },
    );
  },
  'local-light-shadow': (engine, variant, state) => {
    environment(engine, true);
    const floor = engine.addMaterial({ color: [0.7, 0.7, 0.68], roughness: 0.9 });
    const white = matte(engine, room.colors.white);
    state.instances.push(
      { name: 'floor', instance: engine.addMeshInstance(nodeAt(engine), meshWith(engine, plane(14, 14), floor)) },
      { name: 'wall', instance: engine.addMeshInstance(nodeAt(engine), meshWith(engine, box([0, 1.25, -4], [3, 1.25, 0.3]), white)) },
    );
    engine.addLight({ kind: 'point', position: [-0.9, 1.2, -5.6], color: [1, 0.85, 0.6], intensity: 20, range: 8 });
    engine.addLight({ kind: 'spot', position: [0.9, 2.4, -5.8], direction: [0, -0.55, 1], color: [0.7, 0.85, 1], intensity: 30, range: 8, angleDeg: 35 });
  },
  'sky-overhang': (engine, variant, state) => {
    engine.setSky({ model: 'physical', preset: 'day' });
    engine.setLight({ direction: [0.45, -1, 0.25], color: [1, 0.96, 0.9], intensity: 3 });
    const ground = matte(engine, [0.42, 0.42, 0.4]);
    const white = matte(engine, room.colors.white);
    const pillars = [[1.9, 1.9], [-1.9, 1.9], [1.9, -1.9], [-1.9, -1.9]].map(([x, z]) => box([x, 1.5, z], [0.15, 1.5, 0.15]));
    const overhang = joinMeshes([box([0, 3.15, 0], [2.2, 0.15, 2.2]), ...pillars]);
    state.instances.push(
      { name: 'ground', instance: engine.addMeshInstance(nodeAt(engine), meshWith(engine, plane(60, 60), ground)) },
      { name: 'overhang', instance: engine.addMeshInstance(nodeAt(engine), meshWith(engine, overhang, white)) },
      { name: 'box', instance: engine.addMeshInstance(nodeAt(engine), meshWith(engine, box([0.3, 0.4, 0.2], [0.4, 0.4, 0.4]), white)) },
    );
  },
  'glossy-reflection': (engine, variant, state) => {
    engine.setSky({ preset: 'day' });
    engine.setWeather({ preset: 'clear' });
    engine.setLight({ direction: [-0.3, -1, -0.5], color: [1, 0.96, 0.9], intensity: 2 });
    const floor = engine.addSurfaceMaterial({ color: [0.25, 0.25, 0.27], roughness: 0.1, metallic: 0.8 });
    const orange = engine.addSurfaceMaterial({ color: [1, 0.5, 0.1], emissive: [12, 6, 1.5], roughness: 0.6 });
    engine.setSsr({ intensity: 1, maxRoughness: 0.3 });
    state.instances.push(
      { name: 'floor', instance: engine.addMeshInstance(nodeAt(engine), meshWith(engine, plane(40, 40), floor)) },
      { name: 'orange', instance: engine.addMeshInstance(nodeAt(engine, [0, 6, -8]), meshWith(engine, sphere(1, 32), orange)) },
    );
  },
  quality: roomWithObjects,
  'forward-content': (engine, variant, state) => {
    const white = roomScene(engine, state, { white: [0.85, 0.85, 0.82] });
    // The sun of this arrangement shines through the ceiling opening onto the red wall behind the objects
    // (FORWARD_CONTENT_SUN), so the wall has red light to bounce onto them.
    engine.setLight({ direction: FORWARD_CONTENT_SUN, color: room.sun.color, intensity: room.sun.intensity });
    const skinned = engine.addMeshInstance(nodeAt(engine, [-2.4, 0, -4.6]), figureMesh(engine, white));
    engine.setSkinPose(skinned, SKIN_POSE);
    const deformed = engine.addMeshInstance(nodeAt(engine, [-2.4, 0.35, -3.6]), meshWith(engine, sphere(0.35, 24), white));
    engine.setSquash(deformed, { amount: 0.3 });
    engine.setJiggle(deformed, { height: 0.7, amount: 0.5 });
    const pane = engine.addSurfaceMaterial({ color: [0.85, 0.9, 0.95], roughness: 0.2, blend: 'transparent', opacity: 0.45, doubleSided: true });
    const transparent = engine.addMeshInstance(
      engine.addNode(null, [-2.3, 0.7, -5.6], [0, 0.7071068, 0, 0.7071068], [1, 1, 1]),
      meshWith(engine, panel(0.9, 1.2), pane),
    );
    const furMaterial = engine.addSurfaceMaterial({ color: [0.9, 0.88, 0.84], roughness: 0.9, furLength: 0.05, furShells: 16 });
    const fur = engine.addMeshInstance(nodeAt(engine, [-2.4, 0.35, -6.6]), meshWith(engine, sphere(0.35, 24), furMaterial));
    state.instances.push(
      { name: 'skinned', instance: skinned },
      { name: 'deformed', instance: deformed },
      { name: 'transparent', instance: transparent },
      { name: 'fur', instance: fur },
    );
  },
  'cornell-box': cornellBox,
  'cornell-box-linear': cornellLinear,
  'moved-instance': (engine, variant, state) => {
    roomScene(engine, state);
    const block = engine.addNode(null, BLOCK_START, [0, 0, 0, 1], [1, 1, 1]);
    const orange = engine.addSurfaceMaterial({ color: [1, 0.45, 0.05], emissive: [3, 1.3, 0.15], roughness: 0.7 });
    state.instances.push({ name: 'block', instance: engine.addMeshInstance(block, meshWith(engine, box([0, 0, 0], [0.4, 0.4, 0.4]), orange)) });
    state.blockNode = block;
  },
};

// One variant per row. An on variant names its off partner (offMode); an off variant shares the
// arrangement, quality and selection of its on partner and disables GI.
const variants = ROWS.map(row => {
  const on = row.mode.endsWith('-on');
  return {
    id: row.mode,
    description: DESCRIPTIONS[row.mode],
    references: [],
    judge: true,
    arrangement: row.arrangement,
    gi: { enabled: on, debugView: 'none' },
    quality: QUALITY[row.mode] ?? { ...DEFAULT_QUALITY },
    giExpected: on ? 'active' : 'disabled',
    selection: SELECTIONS[row.arrangement],
    ...(on ? { offMode: `${row.arrangement}-off` } : {}),
    ...(row.measured ? { measurement: { pair: pairOf(row), state: on ? 'on' : 'off', preset: `lighting-${row.arrangement}`, features: { globalIllumination: on } } } : {}),
  };
});
export const inventory = ROWS.map(row => ({
  id: `lighting-${row.mode}`,
  criterionName: row.criterionName,
  scenario: 'lighting',
  mode: row.mode,
  filename: row.filename,
  expectedVisibleBehavior: BEHAVIOURS[row.mode],
  requiredFacts: row.facts ?? [...(row.mode.endsWith('-on') ? ON_FACTS : OFF_FACTS), ...row.extraFacts],
  pair: pairOf(row),
  heldConstant: [...HELD[row.arrangement], ...(row.extraHeld ?? [])],
  producer: row.producer,
  ...(row.judgeReference === undefined ? {} : { judgeReference: row.judgeReference, judgeAspect: row.judgeAspect }),
}));

/**
 * `build(engine, lib, variant)` sets the variant's quality, adds its arrangement, sets the variant camera and
 * applies `variant.gi`, then returns the state { instances, moved: false, blockNode: null }.
 */
export async function build(engine, lib, variant) {
  const state = { instances: [], moved: false, blockNode: null };
  engine.setGiQuality(variant.quality);
  ARRANGEMENTS[variant.arrangement](engine, variant, state);
  engine.setCamera(variant.selection.camera);
  engine.setGlobalIllumination(variant.gi);
  return state;
}

const check = (name, passed, detail) => ({ name, passed: passed === true, detail });

/**
 * `observe(engine, variant, state)` reports the seven assertions of contract 'gi-lighting-rows', in factNames
 * order, for any variant. Today it reads giStats(), giQuality(), globalIllumination(), lightStats(), passes(),
 * instanceGlobalIllumination() and the state's instances and moved flag. `gi-active` and `gi-off` read the volume
 * as volume.js does; `local-lights` needs two active local lights; `ssr-enabled` needs the 'ssr' pass enabled and
 * reflections on; `quality-applied` compares giQuality() with the variant's quality; `forward-content` needs the
 * skinned, deformed, transparent and fur instances, none voxelized; `block-moved` needs state.moved. Each detail
 * names every value read, and assertions that do not apply to the variant are still reported.
 * Handoff: gi page: lighting.test.mjs and p4-gi-lighting commands calls
 * `observe(engine: object, variant: object, state: { instances: Array<{ name: string, instance: number }>, moved: boolean, blockNode: number | null }) -> Promise<{ instances: Array<{ name: string, instance: number }>, assertions: Array<{ name: string, passed: boolean, detail: string }>, edits: [] }>`
 * exported by examples/global-illumination/scenes/lighting.js:309 observe; the test attaches it with
 * `import * as lighting from './scenes/lighting.js'` and calls observe on a stub engine; state/keys: the namespace
 * keys `inventory`, `factNames`, `observe`, `default.variants`, `default.build` and `default.ready`, the state keys
 * `instances`, `moved` and `blockNode`, and the assertion names of `factNames`.
 */
export async function observe(engine, variant, state) {
  const stats = engine.giStats();
  const settings = engine.globalIllumination();
  const quality = engine.giQuality();
  const light = engine.lightStats();
  const ssr = engine.passes().find(pass => pass.name === 'ssr');
  const key = `${stats.quality.resolution}/${stats.quality.levels}`;
  const expectedBytes = VOLUME_BYTES[key];
  const wanted = variant.quality;
  const forward = ['skinned', 'deformed', 'transparent', 'fur'].map(name => {
    const entry = state.instances.find(item => item.name === name);
    return { name, present: entry !== undefined, voxelized: entry ? engine.instanceGlobalIllumination(entry.instance).voxelized : null };
  });
  const assertions = [
    check('gi-active',
      stats.active === true && stats.pendingRebuild === false && stats.framesSinceRebuild >= 240
        && stats.allocatedBytes >= expectedBytes && stats.voxelizedForwardInstances >= 1,
      `active ${stats.active}, pendingRebuild ${stats.pendingRebuild}, framesSinceRebuild ${stats.framesSinceRebuild}, `
        + `quality ${key}, allocatedBytes ${stats.allocatedBytes} (expected at least ${expectedBytes}), `
        + `voxelizedForwardInstances ${stats.voxelizedForwardInstances}`),
    check('gi-off',
      stats.active === false && stats.inactiveReason === 'disabled' && stats.allocatedBytes === 0,
      `active ${stats.active}, inactiveReason ${stats.inactiveReason}, allocatedBytes ${stats.allocatedBytes}`),
    check('local-lights', light.activeLights === 2, `activeLights ${light.activeLights}, expected 2`),
    check('ssr-enabled', ssr?.enabled === true && settings.reflections === true,
      `ssr pass ${ssr ? `enabled ${ssr.enabled}` : 'absent'}, reflections ${settings.reflections}`),
    check('quality-applied',
      quality.resolution === wanted.resolution && quality.levels === wanted.levels && quality.probeRays === wanted.probeRays,
      `quality ${quality.resolution}/${quality.levels}/${quality.probeRays}, `
        + `expected ${wanted.resolution}/${wanted.levels}/${wanted.probeRays}`),
    check('forward-content', forward.every(item => item.present && item.voxelized === false),
      forward.map(item => `${item.name} ${item.present ? `voxelized ${item.voxelized}` : 'absent'}`).join(', ')),
    check('block-moved', state.moved === true, `moved ${state.moved}`),
  ];
  return { instances: state.instances, assertions, edits: [] };
}

/**
 * Handoff: gi page: lighting.test.mjs and p4-gi-lighting commands calls `ready` as the default export's ready
 * (this definition, examples/global-illumination/scenes/lighting.js:493 ready), and attaches `observe`
 * as cited by examples/global-illumination/scenes/lighting.js:444 observe, the Handoff of gi page:
 * scenes/lighting.js rows 1-10 (bleed, interior, emissive, local lights, sky overhang). Today `ready` returns
 * { ready: true, stage: 'lighting' } for every arrangement except moved-instance. For moved-instance, while
 * state.moved is false, it calls engine.setTransform(state.blockNode, [2.0, 0.4, -3.2], [0, 0, 0, 1], [1, 1, 1])
 * once and sets state.moved when variant.giExpected is 'disabled', or when giStats() reads active, not
 * pendingRebuild and framesSinceRebuild >= 1; it then returns { ready: state.moved, stage: 'lighting' }.
 * State/keys: state.instances (names skinned, deformed, transparent, fur, block), state.moved and state.blockNode;
 * the namespace keys inventory, factNames, default.variants and default.ready.
 */
function ready(engine, state, variant) {
  if (variant.arrangement !== 'moved-instance') return { ready: true, stage: 'lighting' };
  if (!state.moved) {
    const stats = engine.giStats();
    const built = stats.active === true && stats.pendingRebuild === false && stats.framesSinceRebuild >= 1;
    if (variant.giExpected === 'disabled' || built) {
      engine.setTransform(state.blockNode, BLOCK_TARGET, [0, 0, 0, 1], [1, 1, 1]);
      state.moved = true;
    }
  }
  return { ready: state.moved, stage: 'lighting' };
}

export default {
  id: 'lighting',
  milestone: 'lighting',
  title: 'GI lighting',
  variants,
  build,
  ready,
};
