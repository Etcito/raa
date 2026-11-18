// The sharpen-detail scene (S49, post page, stage sharpen): thin stripes, seven thin poles, a small
// checker patch and three distant cubes, seen by one detail camera. Four variants: the pair 'sharpen'
// at render scale 1 and the pair 'upscale-067-sharpen' at 0.67, each with sharpening on and off; TAA is on
// in every row. Importing this module has no side effects. Geometry comes from '../lib/content.js': box,
// plane, tiledGround and facadeCourses (their doc comments are at examples/post/lib/content.js:60-170).
// The runtime applies variant.settings after build (examples/post/lib/runtime.js:126-130), so build sets
// neither sharpening, the render scale nor any pass.
import { box, facadeCourses, plane, tiledGround } from '../lib/content.js';

// sRGB base colours; every material has roughness 0.85.
const MATERIALS = {
  ground: [0.5, 0.5, 0.5], white: [0.9, 0.9, 0.9], black: [0.05, 0.05, 0.05],
  red: [0.8, 0.1, 0.08], yellow: [0.9, 0.8, 0.1], blue: [0.1, 0.2, 0.8],
};
// The render scale of the upscale rows (POST_UPSCALE_SCALE, S49).
const UPSCALE_SCALE = 0.67;

// One detail camera for all four rows. Every subject box equals its geometry and lies in the frustum
// (assertSubjectsInView, checked in sharpen.test.mjs).
const SELECTION = {
  camera: { eye: [0, 1.6, 5], target: [0, 1.1, -6], fovYDeg: 50, near: 0.1, far: 300 },
  subjects: [
    { name: 'stripe-panel', min: [-2.5, 0.3, -4], max: [-2.48, 2.3, -1] },
    { name: 'thin-poles', min: [0.285, 0, -3.015], max: [2.115, 3, -2.985] },
    { name: 'checker-patch', min: [-1, 0, -2.5], max: [0.28, 0.01, -1.22] },
    { name: 'distant-cubes', min: [-3.5, 0, -60.5], max: [2.5, 1, -29.5] },
  ],
};

// One variant per pair and state. The id is the mode: '<pair>-<state>', which the -on / -off suffix checks.
function variantOf(pair, state) {
  const on = state === 'on';
  const upscaled = pair !== 'sharpen';
  return {
    id: `${pair}-${state}`,
    selection: SELECTION,
    settings: {
      passes: { taa: true },
      ...(upscaled ? { renderScale: UPSCALE_SCALE } : {}),
      sharpening: { enabled: on },
    },
    measurement: {
      pair,
      state,
      preset: upscaled ? 'sharpen-detail-scale-067' : 'sharpen-detail-scale-1',
      features: { sharpening: on, taa: true, renderScale: upscaled ? UPSCALE_SCALE : 1 },
    },
  };
}
const variants = ['sharpen', 'upscale-067-sharpen'].flatMap((pair) => ['on', 'off'].map((state) => variantOf(pair, state)));

const REQUIRED_FACTS = [
  'runtime:passes', 'runtime:viewSizes', 'runtime:ready', 'fact:render-scale', 'fact:taa-enabled', 'fact:sharpen-enabled',
  'assertion:render-scale', 'assertion:taa-enabled', 'assertion:sharpen-enabled',
];
// A row takes its pair, held-constant render scale and producer from its variant.
const row = (mode, filename, expectedVisibleBehavior) => {
  const v = variants.find((x) => x.id === mode);
  return {
    id: mode,
    criterionName: `P5-Q3 sharpen ${mode}`,
    scenario: 'sharpen-detail',
    mode,
    filename,
    expectedVisibleBehavior,
    requiredFacts: REQUIRED_FACTS,
    pair: v.measurement.pair,
    heldConstant: ['sky, sun and ambient lighting', 'stripe panel, thin poles, checker patch and distant cubes',
      'TAA on', `render scale ${v.settings.renderScale ?? 1}`, 'detail camera'],
    producer: v.settings.sharpening.enabled ? 'raa-sharpen' : 'post-sharpen-scenes',
  };
};

export const inventory = [
  row('sharpen-on', 'sharpen-on-fine-detail-crisp.png', 'At render scale 1 with TAA on: thin lines, fine stripes and the edges of distant objects look crisp; high-contrast edges show no dark or bright outline.'),
  row('sharpen-off', 'sharpen-off-fine-detail-soft.png', 'The same view with sharpening off: the same fine detail looks slightly softer.'),
  row('upscale-067-sharpen-on', 'upscale-067-sharpen-on-fine-detail-crisp.png', 'At render scale 0.67 with TAA on, shown at the full display size: fine detail looks crisp for an upscaled image; high-contrast edges show no halo.'),
  row('upscale-067-sharpen-off', 'upscale-067-sharpen-off-fine-detail-blurred.png', 'The same upscaled view with sharpening off: fine detail looks visibly blurred.'),
];

export const factNames = ['render-scale', 'taa-enabled', 'sharpen-enabled'];

// Adds one mesh whose sub-meshes take `materials` (one per sub-mesh, or one for all of them) and one
// instance of it at `at`; returns the instance handle. As addPart in examples/post/scenes/exposure.js.
function addPart(engine, desc, materials, at) {
  const list = Array.isArray(materials) ? materials : [materials];
  const parts = desc.subMeshes ?? [{ first: 0, count: desc.indices.length }];
  const mesh = engine.addMesh({ ...desc, subMeshes: parts.map((part, i) => ({ ...part, material: list[i] ?? list[0] })) });
  return engine.addMeshInstance(engine.addNode(null, at, [0, 0, 0, 1], [1, 1, 1]), mesh);
}

/**
 * `build(engine, lib, variant)` sets the day sky, clear weather, the light, the ambient, and no bloom,
 * adds the stripe panel, the poles, the checker patch, the ground and the distant cubes, and sets the
 * variant camera. It returns one named instance per part (13 parts).
 */
export async function build(engine, lib, variant) {
  engine.setSky({ preset: 'day' });
  engine.setWeather({ preset: 'clear' });
  engine.setLight({ direction: [-0.4, -0.8, -0.45], color: [1, 0.97, 0.92], intensity: 3 });
  engine.setAmbient({ color: [0.55, 0.6, 0.7], intensity: 0.3 });
  engine.setBloom({ strength: 0 });
  const m = Object.fromEntries(Object.entries(MATERIALS).map(([name, color]) => [name, engine.addMaterial({ color, roughness: 0.85 })]));
  const instances = [];
  const place = (name, desc, materials, at = [0, 0, 0]) => instances.push({ name, instance: addPart(engine, desc, materials, at) });
  place('ground', plane(160, 160), m.ground, [0, -0.01, 0]);
  place('stripe-panel', facadeCourses({ x: -2.5, zMin: -4, zMax: -1, yMin: 0.3, yMax: 2.3, course: 0.04, depth: 0.02, skip: { yMin: 99, yMax: 99, zMin: 0, zMax: 0 } }), [m.white, m.black]);
  [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1].forEach((x, i) => place(`thin-pole-${i}`, box([x, 1.5, -3], [0.015, 1.5, 0.015]), m.black));
  place('checker-patch', tiledGround({ x0: -1, z0: -2.5, columns: 16, rows: 16, size: 0.08 }), [m.white, m.black], [0, 0.005, 0]);
  [[[-3, 0.5, -30], m.red], [[2, 0.5, -45], m.yellow], [[0, 0.5, -60], m.blue]]
    .forEach(([center, material], i) => place(`distant-cube-${i}`, box(center, [0.5, 0.5, 0.5]), material));
  engine.setCamera(variant.selection.camera);
  return { instances };
}

/**
 * `observe(engine, variant, state)` reads the render scale, the TAA pass entry and the sharpening settings,
 * and returns the three assertions of the rows in order. The expected render scale is the variant's
 * renderScale, 1 when it names none; each detail names every value it read and the value it expects.
 */
export async function observe(engine, variant, state) {
  const scale = engine.renderScale();
  const taa = engine.passes().find((entry) => entry.name === 'taa')?.enabled === true;
  const sharpening = engine.sharpening();
  const expectedScale = variant.settings.renderScale ?? 1;
  const expectedSharpen = variant.settings.sharpening.enabled;
  return {
    values: { 'render-scale': scale, 'taa-enabled': taa, 'sharpen-enabled': sharpening.enabled },
    assertions: [
      { name: 'render-scale', passed: Math.abs(scale - expectedScale) < 1e-6, detail: `renderScale ${scale}, expected ${expectedScale}` },
      { name: 'taa-enabled', passed: taa === true, detail: `taa pass enabled ${taa}, expected true` },
      { name: 'sharpen-enabled', passed: sharpening.enabled === expectedSharpen, detail: `sharpening enabled ${sharpening.enabled} (sharpness ${sharpening.sharpness}), expected ${expectedSharpen}` },
    ],
  };
}

export default {
  id: 'sharpen-detail',
  milestone: 'sharpen',
  title: 'Sharpening: fine detail at render scale 1 and 0.67',
  variants,
  build,
};
