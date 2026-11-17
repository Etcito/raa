// The mips-ground scene (S49, post page, stage mips): a fine generated checker ground that runs to the horizon
// and three small striped cubes far away, seen by one grazing camera. Two variants make the pair 'mips': the
// same textures added with mips on and with mips off. The material function samples raa_texture_grad with the
// derivatives of the unwrapped uv taken at the top of material_main. Importing this module has no side effects.
// Geometry and textures come from '../lib/content.js' (plane, box, checker, stripes).
import { box, checker, plane, stripes } from '../lib/content.js';

// The scene identity: the default export id and every row's scenario (namePattern in lib/inventory.js).
export const MIPS_SCENE_ID = 'mips-ground';
// The measurement preset label of the pair, checked only for equality across its two variants.
export const MIPS_PAIR_PRESET = 'mips-ground-fine-pattern';
// The material function both surface materials use; registered once per build.
export const MIPS_MATERIAL_FUNCTION = 'post.mips_ground';

// The derivatives are taken once, at the top, from the unwrapped uv; fract only feeds the sample position.
export const MIPS_MATERIAL_WGSL = `fn material_main(in: MaterialInput) -> MaterialOutput {
  let duv_dx = dpdx(in.uv);
  let duv_dy = dpdy(in.uv);
  var o = raa_default_output(in);
  let id = raa_material_word(raa_material_data_base(in.record));
  let c = raa_texture_grad(id, fract(in.uv), duv_dx, duv_dy);
  o.base_color = raa_srgb_to_linear(c.rgb);
  o.roughness = 0.85;
  return o;
}
`;

const GROUND_SIZE = 400; // metres: the plane runs to the horizon
const GROUND_TILE_M = 4; // uv = (x, z) / 4: one 1024-texel tile per 4 m
const GROUND_TEXELS = 1024;
const GROUND_PERIOD = 4; // checker cells of 2 texels
const CUBE_HALF = 0.4; // metres
const CUBE_TEXELS = 256;
const CUBE_PERIOD = 4; // stripes of 2 texels
const CUBE_TILES_PER_FACE = 2; // two stripe tiles across each face
const CUBE_CENTRES = [[-2, 0.4, -40], [1.5, 0.4, -55], [0, 0.4, -75]];
const EXPECTED_TEXTURES = 2;

const boxAround = (centre, half) => ({ min: centre.map((c) => c - half), max: centre.map((c) => c + half) });

// The one grazing ground camera of both rows. Each subject box equals its geometry and lies in the frustum
// (assertSubjectsInView, checked in mips.test.mjs).
const SELECTION = {
  camera: { eye: [0, 1.2, 6], target: [0, 0.4, -60], fovYDeg: 50, near: 0.1, far: 500 },
  subjects: [
    { name: 'ground', min: [-200, -0.01, -200], max: [200, 0, 200] },
    ...CUBE_CENTRES.map((centre, i) => ({ name: `far-cube-${i}`, ...boxAround(centre, CUBE_HALF) })),
  ],
};

// One variant per state; the id is 'mips-<state>', the suffix the inventory checks against measurement.state.
const variantOf = (state) => ({
  id: `mips-${state}`,
  selection: SELECTION,
  settings: {},
  measurement: { pair: 'mips', state, preset: MIPS_PAIR_PRESET, features: { mips: state === 'on' } },
});
const variants = ['on', 'off'].map(variantOf);

const REQUIRED_FACTS = [
  'runtime:passes', 'runtime:viewSizes', 'runtime:ready', 'fact:mips-enabled', 'fact:texture-count',
  'assertion:mips-enabled', 'assertion:texture-count',
];
// A row takes its pair from its variant; the producer is raa-atlas-mips when the variant's textures carry mips.
const row = (mode, filename, expectedVisibleBehavior) => {
  const v = variants.find((x) => x.id === mode);
  return {
    id: mode,
    criterionName: `P5-Q4 mips ${mode}`,
    scenario: MIPS_SCENE_ID,
    mode,
    filename,
    expectedVisibleBehavior,
    requiredFacts: REQUIRED_FACTS,
    pair: v.measurement.pair,
    heldConstant: ['sky, sun and ambient lighting', 'fine-pattern ground tiled to the horizon and the small textured cubes far away', 'grazing ground camera'],
    producer: v.measurement.features.mips ? 'raa-atlas-mips' : 'post-mips-scenes',
  };
};

export const inventory = [
  row('mips-on', 'mips-on-distant-ground-smooth.png', 'A ground covered by a fine generated pattern runs to the horizon with small textured objects far away: the distance turns smoothly into an even tone with no moire bands and no sparkle; the near pattern is sharp.'),
  row('mips-off', 'mips-off-distant-ground-moire.png', 'The same view with the textures added with `mips: false`: the distant ground shows moire bands and noisy aliasing; the near pattern is as sharp as with mips on.'),
];

export const factNames = ['mips-enabled', 'texture-count'];

// Per-vertex uvs for a mesh without them: uvOf(position, normal) gives [u, v] for one vertex, unwrapped beyond 1.
function withUvs(mesh, uvOf) {
  const count = mesh.positions.length / 3;
  const uvs = new Float32Array(2 * count);
  for (let i = 0; i < count; i++) {
    uvs.set(uvOf(mesh.positions.subarray(3 * i, 3 * i + 3), mesh.normals.subarray(3 * i, 3 * i + 3)), 2 * i);
  }
  return { ...mesh, uvs };
}

// Ground: uv = (x, z) / 4.
const groundDesc = () => withUvs(plane(GROUND_SIZE, GROUND_SIZE), (p) => [p[0] / GROUND_TILE_M, p[2] / GROUND_TILE_M]);

// Cube: per face, the two coordinates orthogonal to the normal's dominant axis, relative to the box min, divided
// by the side and times 2.
function cubeDesc(centre) {
  const min = centre.map((c) => c - CUBE_HALF);
  const side = 2 * CUBE_HALF;
  return withUvs(box(centre, [CUBE_HALF, CUBE_HALF, CUBE_HALF]), (p, n) => {
    const dominant = [0, 1, 2].reduce((best, k) => (Math.abs(n[k]) > Math.abs(n[best]) ? k : best), 0);
    const [a, b] = [0, 1, 2].filter((k) => k !== dominant);
    return [((p[a] - min[a]) / side) * CUBE_TILES_PER_FACE, ((p[b] - min[b]) / side) * CUBE_TILES_PER_FACE];
  });
}

// Adds one mesh whose sub-meshes take `material`, and one instance of it at `at`; returns the instance handle.
// As addPart in examples/post/scenes/sharpen.js.
function addPart(engine, desc, material, at) {
  const parts = desc.subMeshes ?? [{ first: 0, count: desc.indices.length }];
  const mesh = engine.addMesh({ ...desc, subMeshes: parts.map((part) => ({ ...part, material })) });
  return engine.addMeshInstance(engine.addNode(null, at, [0, 0, 0, 1], [1, 1, 1]), mesh);
}

// The surface material of one texture: its data block holds the texture id that the material function reads.
function surfaceOf(engine, texture) {
  const base = engine.addMaterialData(Uint32Array.of(texture));
  return engine.addSurfaceMaterial({ function: MIPS_MATERIAL_FUNCTION, data: base });
}

/**
 * `build(engine, lib, variant)` sets the day sky, clear weather, the light, the ambient and no bloom, registers the
 * material function, adds the checker ground and the stripe texture with the variant's mips flag, then adds the
 * ground and the three far cubes and sets the variant camera. It returns one named instance per part (4) and the
 * two textures in addTexture order (2).
 */
export async function build(engine, lib, variant) {
  const mips = variant.measurement.features.mips;
  engine.setSky({ preset: 'day' });
  engine.setWeather({ preset: 'clear' });
  engine.setLight({ direction: [-0.4, -0.8, -0.45], color: [1, 0.97, 0.92], intensity: 3 });
  engine.setAmbient({ color: [0.55, 0.6, 0.7], intensity: 0.3 });
  engine.setBloom({ strength: 0 });
  engine.addMaterialFunction(MIPS_MATERIAL_FUNCTION, MIPS_MATERIAL_WGSL);
  const ground = engine.addTexture({ ...checker(GROUND_TEXELS, GROUND_PERIOD), mips });
  const cube = engine.addTexture({ ...stripes(CUBE_TEXELS, CUBE_PERIOD), mips });
  const groundMaterial = surfaceOf(engine, ground);
  const cubeMaterial = surfaceOf(engine, cube);
  const instances = [];
  const place = (name, desc, material, at = [0, 0, 0]) => instances.push({ name, instance: addPart(engine, desc, material, at) });
  place('ground', groundDesc(), groundMaterial, [0, -0.01, 0]);
  CUBE_CENTRES.forEach((centre, i) => place(`far-cube-${i}`, cubeDesc(centre), cubeMaterial));
  engine.setCamera(variant.selection.camera);
  return { instances, textures: [{ id: ground, mips }, { id: cube, mips }] };
}

/**
 * `observe(engine, variant, state)` reads the texture flags of the built state (the engine is not read) and returns
 * the two assertions of the rows in order. `mips-enabled` is the flag every texture shares (false when they differ or
 * there are none); each detail names the flags it read and the value it expects.
 */
export async function observe(engine, variant, state) {
  const flags = state.textures.map((texture) => texture.mips);
  const expected = variant.measurement.features.mips;
  const common = flags.length > 0 && flags.every((flag) => flag === flags[0]) ? flags[0] : false;
  const count = flags.length;
  return {
    values: { 'mips-enabled': common, 'texture-count': count },
    assertions: [
      { name: 'mips-enabled', passed: flags.every((flag) => flag === expected), detail: `texture mips [${flags.join(', ')}], expected ${expected} for every texture` },
      { name: 'texture-count', passed: count === EXPECTED_TEXTURES, detail: `texture count ${count}, expected ${EXPECTED_TEXTURES}` },
    ],
  };
}

export default {
  id: MIPS_SCENE_ID,
  milestone: 'mips',
  title: 'Atlas mip chain: fine-pattern ground to the horizon',
  variants,
  build,
};
