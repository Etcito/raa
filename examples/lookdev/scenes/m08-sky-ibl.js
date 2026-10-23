// Lookdev scene m08-sky-ibl (chain milestone 2, R8): the six lookdev objects on the tile
// floor under the procedural `day` sky with clouds, lit by a warm sun and by the image-based
// lighting derived from the sky (SH9 diffuse + prefiltered specular). On/off pairs for the
// sky (`sky.env` + `sky.background`) and for the IBL (`sky.env` only) (Q15), the snow and
// night presets, and glossy spheres reflecting the sky.
// Pure module (S14): no top-level side effects, node reads id/milestone/variants.

// Display-referred linear background (S12), seen only when the sky passes are disabled.
const BACKGROUND = [0.3, 0.34, 0.4];

// Warm daytime sun high in the upper right of the default framing (the sky draws its disc
// opposite the travel direction), so the disc is in frame and the objects are side-lit.
const SUN = { direction: [-0.58, -0.43, 0.69], color: [1, 0.93, 0.82], intensity: 4 };
// Cool dim moonlight for the night preset, same direction as the sun.
const MOON = { direction: [-0.58, -0.43, 0.69], color: [0.6, 0.7, 1], intensity: 0.4 };
// Flat ambient: replaced by the sky's SH ambient while the env is on, shown only when off.
const AMBIENT = { color: [0.55, 0.6, 0.7], intensity: 0.35 };

// Orbit cameras: [target, distance m, yaw rad, pitch rad] (yaw 0 looks down -Z; a negative
// pitch puts the eye below the target, looking slightly up).
const WIDE = [[0, 0.9, 0], 4.6, 0, -0.08];
// The close view looks level (60 deg vertical fov): the sun disc (about 25 deg up) and the
// deep blue upper sky stay inside the top of the frame.
const CLOSE = [[0, 0.45, 0], 3, 0.15, 0];

const yawQuat = (rad) => [0, Math.sin(rad / 2), 0, Math.cos(rad / 2)];

// Same placements as m07-base: back row hammer, neon, ice; front row plush, slime.
const PLACES = {
  hammer: { t: [-1.15, 0, -0.45], r: yawQuat(0.35) },
  neon: { t: [0, 0, -0.9], r: yawQuat(0) },
  ice: { t: [1.15, 0, -0.45], r: yawQuat(-0.3) },
  plush: { t: [-0.75, 0, 0.75], r: yawQuat(0) },
  slime: { t: [0.75, 0, 0.75], r: yawQuat(0) },
};

// Glossy spheres of the `glossy` variant, in front of the objects (radius 0.28 m).
const SPHERE_RADIUS = 0.28;
const SPHERES = [
  { t: [-0.75, SPHERE_RADIUS, 1.55], color: [0.95, 0.93, 0.9], roughness: 0.05, metallic: 1 },
  { t: [0, SPHERE_RADIUS, 1.55], color: [0.95, 0.78, 0.45], roughness: 0.3, metallic: 1 },
  { t: [0.75, SPHERE_RADIUS, 1.55], color: [0.85, 0.12, 0.2], roughness: 0.05, metallic: 0 },
];

// Sets each sub-mesh material of `desc` (in order) and returns the mesh handle.
function upload(engine, desc, materials) {
  const subMeshes = (desc.subMeshes ?? [{ first: 0, count: desc.indices.length }]).map((s, i) => ({
    ...s,
    material: materials[Math.min(i, materials.length - 1)],
  }));
  return engine.addMesh({ ...desc, subMeshes });
}

function place(engine, mesh, where) {
  const node = engine.addNode(null, where.t, where.r ?? yawQuat(0), [1, 1, 1]);
  return engine.addMeshInstance(node, mesh);
}

const R8 = 'R8: an inflatable striped hammer, an ice block, a pink plush sphere, a glowing neon '
  + 'tube, a green slime pool and colourful floor tiles';
const SKY = 'a bright blue gradient sky, deep blue at the top fading to a pale warm horizon, a '
  + 'small bright sun disc in the upper right and soft white procedural clouds';

export default {
  id: 'm08-sky-ibl',
  milestone: 'm08',
  title: 'Procedural sky with clouds and image-based lighting from it',
  engine: { msaa: 4, toneMapper: 'agx', exposureEv: 0 },
  variants: [
    {
      id: 'sky-on',
      description: `${R8} under ${SKY} filling the upper half of the frame; the objects pick up `
        + 'soft sky-coloured ambient light (bluish from above, warm sunlit sides) and the glossy '
        + 'hammer and ice show sky reflections; bright, cheerful daytime mood like the reference.',
      references: ['ref-1'],
      judge: true,
    },
    {
      id: 'sky-off',
      description: `${R8}, same framing as sky-on, but the sky passes are off: the background `
        + 'is one flat muted grey-blue colour (no gradient, sun disc or clouds) and the objects '
        + 'are lit by the sun plus a flat uniform ambient, without sky-coloured light.',
      references: [],
      judge: false,
      passes: { disable: ['sky.env', 'sky.background'] },
    },
    {
      id: 'ibl-on',
      description: `Closer view of ${R8.slice(4)} with ${SKY} visible behind them; the objects `
        + 'are lit by soft sky ambient light (blue tint on upward faces, gentle soft shading '
        + 'all around) and glossy surfaces (hammer, ice, slime) reflect the blue sky and the '
        + 'bright horizon; bright daytime mood like the reference.',
      references: ['ref-1'],
      judge: true,
      settings: { camera: 'close' },
    },
    {
      id: 'ibl-off',
      description: `${R8}, same framing as ibl-on with the sky still visible behind, but the `
        + 'environment lighting is off: the objects are lit by the sun and a flat uniform '
        + 'ambient only, with no sky-coloured tint or sky reflections, so they look flat and '
        + 'dull compared with ibl-on.',
      references: [],
      judge: false,
      settings: { camera: 'close' },
      passes: { disable: ['sky.env'] },
    },
    {
      id: 'preset-snow',
      description: `${R8} under the snow preset: an overcast pale grey-blue sky with a bright `
        + 'whitish horizon, dense soft clouds and a dim sun disc; the objects take a cool, soft, '
        + 'bright ambient light.',
      references: [],
      judge: false,
      settings: { preset: 'snow' },
    },
    {
      id: 'preset-night',
      description: `${R8} under the night preset: a dark navy sky with a faint horizon glow and `
        + 'a small cool moon-like disc; the scene is dim and cool blue, only the neon tube glows '
        + 'brightly.',
      references: [],
      judge: false,
      settings: { preset: 'night' },
    },
    {
      id: 'glossy',
      description: `${R8} with three spheres in front: a mirror-like chrome sphere (metallic, `
        + 'roughness 0.05) sharply reflecting the blue sky, horizon and clouds, a gold metallic '
        + 'sphere (roughness 0.3) with a blurred sky reflection, and a glossy red plastic sphere '
        + 'with a sharp sky highlight over its red base; the sky is visible behind.',
      references: [],
      judge: false,
      settings: { camera: 'close', spheres: true },
    },
  ],

  async build(engine, lib, variant) {
    const settings = variant.settings ?? {};
    const preset = settings.preset ?? 'day';
    engine.setBackground(BACKGROUND);
    engine.setLight(preset === 'night' ? MOON : SUN);
    engine.setAmbient(AMBIENT);
    engine.setSky({ preset });
    engine.setEnvironmentLighting({ diffuse: 1, specular: 1 });
    const [target, distance, yaw, pitch] = settings.camera === 'close' ? CLOSE : WIDE;
    engine.setOrbit(target, distance, yaw, pitch);

    const { objects, geometry, pbr, flat } = lib;
    // Vertex colours carry the hammer stripes and tile colours, so their bases are white.
    const tiles = pbr(engine, [1, 1, 1], { roughness: 0.55 });
    const hammer = pbr(engine, [1, 1, 1], { roughness: 0.12 });
    const ice = pbr(engine, [0.78, 0.9, 0.97], { roughness: 0.08 });
    const plush = pbr(engine, [0.86, 0.46, 0.56], { roughness: 1 });
    const neon = flat(engine, [0.02, 0.01, 0.02], { emissive: [7, 1.1, 3.6] });
    const basin = pbr(engine, [0.55, 0.36, 0.78], { roughness: 0.4 });
    const slime = pbr(engine, [0.5, 0.95, 0.22], { roughness: 0.1 });

    place(engine, upload(engine, objects.floorTiles(8), [tiles]), { t: [0, 0, 0] });
    place(engine, upload(engine, objects.inflatableHammer(), [hammer, hammer]), PLACES.hammer);
    place(engine, upload(engine, objects.neonTube(), [neon]), PLACES.neon);
    place(engine, upload(engine, objects.iceBlock(), [ice]), PLACES.ice);
    place(engine, upload(engine, objects.plushSphere(), [plush]), PLACES.plush);
    place(engine, upload(engine, objects.slimePool(), [basin, slime]), PLACES.slime);

    if (!settings.spheres) return;
    const sphere = geometry.uvSphere(SPHERE_RADIUS, 64, 32);
    for (const s of SPHERES) {
      const material = pbr(engine, s.color, { roughness: s.roughness, metallic: s.metallic });
      place(engine, upload(engine, sphere, [material]), { t: s.t });
    }
  },
};
