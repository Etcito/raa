// Lookdev scene m08-sky-ibl (chain milestone 2, R8): the six lookdev objects on the tile
// floor under the procedural `day` sky with clouds, lit by a warm sun and by the image-based
// lighting derived from the sky (SH9 diffuse + prefiltered specular). On/off pairs for the
// sky (`sky.env` + `sky.background`) and for the IBL (`sky.env` only) (Q15), the snow and
// night presets, and glossy spheres reflecting the sky.
// Pure module (S14): no top-level side effects, node reads id/milestone/variants.

// Display-referred linear background (S12), seen only when the sky passes are disabled.
const BACKGROUND = [0.3, 0.34, 0.4];

// Warm daytime sun about 17 deg up and 26 deg right of -Z (the sky draws its disc opposite
// the travel direction): in the upper right of both framings but away from the frame edge,
// where the perspective stretched the disc into an ellipse, and the objects are side-lit.
const SUN = { direction: [-0.419, -0.292, 0.86], color: [1, 0.93, 0.82], intensity: 5 };
// Cool dim moonlight for the night preset, same direction as the sun.
const MOON = { direction: [-0.419, -0.292, 0.86], color: [0.6, 0.7, 1], intensity: 0.4 };
// Flat ambient: replaced by the sky's SH ambient while the env is on, shown only when off.
const AMBIENT = { color: [0.55, 0.6, 0.7], intensity: 0.35 };

// Overrides of the `day` preset: a saturated blue zenith (AgX desaturates bright blues), a
// pale warm horizon and ground, a fast gradient (softness 5: blue down to about 10 deg,
// then a narrow pale band at the horizon), a stylised sun disc (22 mrad, about 28 px across
// instead of the 6 px of the real sun, so it reads as a disc inside its glow) and larger,
// denser cumulus.
const DAY_SKY = {
  zenith: [0.03, 0.17, 0.95],
  horizon: [0.7, 0.8, 0.92],
  ground: [0.5, 0.45, 0.4],
  intensity: 1.6,
  horizonSoftness: 5,
  sunDiscSize: 0.022,
  sunDiscIntensity: 20,
  clouds: { coverage: 0.5, density: 1.2 },
};

// Orbit cameras: [target, distance m, yaw rad, pitch rad] (yaw 0 looks down -Z; a positive
// pitch puts the eye above the target, looking slightly down). The eye stays about 1 m up,
// so the horizon sits just above mid-frame: sky in the upper part, tiles in the lower part.
// The close eye (0.85 m) looks at the slime at about 15 deg, where its Fresnel reflection
// of the sky and clouds is clearly visible without washing out the green.
const WIDE = [[0, 0.5, 0], 3.4, 0, 0.12];
const CLOSE = [[0.1, 0.7, 0], 3.0, 0.12, 0.05];

const yawQuat = (rad) => [0, Math.sin(rad / 2), 0, Math.cos(rad / 2)];

// Back row hammer, neon, ice; front row plush, slime. The plush (0.9 m) sits left of the
// sight lines from both eyes to the neon S and to the hammer, so neither is hidden.
const PLACES = {
  hammer: { t: [-1.75, 0, -0.35], r: yawQuat(0.5) },
  neon: { t: [0.3, 0, -0.85], r: yawQuat(0) },
  ice: { t: [1.2, 0, -0.45], r: yawQuat(-0.4) },
  plush: { t: [-0.55, 0, 0.6], r: yawQuat(0) },
  slime: { t: [0.8, 0, 0.55], r: yawQuat(0) },
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

const R8 = 'R8: simple untextured procedural stand-in shapes (no fur, bloom or transparency '
  + 'yet): an upright striped inflatable toy hammer on the left, a pink sphere standing for a '
  + 'plush, a thin pink S-shaped neon tube, a blue rounded ice block, a round purple basin '
  + 'holding a green slime surface, on a large floor of pastel tiles reaching the horizon';
const SKY = 'a bright blue gradient sky, deep blue at the top fading to a pale warm horizon, a '
  + 'small white sun disc with a bright glow in the upper right and soft white puffy '
  + 'procedural clouds';
// R8 judges the sky and the light mood against ref-1; the objects are placeholders until the
// material, shadow and transparency milestones, so the judge is told not to score them.
const BASIS = 'Engine lookdev capture of the sky-and-sky-lighting milestone. Score two things: '
  + 'that the capture shows what this description lists, and that its sky and daylight mood '
  + '(blue sky with white clouds, bright, clear, cheerful daytime light) match the reference. '
  + 'The reference is a finished illustration and is NOT the target for the objects: their '
  + 'plain modelling, the absence of characters, textures, fur, bloom, transparency, cast '
  + 'shadows and post effects are expected at this milestone and must not lower the score. '
  + 'Shows: ';

export default {
  id: 'm08-sky-ibl',
  milestone: 'm08',
  title: 'Procedural sky with clouds and image-based lighting from it',
  engine: { msaa: 4, toneMapper: 'agx', exposureEv: 0.3 },
  variants: [
    {
      id: 'sky-on',
      description: `${BASIS}${R8.slice(4)}, under ${SKY} filling the upper part of the `
        + 'frame; the objects pick up soft sky-coloured ambient light (bluish from above, warm '
        + 'sunlit right sides, no black areas) and the glossy hammer and ice show soft sky '
        + 'highlights.',
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
      description: `${BASIS}a closer view of ${R8.slice(4)}, with ${SKY} filling the upper `
        + 'half of the frame behind them; the objects are lit by soft sky ambient light (blue '
        + 'tint on upward faces, gentle soft shading all around, no black shadows, pastel tile '
        + 'colours); the green slime surface and the blue ice block show a faint pale sheen of '
        + 'the sky, and the hammer head has soft glossy highlights.',
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
    engine.setSky(preset === 'day' ? { preset, ...DAY_SKY } : { preset });
    // A softer sky irradiance keeps the object colours (the blue zenith otherwise turns the
    // pink plush magenta); reflections stay at full strength.
    engine.setEnvironmentLighting({ diffuse: 0.75, specular: 1 });
    // AgX `punchy` and +0.3 EV: brighter, more saturated daylight, closer to ref-1's mood.
    engine.setToneMapping({ look: 'punchy' });
    const [target, distance, yaw, pitch] = settings.camera === 'close' ? CLOSE : WIDE;
    engine.setOrbit(target, distance, yaw, pitch);

    const { objects, geometry, pbr, flat } = lib;
    // Vertex colours carry the hammer stripes and tile colours, so their bases are white.
    // Ice and slime have deeper bases and near-minimum roughness, so their sky reflections
    // (a few percent at normal incidence) stand out instead of vanishing on a pale base.
    const tiles = pbr(engine, [1, 1, 1], { roughness: 0.55 });
    const hammer = pbr(engine, [1, 1, 1], { roughness: 0.08 });
    const ice = pbr(engine, [0.35, 0.6, 0.85], { roughness: 0.03 });
    const plush = pbr(engine, [1, 0.45, 0.5], { roughness: 1 });
    // A moderate emissive keeps the neon a saturated pink after AgX (brighter goes white).
    const neon = flat(engine, [0.02, 0.01, 0.02], { emissive: [3.2, 0.3, 1.5] });
    const basin = pbr(engine, [0.55, 0.36, 0.78], { roughness: 0.4 });
    const slime = pbr(engine, [0.12, 0.6, 0.05], { roughness: 0.04 });

    // 64 x 64 tiles (32 m): the floor reaches the horizon haze instead of ending in a band
    // of the sky's ground colour.
    place(engine, upload(engine, objects.floorTiles(64), [tiles]), { t: [0, 0, 0] });
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
