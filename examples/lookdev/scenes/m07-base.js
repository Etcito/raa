// Lookdev scene m07-base (chain milestone 1, R7): the six lookdev objects (inflatable
// hammer, ice block, plush sphere, neon tube, slime pool, floor tiles) uploaded with
// addMesh and drawn with addMeshInstance, shaded by v0 materials (pbr and flat) under a
// bright sun, so the HDR chain, AgX/ACES tone mapping, exposure and MSAA can be reviewed.
// Pure module (S14): no top-level side effects, node reads id/milestone/variants.

// Display-referred linear background (S12). It differs from CLEAR_COLOR on purpose, so
// the `empty` variant still records nonEmpty while showing only this flat colour.
const BACKGROUND = [0.3, 0.34, 0.4];

// Warm sun travelling down and away from the camera; the intensity pushes lit glossy
// surfaces well above 1.0 in the HDR target so the tone mapper has to roll them off.
const SUN = { direction: [-0.45, -1, -0.55], color: [1, 0.95, 0.88], intensity: 4.5 };
const AMBIENT = { color: [0.55, 0.62, 0.78], intensity: 0.35 };

// Orbit cameras: [target, distance m, yaw rad, pitch rad] (yaw 0 looks down -Z).
const WIDE = [[0, 0.45, 0], 4.2, 0, 0.42];
const CLOSE_UP = [[-1.15, 0.4, -0.45], 1.7, -0.35, 0.45];

const yawQuat = (rad) => [0, Math.sin(rad / 2), 0, Math.cos(rad / 2)];

// Object placements on the 8 x 8 tile floor (top faces at y = 0): back row hammer, neon,
// ice; front row plush, slime.
const PLACES = {
  hammer: { t: [-1.15, 0, -0.45], r: yawQuat(0.35) },
  neon: { t: [0, 0, -0.9], r: yawQuat(0) },
  ice: { t: [1.15, 0, -0.45], r: yawQuat(-0.3) },
  plush: { t: [-0.75, 0, 0.75], r: yawQuat(0) },
  slime: { t: [0.75, 0, 0.75], r: yawQuat(0) },
};

// Sets each sub-mesh material of `desc` (in order) and returns the mesh handle.
function upload(engine, desc, materials) {
  const subMeshes = (desc.subMeshes ?? [{ first: 0, count: desc.indices.length }]).map((s, i) => ({
    ...s,
    material: materials[Math.min(i, materials.length - 1)],
  }));
  return engine.addMesh({ ...desc, subMeshes });
}

function place(engine, mesh, where) {
  const node = engine.addNode(null, where.t, where.r, [1, 1, 1]);
  return engine.addMeshInstance(node, mesh);
}

const R7 = 'R7: hammer, ice block, plush sphere, neon tube, slime pool and colourful tiles '
  + 'recognisable; flat background colour';

export default {
  id: 'm07-base',
  milestone: 'm07',
  title: 'Meshes and HDR base: six lookdev objects, bright sun, flat background',
  engine: { msaa: 4, toneMapper: 'agx', exposureEv: 0 },
  variants: [
    {
      id: 'agx',
      description: `${R7}; AgX tone mapping: the neon S glows pink-white at its core and the sun `
        + 'highlights on the glossy hammer roll off smoothly to a desaturated white without hue '
        + 'skew or clipped flat patches; edges anti-aliased (MSAA 4x).',
      references: [],
      judge: false,
      engine: { toneMapper: 'agx' },
    },
    {
      id: 'aces',
      description: `${R7}; ACES tone mapping, same framing as agx: visibly more contrast and `
        + 'saturation than agx (deeper shadows, more saturated neon and tiles), highlights still '
        + 'roll off without clipping.',
      references: [],
      judge: false,
      engine: { toneMapper: 'aces' },
    },
    {
      id: 'exposure-up',
      description: `${R7}; AgX at +1.5 EV: the whole lit scene is clearly brighter than agx, `
        + 'large highlight areas on the hammer, ice and tiles compress towards white smoothly '
        + '(no hard clipping edges or hue shifts); the background colour is unchanged by exposure.',
      references: [],
      judge: false,
      engine: { exposureEv: 1.5 },
    },
    {
      id: 'msaa-on',
      description: `${R7}; MSAA 4x: silhouettes of the neon tube, hammer, tile edges and pool rim `
        + 'are smooth with no stair-stepping against the background and the floor.',
      references: [],
      judge: false,
      engine: { msaa: 4 },
    },
    {
      id: 'msaa-off',
      description: `${R7}; MSAA off (1 sample), same framing as msaa-on: the same image but with `
        + 'visible jagged stair-step edges on the thin neon tube, tile edges and object '
        + 'silhouettes, which msaa-on removes.',
      references: [],
      judge: false,
      engine: { msaa: 1 },
    },
    {
      id: 'vertex-colors',
      description: 'R7 close-up of the inflatable hammer on the tiles: the coloured stripes and '
        + 'panels of the hammer (vertex colours) and the per-tile colours (cream, sage, '
        + 'terracotta, blue, each tile slightly different in brightness) are clearly visible '
        + 'with grout gaps between tiles; glossy hammer highlight rolls off smoothly.',
      references: [],
      judge: false,
      settings: { camera: 'close-up' },
    },
    {
      id: 'empty',
      description: 'R7: no objects, only the background: the whole frame is exactly one flat '
        + 'colour (display-referred linear 0.3, 0.34, 0.4, a muted grey-blue), no gradient, '
        + 'noise, banding or tone-mapping shift.',
      references: [],
      judge: false,
      settings: { empty: true },
    },
  ],

  async build(engine, lib, variant) {
    const settings = variant.settings ?? {};
    engine.setBackground(BACKGROUND);
    engine.setLight(SUN);
    engine.setAmbient(AMBIENT);
    const [target, distance, yaw, pitch] = settings.camera === 'close-up' ? CLOSE_UP : WIDE;
    engine.setOrbit(target, distance, yaw, pitch);
    if (settings.empty) return;

    const { objects, pbr, flat } = lib;
    // Vertex colours carry the hammer stripes and tile colours, so their bases are white.
    const tiles = pbr(engine, [1, 1, 1], { roughness: 0.55 });
    const hammer = pbr(engine, [1, 1, 1], { roughness: 0.12 });
    const ice = pbr(engine, [0.78, 0.9, 0.97], { roughness: 0.08 });
    const plush = pbr(engine, [0.86, 0.46, 0.56], { roughness: 1 });
    // Unlit tube whose linear HDR emissive (peak 7) is far above 1: the core must roll off.
    const neon = flat(engine, [0.02, 0.01, 0.02], { emissive: [7, 1.1, 3.6] });
    const basin = pbr(engine, [0.55, 0.36, 0.78], { roughness: 0.4 });
    const slime = pbr(engine, [0.5, 0.95, 0.22], { roughness: 0.1 });

    place(engine, upload(engine, objects.floorTiles(8), [tiles]), { t: [0, 0, 0], r: yawQuat(0) });
    place(engine, upload(engine, objects.inflatableHammer(), [hammer, hammer]), PLACES.hammer);
    place(engine, upload(engine, objects.neonTube(), [neon]), PLACES.neon);
    place(engine, upload(engine, objects.iceBlock(), [ice]), PLACES.ice);
    place(engine, upload(engine, objects.plushSphere(), [plush]), PLACES.plush);
    place(engine, upload(engine, objects.slimePool(), [basin, slime]), PLACES.slime);
  },
};
