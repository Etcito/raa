// Lookdev scene m09-shadows-ao (chain milestone 3, R9): an obstacle-like arrangement (the
// inflatable hammer, stacked rounded blocks forming creases and inner corners, spheres
// resting on the floor, small 5-10 cm objects and a tall pillar in the right foreground) on
// large glossy candy floor panels under the m08 `day` sky, lit by a side sun about 35 deg
// high from the right and slightly behind, so the shadows fall to the left and a little
// toward the camera. On/off pairs for the cascaded sun shadows (`shadow.csm`), GTAO
// (`ao.gtao`) and contact shadows (`shadow.contact`) (Q15), all three together, a wide
// penumbra and a low sun with a long pillar shadow.
// Pure module (S14): no top-level side effects, node reads id/milestone/variants.

// Display-referred linear background (S12), unused while the sky is on.
const BACKGROUND = [0.3, 0.34, 0.4];

// Sun travel direction at elevation `deg`, coming from 80 deg to the right of -Z (as in
// ref-1, upper right, slightly behind): the shadows fall to the left and a little toward the
// camera, so they run across the screen at full length (at 65 deg the pillar stripe pointed
// at the camera, foreshortened, and the judge did not see it).
function sunDirection(deg) {
  const e = (deg * Math.PI) / 180;
  const a = (80 * Math.PI) / 180;
  return [-Math.cos(e) * Math.sin(a), -Math.sin(e), Math.cos(e) * Math.cos(a)];
}
const SUN_HIGH_DEG = 35;
// About 8 deg: the 2.2 m pillar casts a shadow of about 16 m, across the whole frame.
const SUN_LOW_DEG = 8;
// Sun strong against the sky fill (IBL diffuse 0.6 in build): with 5.5 and full sky diffuse
// the shadowed floor was barely darker than the lit floor and the judge saw no shadows.
const SUN = { color: [1, 0.93, 0.82], intensity: 8 };
// Flat ambient: replaced by the sky's SH ambient while the env is on.
const AMBIENT = { color: [0.55, 0.6, 0.7], intensity: 0.35 };

// The m08 `day` sky overrides (m08-sky-ibl.js DAY_SKY).
const DAY_SKY = {
  zenith: [0.03, 0.17, 0.95],
  horizon: [0.32, 0.55, 0.95],
  ground: [0.5, 0.45, 0.4],
  intensity: 1.6,
  horizonSoftness: 5,
  sunDiscSize: 0.022,
  sunDiscIntensity: 20,
  clouds: { coverage: 0.5, density: 2.5, scale: 1800 },
};

// Candy panel palette (linear vertex colours): pink, sky blue, yellow, mint, a little lighter
// than the m08 tiles so the sky-tinted shadows read on them (ref-1's floor panels).
const TILE_PALETTE = [
  [1, 0.16, 0.4],
  [0.1, 0.42, 1],
  [1, 0.72, 0.1],
  [0.24, 0.85, 0.3],
];
// Floor panels: PANELS x PANELS panels of PANEL_PITCH m (1.5 m, as ref-1's large panels; the
// 0.5 m tiles' grout lines broke every shadow into small pieces).
const PANELS = 20;
const PANEL_PITCH = 1.5;
const PANEL_HALF = [0.73, 0.03, 0.73];

// Base feature settings of every variant; a variant's settings.shadows / ao / contact merge
// over them. A 30 m shadow distance (default 60) keeps the cascades dense on this small set.
const SHADOWS = { softness: 1, distance: 30 };
const AO = { radius: 0.8, intensity: 1, power: 1.5 };
const CONTACT = { distance: 0.35, thickness: 0.05, intensity: 1 };

// Orbit cameras: [target, distance m, yaw rad, pitch rad] (yaw 0 looks down -Z; a positive
// pitch puts the eye above the target). WIDE sees the whole set and the floor in front of it,
// where the shadows land; CREASES looks at the stacked blocks; SMALL at the small objects.
const CAMERAS = {
  wide: [[0.3, 0.45, 0.2], 4.4, 0, 0.4],
  creases: [[0.25, 0.45, -0.7], 1.7, 0.25, 0.32],
  small: [[0.05, 0.05, 1.0], 0.9, 0.1, 0.5],
};

const yawQuat = (rad) => [0, Math.sin(rad / 2), 0, Math.cos(rad / 2)];

// The candy floor: one rounded panel mesh per palette colour, instanced on a PANELS grid
// (top face at y 0) with a deterministic hashed colour per panel.
function candyFloor(engine, geometry, material) {
  const meshes = TILE_PALETTE.map((color) => upload(engine,
    geometry.roundedBox(PANEL_HALF, 0.02, 2, { color: [...color, 1] }), [material]));
  for (let j = 0; j < PANELS; j++) {
    for (let i = 0; i < PANELS; i++) {
      const h = Math.imul(i * 73856093 ^ j * 19349663, 0x9e3779b1) >>> 0;
      const x = (i - (PANELS - 1) / 2) * PANEL_PITCH;
      const z = (j - (PANELS - 1) / 2) * PANEL_PITCH;
      place(engine, meshes[(h >>> 7) % meshes.length], [x, -PANEL_HALF[1], z]);
    }
  }
}

// Rounded blocks (half extents m, centre, linear colour): a wide base block, a cube stacked
// on it (inner corners on its top face) and a lower block touching its right side (a vertical
// crease and a floor corner).
const BLOCKS = [
  { half: [0.45, 0.3, 0.35], t: [0.3, 0.3, -0.75], color: [0.1, 0.35, 0.95] },
  { half: [0.22, 0.22, 0.22], t: [0.12, 0.82, -0.8], color: [1, 0.75, 0.05] },
  { half: [0.2, 0.18, 0.3], t: [0.95, 0.18, -0.65], color: [1, 0.2, 0.45] },
];
const BLOCK_RADIUS = 0.04;
// Spheres resting on the floor (radius m, x, z, colour).
const SPHERES = [
  { r: 0.3, x: -0.55, z: 0.1, color: [0.95, 0.15, 0.3] },
  { r: 0.2, x: 0.25, z: 0.05, color: [0.2, 0.85, 0.3] },
  { r: 0.26, x: 1.25, z: 0.15, color: [0.98, 0.65, 0.08] },
];
// Small objects for the contact shadows: spheres (radius) and blocks (half edge), 5-10 cm.
const SMALL = [
  { kind: 'sphere', s: 0.04, x: -0.25, z: 0.95, color: [1, 0.85, 0.1] },
  { kind: 'box', s: 0.04, x: -0.05, z: 1.05, color: [0.15, 0.4, 1] },
  { kind: 'sphere', s: 0.03, x: 0.1, z: 0.9, color: [1, 0.3, 0.55] },
  { kind: 'box', s: 0.05, x: 0.3, z: 1.0, color: [0.3, 0.9, 0.35] },
  { kind: 'sphere', s: 0.05, x: 0.45, z: 1.15, color: [0.95, 0.95, 0.95] },
];
// Tall pillar (0.3 x 2.2 x 0.3 m) in the right foreground, fully in frame: its 3.1 m shadow
// stripe runs to the left across the open floor in front of the small objects (at the back
// right it ran behind the blocks, at z 0.5 with the 65 deg sun it pointed at the camera).
const PILLAR = { half: [0.15, 1.1, 0.15], t: [1.9, 1.1, 1.6], color: [0.95, 0.9, 0.85] };
// Grout slab under the panels (top at y -0.04, inside the 6 cm panels): fills the 4 cm gaps,
// through which the sky showed as bright seams; pale like ref-1's panel seams.
const GROUT = { half: [15.2, 0.05, 15.2], t: [0, -0.09, 0], color: [0.75, 0.75, 0.8] };
const HAMMER = { t: [-1.3, 0, -0.6], r: yawQuat(0.4) };

// Sets each sub-mesh material of `desc` (in order) and returns the mesh handle.
function upload(engine, desc, materials) {
  const subMeshes = (desc.subMeshes ?? [{ first: 0, count: desc.indices.length }]).map((s, i) => ({
    ...s,
    material: materials[Math.min(i, materials.length - 1)],
  }));
  return engine.addMesh({ ...desc, subMeshes });
}

function place(engine, mesh, t, r = yawQuat(0)) {
  const node = engine.addNode(null, t, r, [1, 1, 1]);
  return engine.addMeshInstance(node, mesh);
}

const R9 = 'R9: simple untextured glossy procedural stand-in shapes on a large floor of '
  + 'bright candy-coloured square panels (pink, blue, yellow, green) under a blue sky with '
  + 'white clouds: an upright striped inflatable toy hammer on the left, a wide blue rounded '
  + 'block with a yellow cube stacked on it and a pink block against its right side, three '
  + 'coloured spheres resting on the floor in front, a row of tiny spheres and cubes (5-10 '
  + 'cm) in front of them and a tall pale grey pillar standing in the right foreground';
const SHADOWS_TEXT = 'soft sun shadows fall to the left and a little toward the camera: '
  + 'sharp where an object touches the floor and widening into a soft penumbra with '
  + 'distance; the pillar casts a long dark shadow stripe from its base to the left across '
  + 'the foreground floor panels';
const AO_TEXT = 'ambient occlusion darkens the creases between the stacked blocks, their inner '
  + 'corners and the floor around their bases and under the spheres';
const CONTACT_TEXT = 'the tiny objects sit on the floor with small dark shadows at their base';
// R9 judges the light and shadow mood against ref-1; the objects are placeholders until the
// material and transparency milestones, so the judge is told not to score them.
const BASIS = 'Engine lookdev capture of the shadows-and-ambient-occlusion milestone. Score two '
  + 'things: that the capture shows what this description lists, and that its light and shadow '
  + 'mood (bright sunny daylight, soft coloured shadows on the floor, grounded objects, '
  + 'cheerful clear lighting) matches the reference. The reference is a finished illustration '
  + 'and is NOT the target for the objects: their plain placeholder modelling, the absence of '
  + 'characters, textures, fur, bloom, transparency, reflections and post effects are expected '
  + 'at this milestone and must not lower the score; nor must the sparser, plainer set (few '
  + 'simple shapes, no lens glare or sun rays), which belongs to later milestones. Shows: ';

const off = (passes) => ({ disable: passes });

export default {
  id: 'm09-shadows-ao',
  milestone: 'm09',
  title: 'Soft cascaded sun shadows, GTAO and contact shadows',
  engine: { msaa: 4, toneMapper: 'agx', exposureEv: 0.3 },
  variants: [
    {
      id: 'all-on',
      description: `${BASIS}${R9.slice(4)}, lit by a warm sun from the upper right, slightly `
        + `behind the objects; ${SHADOWS_TEXT}, with no stripes, acne, gaps under objects or seams; `
        + `${AO_TEXT}; ${CONTACT_TEXT}. Shaded sides stay readable in the bluish sky light.`,
      references: ['ref-1'],
      judge: true,
    },
    {
      id: 'all-off',
      description: `${R9}, same framing as all-on, but shadows, ambient occlusion and contact `
        + 'shadows are all off: no shadow falls on the floor (not even from the pillar), the '
        + 'objects look pasted on the floor and the creases between blocks are as bright as '
        + 'the open faces.',
      references: [],
      judge: false,
      passes: off(['shadow.csm', 'ao.gtao', 'shadow.contact']),
    },
    {
      id: 'shadows-on',
      description: `${R9}; ${SHADOWS_TEXT}; every object, the hammer included, casts its own `
        + 'soft shadow to the left.',
      references: [],
      judge: false,
    },
    {
      id: 'shadows-off',
      description: `${R9}, same framing as shadows-on, but the cascaded sun shadows are off: `
        + 'no cast shadows on the floor or on other objects, no pillar shadow; only a faint '
        + 'darkening at contacts and in creases (ambient occlusion and contact shadows) remains.',
      references: [],
      judge: false,
      passes: off(['shadow.csm']),
    },
    {
      id: 'ao-on',
      description: `${R9.slice(0, 4)} a close view of the stacked rounded blocks (blue base, `
        + 'yellow cube on top, pink block against its side) on the candy floor panels; '
        + `${AO_TEXT}, a soft gradual darkening into the corners on top of the sun shadows.`,
      references: [],
      judge: false,
      settings: { camera: 'creases' },
    },
    {
      id: 'ao-off',
      description: `${R9.slice(0, 4)} the same close view of the stacked blocks as ao-on, but `
        + 'ambient occlusion is off: the creases, inner corners and the floor along the block '
        + 'bases are as bright as the open faces in the shaded areas, so the blocks look flatter.',
      references: [],
      judge: false,
      settings: { camera: 'creases' },
      passes: off(['ao.gtao']),
    },
    {
      id: 'contact-on',
      description: `${R9.slice(0, 4)} a close view from above of tiny spheres and cubes (5-10 `
        + 'cm, yellow, blue, pink, green, white) on the candy floor panels; each has a soft sun '
        + 'shadow to the left and a thin dark contact shadow right where it touches the '
        + 'floor, so it sits firmly on the floor.',
      references: [],
      judge: false,
      settings: { camera: 'small' },
    },
    {
      id: 'contact-off',
      description: `${R9.slice(0, 4)} the same close view of the tiny objects as contact-on, `
        + 'but contact shadows are off: the thin dark line where each object touches the floor '
        + 'is missing or faint, their soft sun shadows stay, so they look slightly floating.',
      references: [],
      judge: false,
      settings: { camera: 'small' },
      passes: off(['shadow.contact']),
    },
    {
      id: 'softness-wide',
      description: `${R9}, same framing as all-on, with shadow softness 3: the shadows are still `
        + 'sharp at contact but blur into a much wider, softer penumbra with distance than in '
        + 'all-on; the far end of the pillar shadow is a broad soft band.',
      references: [],
      judge: false,
      settings: { shadows: { softness: 3 } },
    },
    {
      id: 'long-shadow',
      description: `${R9}, lit by a low warm sun about 8 deg above the horizon on the right: `
        + 'very long soft shadows stretch to the left; the pillar shadow is a '
        + 'long stripe crossing the whole floor of the frame; the scene is darker and warmer.',
      references: [],
      judge: false,
      settings: { sun: 'low' },
    },
  ],

  async build(engine, lib, variant) {
    const settings = variant.settings ?? {};
    engine.setBackground(BACKGROUND);
    const elevation = settings.sun === 'low' ? SUN_LOW_DEG : SUN_HIGH_DEG;
    engine.setLight({ ...SUN, direction: sunDirection(elevation) });
    engine.setAmbient(AMBIENT);
    engine.setSky({ preset: 'day', ...DAY_SKY });
    // Sky diffuse 0.6: the shadowed floor keeps the blue sky tint of ref-1 and stays clearly
    // darker than the sunlit floor.
    engine.setEnvironmentLighting({ diffuse: 0.6, specular: 1 });
    engine.setToneMapping({ look: 'punchy' });
    engine.setShadows({ ...SHADOWS, ...settings.shadows });
    engine.setAo({ ...AO, ...settings.ao });
    engine.setContactShadows({ ...CONTACT, ...settings.contact });
    const [target, distance, yaw, pitch] = CAMERAS[settings.camera ?? 'wide'];
    engine.setOrbit(target, distance, yaw, pitch);

    const { objects, geometry, pbr } = lib;
    // Glossy surfaces (ref-1's inflatables and shiny floor): the sky reflects in them.
    const tiles = pbr(engine, [1, 1, 1], { roughness: 0.3 });
    const hammer = pbr(engine, [1, 1, 1], { roughness: 0.08 });
    candyFloor(engine, geometry, tiles);
    const grout = geometry.roundedBox(GROUT.half, 0.01, 2);
    place(engine, upload(engine, grout, [pbr(engine, GROUT.color, { roughness: 0.9 })]), GROUT.t);
    place(engine, upload(engine, objects.inflatableHammer(), [hammer, hammer]), HAMMER.t, HAMMER.r);

    const pillar = geometry.roundedBox(PILLAR.half, 0.04, 4);
    place(engine, upload(engine, pillar, [pbr(engine, PILLAR.color, { roughness: 0.3 })]), PILLAR.t);
    for (const b of BLOCKS) {
      const desc = geometry.roundedBox(b.half, BLOCK_RADIUS, 4);
      place(engine, upload(engine, desc, [pbr(engine, b.color, { roughness: 0.15 })]), b.t);
    }
    for (const s of SPHERES) {
      const desc = geometry.uvSphere(s.r, 64, 32);
      place(engine, upload(engine, desc, [pbr(engine, s.color, { roughness: 0.12 })]), [s.x, s.r, s.z]);
    }
    for (const o of SMALL) {
      const desc = o.kind === 'sphere'
        ? geometry.uvSphere(o.s, 32, 16)
        : geometry.roundedBox([o.s, o.s, o.s], 0.008, 3);
      place(engine, upload(engine, desc, [pbr(engine, o.color, { roughness: 0.2 })]), [o.x, o.s, o.z]);
    }
  },
};
