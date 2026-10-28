// S14: importing metadata creates no scene, GPU resources, timers or DOM state.
import { lathe } from '../lib/geometry.js';

const PIVOT_Y = 1.8;
const HEIGHT = 3.6;
const BLEND_HALF_WIDTH = 0.55;
const BEND = -65 * Math.PI / 180;

function creatureMesh() {
  // One connected surface, including rounded end caps; no rigid segment seam.
  // Rings through the joint make the gradual weight transition visible in wireframe.
  const profile = Array.from({ length: 49 }, (_, i) => {
    const y = HEIGHT * i / 48;
    const cap = Math.max(0, Math.abs(y - PIVOT_Y) - 1.28) / 0.52;
    const radius = i === 0 || i === 48 ? 0
      : 0.44 * Math.sqrt(Math.max(0, 1 - cap * cap)) * (1 + 0.08 * y / HEIGHT);
    return [radius, y];
  });
  const mesh = lathe(profile, 48, {
    // Two small charcoal eyes painted into the same neutral mesh, fully above
    // the blend region. No separate eye instances can conceal an overlay mismatch.
    color: ([x, y, z]) => {
      const eye = z > 0 && ((Math.abs(x) - 0.19) / 0.085) ** 2
        + ((y - 3.1) / 0.105) ** 2 < 1;
      return eye ? [0.025, 0.025, 0.025, 1] : [1, 1, 1, 1];
    },
  });
  const count = mesh.positions.length / 3;
  const joints = new Uint16Array(count * 4);
  const weights = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    const y = mesh.positions[i * 3 + 1];
    const t = Math.max(0, Math.min(1,
      (y - PIVOT_Y + BLEND_HALF_WIDTH) / (2 * BLEND_HALF_WIDTH)));
    const upper = t * t * (3 - 2 * t);
    joints[i * 4 + 1] = 1;
    weights[i * 4] = 1 - upper;
    weights[i * 4 + 1] = upper;
  }
  return { ...mesh, joints, weights };
}

function bentPose() {
  const c = Math.cos(BEND), s = Math.sin(BEND);
  // Column-major final skin matrices: root identity, then T(p) Rz T(-p).
  // The upper bone's rest model is T(p); its inverse bind is T(-p).
  return new Float32Array([
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0,
    s * PIVOT_Y, (1 - c) * PIVOT_Y, 0, 1,
  ]);
}

const SUBJECT = 'A single neutral grey, rounded two-segment creature with two small dark '
  + 'eyes on a plain grey-blue background, under soft directional lighting. ';
const POSED = 'The lower segment stays upright while the upper segment bends 65 degrees '
  + 'to image right around the middle. A broad, smoothly blended elbow joins them: '
  + 'the inside compresses and the outside stretches with no gap, detached pieces or hard seam. ';

export default {
  id: 'm18-skinning',
  milestone: 'm18',
  title: 'Skinning: continuous bent creature and aligned editor overlays',
  engine: { msaa: 4, toneMapper: 'agx', exposureEv: 0 },
  variants: [
    {
      id: 'skin-on',
      description: SUBJECT + POSED
        + 'The smooth matte surface and uninterrupted bent silhouette clearly reveal joint '
        + 'blending. No wireframe or selection outline is visible.',
      references: [],
      judge: true,
    },
    {
      id: 'skin-off',
      description: SUBJECT
        + 'Skinning is disabled while the same bent pose remains stored. The exact same '
        + 'mesh appears in its straight upright rest pose, with rounded ends and no bend. '
        + 'Camera, lighting, material and instance transform match skin-on exactly; '
        + 'wireframe and selection outline are off.',
      references: [],
      judge: false,
      passes: { disable: ['skin'] },
    },
    {
      id: 'skin-wire-on',
      description: SUBJECT + POSED
        + 'The pose, framing and lighting match skin-on exactly. Dark editor wireframe '
        + 'triangles lie on the posed surface, including the blended elbow and bent upper '
        + 'segment. An orange selection outline hugs the entire bent silhouette. Neither '
        + 'overlay leaves a straight rest-pose ghost or detaches from the surface.',
      references: [],
      judge: true,
      settings: { wire: true },
    },
  ],

  async build(engine, lib, variant) {
    engine.setBackground([0.23, 0.26, 0.30]);
    engine.setLight({ direction: [-0.6, -1, -0.8], color: [1, 0.97, 0.93], intensity: 3 });
    engine.setAmbient({ color: [0.8, 0.85, 1], intensity: 0.55 });
    // A fixed near-front view keeps the XY bend and both eyes readable, with room
    // for the taller rest shape. No variant-specific reframing or transforms.
    engine.setOrbit([0.65, 1.75, 0], 5.8, 0.12, 0.08);
    const material = engine.addSurfaceMaterial({
      color: [0.64, 0.64, 0.64], roughness: 0.72, metallic: 0,
    });
    const mesh = engine.addMesh(creatureMesh());
    const node = engine.addNode(null, [0, 0, 0], [0, 0, 0, 1], [1, 1, 1]);
    const instance = engine.addInstance(node, mesh, material);
    engine.setSkinPose(instance, bentPose());
    engine.setWireframe(variant.settings?.wire ? 'overlay' : 'off', instance);
    engine.setSelectionOutline(instance, variant.settings?.wire ? { color: [1, 0.55, 0.08] } : null);
  },
};
