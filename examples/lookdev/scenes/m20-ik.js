// S14: importing metadata creates no scene, GPU resources, timers or DOM state.
// Visual IK over the m18 skin API. The lookdev page supplies the animation brick as
// lib.anim ({ loadRig(json), ik: { footPlacement, handReach, lookAt } }).
import { uvSphere, sweepTube, roundedBox, grid, merge } from '../lib/geometry.js';

const TARGET = [0.42, 1.38, 0.38];
const HITS = [
  { point: [0.14, 0, 0], normal: [0, 1, 0] },
  { point: [-0.14, 0.3, 0], normal: [0, 1, 0] },
];
const DARK = [0.02, 0.02, 0.02, 1];

const BONES = [
  ['hips', null, [0, 0.95, 0]],
  ['spine', 'hips', [0, 0.15, 0]],
  ['chest', 'spine', [0, 0.3, 0]],
  ['neck', 'chest', [0, 0.22, 0]],
  ['head', 'neck', [0, 0.1, 0]],
  ['upperArm.l', 'chest', [0.22, 0.16, 0]],
  ['lowerArm.l', 'upperArm.l', [0, -0.3, 0]],
  ['hand.l', 'lowerArm.l', [0, -0.27, 0]],
  ['upperArm.r', 'chest', [-0.22, 0.16, 0]],
  ['lowerArm.r', 'upperArm.r', [0, -0.3, 0]],
  ['hand.r', 'lowerArm.r', [0, -0.27, 0]],
  ['upperLeg.l', 'hips', [0.14, 0, 0]],
  ['lowerLeg.l', 'upperLeg.l', [0, -0.5, 0]],
  ['foot.l', 'lowerLeg.l', [0, -0.45, 0]],
  ['upperLeg.r', 'hips', [-0.14, 0, 0]],
  ['lowerLeg.r', 'upperLeg.r', [0, -0.5, 0]],
  ['foot.r', 'lowerLeg.r', [0, -0.45, 0]],
];
const BONE = Object.fromEntries(BONES.map(([name], i) => [name, i]));
// Ten floats per bone (t, r, s), rest local to the parent; the rig root sits at the origin.
const REST = Float32Array.from(BONES.flatMap(([, , t]) => [...t, 0, 0, 0, 1, 1, 1, 1]));

const RIG = JSON.stringify({
  format: 'raa.rig',
  version: 1,
  name: 'm20-biped',
  bones: BONES.map(([name, parent, t]) => ({ name, parent, rest: { t, r: [0, 0, 0, 1], s: [1, 1, 1] } })),
  roles: Object.fromEntries(BONES.map(([name]) => [name, name])),
  ikChains: [
    { name: 'legL', solver: 'twoBone', bones: ['upperLeg.l', 'lowerLeg.l', 'foot.l'], pole: [0, 0, 1], iterations: 16 },
    { name: 'legR', solver: 'twoBone', bones: ['upperLeg.r', 'lowerLeg.r', 'foot.r'], pole: [0, 0, 1], iterations: 16 },
    { name: 'armL', solver: 'twoBone', bones: ['upperArm.l', 'lowerArm.l', 'hand.l'], pole: [0, -1, 0], iterations: 16 },
  ],
});

function ellipsoid(radii, centre, rgba) {
  const s = uvSphere(1, 24, 16);
  const n = s.positions.length / 3;
  const positions = new Float32Array(s.positions.length);
  const normals = new Float32Array(s.normals.length);
  for (let i = 0; i < n; i++) {
    const nrm = [0, 1, 2].map((k) => s.normals[3 * i + k] / radii[k]);
    const len = Math.hypot(...nrm);
    for (let k = 0; k < 3; k++) {
      positions[3 * i + k] = s.positions[3 * i + k] * radii[k] + centre[k];
      normals[3 * i + k] = nrm[k] / len;
    }
  }
  const colors = rgba ? new Float32Array(4 * n).map((_, i) => rgba[i % 4]) : undefined;
  return { positions, normals, indices: s.indices, uvs: s.uvs, colors };
}

// Every part is rigidly bound to one bone; the merged mesh keeps one bone per vertex.
function figureMesh() {
  const parts = [];
  const add = (bone, desc) => parts.push({ bone: BONE[bone], desc });
  const ball = (bone, c, r) => add(bone, ellipsoid([r, r, r], c));
  const tube = (bone, a, b, r) => add(bone, sweepTube([a, b], r, 14));

  add('hips', ellipsoid([0.18, 0.12, 0.13], [0, 0.95, 0]));
  add('chest', ellipsoid([0.2, 0.26, 0.14], [0, 1.33, 0]));
  tube('neck', [0, 1.6, 0], [0, 1.74, 0], 0.05);
  add('head', ellipsoid([0.1, 0.12, 0.11], [0, 1.83, 0]));
  add('head', ellipsoid([0.022, 0.022, 0.012], [-0.04, 1.85, 0.105], DARK));
  add('head', ellipsoid([0.022, 0.022, 0.012], [0.04, 1.85, 0.105], DARK));

  for (const [side, sx] of [['l', 1], ['r', -1]]) {
    const hip = [0.14 * sx, 0.95, 0];
    const knee = [0.14 * sx, 0.45, 0];
    const ankle = [0.14 * sx, 0, 0];
    const shoulder = [0.22 * sx, 1.56, 0];
    const elbow = [0.22 * sx, 1.26, 0];
    const wrist = [0.22 * sx, 0.99, 0];

    ball(`upperLeg.${side}`, hip, 0.09);
    tube(`upperLeg.${side}`, hip, knee, 0.085);
    ball(`lowerLeg.${side}`, knee, 0.07);
    tube(`lowerLeg.${side}`, knee, [knee[0], 0.05, 0], 0.065);
    add(`foot.${side}`, ellipsoid([0.07, 0.05, 0.16], [ankle[0], 0.05, 0.08]));

    ball(`upperArm.${side}`, shoulder, 0.07);
    tube(`upperArm.${side}`, shoulder, elbow, 0.055);
    ball(`lowerArm.${side}`, elbow, 0.05);
    tube(`lowerArm.${side}`, elbow, wrist, 0.05);
    add(`hand.${side}`, ellipsoid([0.05, 0.1, 0.035], [wrist[0], 0.9, 0]));
  }

  const { subMeshes, ...mesh } = merge(parts.map((p) => p.desc));
  const count = mesh.positions.length / 3;
  const joints = new Uint16Array(count * 4);
  const weights = new Float32Array(count * 4);
  let v = 0;
  for (const { bone, desc } of parts) {
    const n = desc.positions.length / 3;
    for (let i = v; i < v + n; i++) {
      joints[i * 4] = bone;
      weights[i * 4] = 1;
    }
    v += n;
  }
  return { ...mesh, joints, weights };
}

// ik-on: feet on their surfaces, the reaching hand on the target, the head looking at it.
// ik-off: the same figure with the solver bypassed (handReach at weight 0 returns the rest pose).
function skinMatrices(anim, rig, solve) {
  if (!solve) return anim.ik.handReach(rig, REST, 'armL', TARGET, 0).skin;
  const feet = anim.ik.footPlacement(rig, REST, ['legL', 'legR'], HITS);
  const reach = anim.ik.handReach(rig, feet, 'armL', TARGET, 1);
  return anim.ik.lookAt(rig, reach, 'head', TARGET, [0, 0, 1], 70).skin;
}

const SUBJECT = 'A neutral grey, rigid-segment biped mannequin with two small dark eyes stands '
  + 'across a plain grey step on a flat dark grey floor, under soft directional light, against '
  + 'a dark blue-grey background. A small glossy orange sphere floats in front of the figure at '
  + 'chest height. ';

export default {
  id: 'm20-ik',
  milestone: 'm20',
  title: 'Visual IK: foot placement, hand reach and look-at on a skinned biped',
  engine: { msaa: 4, toneMapper: 'agx', exposureEv: 0 },
  variants: [
    {
      id: 'ik-on',
      description: SUBJECT
        + 'The right foot rests flat on the top surface of the step and the left foot is flat on '
        + 'the floor, with the right knee bent to reach up. The left arm is raised forward and its '
        + 'hand touches the orange sphere, and the head turns to look at that same sphere. Both '
        + 'feet sit on their surfaces with no clipping and no gap.',
      references: [],
      judge: true,
      settings: { ik: true },
    },
    {
      id: 'ik-off',
      description: SUBJECT
        + 'The same figure, geometry, camera, lighting and orange sphere, with IK disabled: the '
        + 'authored rest pose is shown, the arms hang at its sides, the head faces straight forward, '
        + 'and the right foot is pushed into the front of the step with no contact at its top surface.',
      references: [],
      judge: false,
      settings: { ik: false },
    },
  ],

  async build(engine, lib, variant) {
    engine.setBackground([0.23, 0.26, 0.30]);
    engine.setLight({ direction: [-0.6, -1, -0.8], color: [1, 0.97, 0.93], intensity: 3 });
    engine.setAmbient({ color: [0.8, 0.85, 1], intensity: 0.55 });
    engine.setOrbit([0, 1.0, 0], 5.0, 0.35, 0.3);

    const at = (t) => engine.addNode(null, t, [0, 0, 0, 1], [1, 1, 1]);
    const floor = engine.addSurfaceMaterial({ color: [0.3, 0.32, 0.35], roughness: 0.8, metallic: 0 });
    const stone = engine.addSurfaceMaterial({ color: [0.55, 0.57, 0.6], roughness: 0.7, metallic: 0 });
    const body = engine.addSurfaceMaterial({ color: [0.64, 0.64, 0.64], roughness: 0.72, metallic: 0 });
    const marker = engine.addSurfaceMaterial({
      color: [1, 0.45, 0.05], roughness: 0.3, metallic: 0, emissive: [0.8, 0.3, 0.02],
    });

    engine.addInstance(at([0, 0, 0]), engine.addMesh(grid(8, 8, 1, 1)), floor);
    engine.addInstance(at([-0.3, 0.15, 0]), engine.addMesh(roundedBox([0.3, 0.15, 0.5], 0.02, 2)), stone);
    engine.addInstance(at(TARGET), engine.addMesh(uvSphere(0.07, 24, 16)), marker);
    const figure = engine.addInstance(at([0, 0, 0]), engine.addMesh(figureMesh()), body);

    const anim = lib?.anim;
    if (!anim) throw new Error('m20-ik: lib.anim is required');
    const rig = anim.loadRig(RIG);
    try {
      engine.setSkinPose(figure, skinMatrices(anim, rig, variant.settings?.ik === true));
    } finally {
      rig.free();
    }
  },
};
