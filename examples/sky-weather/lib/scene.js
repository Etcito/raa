// One metre-valued layout drives both the real primitives and selection bounds.
const objects = [
  { name: 'ground', kind: 'box', t: [0, -.1, -80], half: [200, .1, 200], color: [.48, .46, .4], roughness: .9 },
  { name: 'matte-reference', kind: 'sphere', t: [-2.1, 1, 0], radius: 1, color: [.66, .26, .14], roughness: 1 },
  { name: 'reflective-sphere', kind: 'sphere', t: [1.35, 1.1, .4], radius: 1.1, color: [.94, .96, .99], roughness: .08, metallic: 1 },
  { name: 'distant-left', kind: 'box', t: [-17, 4, -48], half: [5, 4, 4], color: [.4, .46, .5], roughness: .95 },
  { name: 'distant-centre', kind: 'box', t: [0, 3, -62], half: [7, 3, 5], color: [.47, .5, .52], roughness: .95 },
  { name: 'distant-right', kind: 'box', t: [20, 6, -76], half: [6, 6, 5], color: [.36, .42, .47], roughness: .95 },
];

/** @param {import('../../../packages/raa/index.js').CameraDesc} camera */
export function sampleSelection(camera) {
  if (!camera || !(camera.position ?? camera.eye) || !camera.target) throw new Error('sample camera requires eye and target');
  if (camera.projection && camera.projection !== 'perspective') throw new Error('sample camera requires perspective projection');
  if (camera.up && camera.up.some((v, i) => v !== [0, 1, 0][i])) throw new Error('sample camera requires +Y up');
  return {
    camera: {
      eye: [...(camera.position ?? camera.eye)], target: [...camera.target],
      fovYDeg: camera.fovYDeg ?? 60, near: camera.near ?? .1, far: camera.far ?? 200, aspect: 16 / 9,
    },
    subjects: objects.map(object => {
      const half = object.half ?? [object.radius, object.radius, object.radius];
      return { name: object.name, min: object.t.map((v, i) => v - half[i]), max: object.t.map((v, i) => v + half[i]) };
    }),
  };
}

// Geometry and camera only: sky, time and weather remain the scene producer's settings.
export async function buildSkySample(engine, lib, camera) {
  const selected = sampleSelection(camera).camera;
  const { aspect, ...desc } = selected;
  engine.setCamera(desc);
  for (const object of objects) {
    const params = object.kind === 'sphere' ? { radius: object.radius } : { halfExtents: [...object.half] };
    const mesh = engine.primitiveMesh(object.kind, params, object.kind === 'sphere' ? 64 : 24);
    const material = engine.addMaterial({
      model: 'pbr', color: [...object.color], roughness: object.roughness, metallic: object.metallic ?? 0,
    });
    const node = engine.addNode(null, [...object.t], [0, 0, 0, 1], [1, 1, 1]);
    engine.addInstance(node, mesh, material);
  }
}
