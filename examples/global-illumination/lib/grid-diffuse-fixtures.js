import lighting, { CORNELL_LIGHT_INTENSITY } from '../scenes/lighting.js';
import { box } from './content.js';
import { digest } from './gallery-path-model.js';
import { interactiveInterior } from './interactive-interior.js';

const unitScale = [1, 1, 1];
const identity = [0, 0, 0, 1];
const fixedHz = 60;
const framesPerTrajectory = 18;

function metalReceiver(engine, center, half, roughness) {
  const mesh = box(center, half);
  const material = engine.addSurfaceMaterial({ color: [0.85, 0.85, 0.85], roughness, metallic: 1 });
  const handle = engine.addMesh({ ...mesh, subMeshes: [{ first: 0, count: mesh.indices.length, material }] });
  return engine.addMeshInstance(engine.addNode(null, [0, 0, 0], identity, unitScale), handle);
}

const roomFixtures = {
  cornell: {
    split: 'train',
    geometry: 'Existing Cornell box, its moving short block, and two glossy receiver plates',
    offsets: { camera: [0.7, -0.2, 0.9], cut: [-0.45, 0.14, 0.65], block: [0.54, 0, 0.26] },
    async build(engine) {
      const variant = lighting.variants.find(item => item.id === 'cornell-box-on');
      const state = await lighting.build(engine, {}, { ...variant, movingBlocker: true });
      metalReceiver(engine, [2.42, 0.015, 1.45], [0.93, 0.012, 0.94], 0.075);
      metalReceiver(engine, [4.08, 0.025, 1.6], [0.64, 0.02, 0.95], 0.23);
      return {
        camera: variant.selection.camera,
        apply(timeMs, progress, panels) {
          const offset = roomFixtures.cornell.offsets.block.map(value => value * progress);
          engine.setTransform(state.blockNode, offset, identity, unitScale);
          engine.updateLight(state.areaLight, { intensity: panels ? CORNELL_LIGHT_INTENSITY : 0 });
        },
      };
    },
  },
  'held-interior': {
    split: 'test',
    geometry: 'Interactive gallery interior with skylight closed and original glossy receivers',
    offsets: { camera: [-0.5, 0.11, -0.82], cut: [0.93, -0.12, -0.75] },
    async build(engine) {
      const scene = interactiveInterior.default;
      const variant = scene.variants.find(item => item.id === 'performance');
      const state = scene.build(engine, {}, {
        ...variant, skylightBlocked: true,
        effects: { ...variant.effects, ssr: false, dof: false, bloom: false, 'moving-objects': true },
      });
      return {
        camera: variant.selection.camera,
        apply(timeMs, progress, panels) {
          scene.frame(engine, timeMs, variant, state);
          scene.setOption(engine, state, 'area-lights', panels);
        },
      };
    },
  },
};

const add = (point, delta, scale = 1) => point.map((value, i) => value + delta[i] * scale);

export function gridDiffuseTrajectory(scene, camera) {
  const { split, offsets } = roomFixtures[scene];
  const seedBase = split === 'train' ? 32768 : 8388608;
  return Array.from({ length: framesPerTrajectory }, (_, index) => {
    const moving = index >= 3 && index <= 8;
    const progress = Math.max(0, Math.min(1, (index - 2) / 6));
    const cut = index >= 11;
    const paneOffset = cut ? offsets.cut : offsets.camera;
    const amount = cut ? 1 : progress;
    const condition = index >= 13 ? 'lights-off'
      : index === 11 ? 'camera-cut' : moving ? 'camera-motion' : index < 3 ? 'static' : 'camera-stopped';
    return {
      id: `${scene}-${String(index).padStart(2, '0')}`,
      index, condition,
      timeMs: 1000 + index * (1000 / fixedHz),
      seed: seedBase + index * 4096,
      reset: index === 0 || index === 11,
      objectProgress: progress,
      panels: index < 13,
      camera: {
        ...camera,
        eye: add(camera.eye, paneOffset, amount),
        target: add(camera.target, paneOffset, amount * 0.4),
      },
    };
  });
}

function recordGeometry(engine, records) {
  return new Proxy(engine, { get(target, key) {
    const method = Reflect.get(target, key);
    if (key === 'addMesh') return mesh => {
      records.push({
        positions: mesh.positions,
        indices: mesh.indices,
        normals: mesh.normals,
      });
      return method.call(target, mesh);
    };
    return typeof method === 'function' ? method.bind(target) : method;
  } });
}

async function geometrySha256(meshes) {
  const records = [];
  for (const mesh of meshes) {
    const attributes = {};
    for (const name of ['positions', 'indices', 'normals']) {
      const values = mesh[name];
      if (ArrayBuffer.isView(values)) {
        const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
        attributes[name] = [values.length, await digest(bytes)];
      }
    }
    records.push(attributes);
  }
  if (!records.some(record => record.positions && record.indices)) throw new Error('Fixture has no mesh geometry');
  return digest(new TextEncoder().encode(JSON.stringify(records)));
}

export async function createGridDiffuseFixture(engine, id) {
  const fixture = roomFixtures[id];
  if (!fixture) throw new Error(`Unknown grid diffuse fixture ${id}`);
  const geometry = [];
  const runtime = await fixture.build(recordGeometry(engine, geometry));
  const tape = gridDiffuseTrajectory(id, runtime.camera);
  return {
    scene: { id, split: fixture.split, geometry: fixture.geometry, geometrySha256: await geometrySha256(geometry) },
    fixedHz,
    tape,
    apply(frame) {
      engine.setCamera(frame.camera);
      runtime.apply(frame.timeMs, frame.objectProgress, frame.panels);
    },
  };
}
