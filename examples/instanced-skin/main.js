// Instanced skinning counts: N copies (1..200) of one skinned mesh, each with its own
// animation phase, and per frame the structural counts the engine reports: draw calls
// (one lit pass; Engine.stats().drawCalls) and the skin pass's compute dispatches and
// palette uploads (Engine.skinStats()). Its only purpose is to show that the draw count
// does not grow with N. No timing is measured.
//
// Mesh: the instanced character's saved .mesh (S24 wire format) when `?mesh=<url>` (or
// ./penguin.body.mesh next to this page) loads; otherwise a 12-joint synthetic capsule.
// Publish alongside raa/ (dist/bricks/raa or dist/package/raa) and common/ (web/common).
import { loadRaa } from './raa/index.js';
import { nextFrame } from './common/run-loop.js';
import { registerNmcGpu } from './common/nmc-gpu.js';

const canvas = document.getElementById('c');
const slider = document.getElementById('n');
const nOut = document.getElementById('nOut');
const live = document.getElementById('live');
const table = document.getElementById('table');
const errorText = document.getElementById('error');
const sourceText = document.getElementById('source');
const SWEEP = [1, 2, 5, 10, 20, 60, 100, 150, 200];

// ---- mesh sources ----

/** Decodes the S24 .mesh wire format (meshWire encoding). */
function decodeMeshWire(bytes) {
  if (bytes[0] !== 1) throw new Error('unsupported mesh format');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const length = view.getUint32(1);
  const count = view.getUint32(5);
  const buffers = [];
  let offset = 9 + length;
  for (let i = 0; i < count; i++) {
    const size = view.getUint32(offset);
    offset += 4;
    buffers.push(bytes.slice(offset, offset + size).buffer);
    offset += size;
  }
  const types = { Float32Array, Float64Array, Uint32Array, Uint8Array, Uint16Array, Int32Array };
  return JSON.parse(new TextDecoder().decode(bytes.subarray(9, 9 + length)), (_key, v) => {
    if (v && typeof v === 'object' && '$buffer' in v) return buffers[v.$buffer];
    if (v && typeof v === 'object' && '$array' in v) {
      const Type = types[v.$array];
      if (!Type) throw new Error(`unsupported mesh array ${v.$array}`);
      return new Type(buffers[v.buffer], v.offset, v.length / Type.BYTES_PER_ELEMENT);
    }
    return v;
  });
}

/** The first skinned part of a decoded .mesh, its part meshes concatenated. */
function penguinFromWire(asset) {
  for (const group of asset.data.groups) {
    for (const part of group.parts) {
      if (!part.skin) continue;
      const parts = part.meshes;
      const vertices = parts.reduce((n, m) => n + m.positions.length / 3, 0);
      const indices = parts.reduce((n, m) => n + m.indices.length, 0);
      const out = {
        positions: new Float32Array(vertices * 3),
        normals: new Float32Array(vertices * 3),
        indices: new Uint32Array(indices),
        parts: [],
        joints: new Uint16Array(part.skin.joints),
        weights: new Float32Array(part.skin.weights),
        label: `penguin .mesh (${vertices} vertices, ${indices / 3} triangles, ${parts.length} materials)`,
      };
      let v = 0;
      let i = 0;
      for (const m of parts) {
        out.positions.set(m.positions, v * 3);
        out.normals.set(m.normals, v * 3);
        for (let k = 0; k < m.indices.length; k++) out.indices[i + k] = m.indices[k] + v;
        out.parts.push({ first: i, count: m.indices.length, name: m.material });
        v += m.positions.length / 3;
        i += m.indices.length;
      }
      if (out.joints.length !== vertices * 4) throw new Error('skin does not cover the part');
      return out;
    }
  }
  throw new Error('no skinned part in the .mesh');
}

/** 12-joint capsule: one joint per height band, weights blended between neighbours. */
function syntheticCapsule() {
  const JOINTS = 12;
  const RINGS = 48;
  const SEG = 24;
  const positions = [];
  const normals = [];
  const joints = [];
  const weights = [];
  const indices = [];
  for (let r = 0; r <= RINGS; r++) {
    const t = r / RINGS;
    const y = t * 1.2;
    const band = t * (JOINTS - 1);
    const j0 = Math.min(Math.floor(band), JOINTS - 2);
    const w1 = band - j0;
    const radius = 0.18 + 0.06 * Math.sin(t * Math.PI);
    for (let s = 0; s <= SEG; s++) {
      const a = (s / SEG) * Math.PI * 2;
      positions.push(radius * Math.cos(a), y, radius * Math.sin(a));
      normals.push(Math.cos(a), 0, Math.sin(a));
      joints.push(j0, j0 + 1, 0, 0);
      weights.push(1 - w1, w1, 0, 0);
    }
  }
  const row = SEG + 1;
  for (let r = 0; r < RINGS; r++) {
    for (let s = 0; s < SEG; s++) {
      const a = r * row + s;
      const b = a + row;
      indices.push(a, b, a + 1, a + 1, b, b + 1);
    }
  }
  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    indices: new Uint32Array(indices),
    parts: [{ first: 0, count: indices.length, name: 'capsule' }],
    joints: new Uint16Array(joints),
    weights: new Float32Array(weights),
    label: `synthetic 12-joint capsule (${positions.length / 3} vertices)`,
  };
}

async function loadMesh() {
  const url = new URLSearchParams(location.search).get('mesh') ?? './penguin.body.mesh';
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return penguinFromWire(decodeMeshWire(new Uint8Array(await response.arrayBuffer())));
  } catch (error) {
    console.info(`instanced-skin: ${url} not loadable (${error.message}); using the capsule`);
    return syntheticCapsule();
  }
}

// ---- poses ----

/** Joint pivots: the weighted centroid of the vertices each joint influences. */
function pivots(mesh, jointCount) {
  const sum = new Float64Array(jointCount * 4);
  for (let v = 0; v < mesh.joints.length / 4; v++) {
    for (let k = 0; k < 4; k++) {
      const w = mesh.weights[v * 4 + k];
      if (w <= 0) continue;
      const j = mesh.joints[v * 4 + k];
      sum[j * 4] += w * mesh.positions[v * 3];
      sum[j * 4 + 1] += w * mesh.positions[v * 3 + 1];
      sum[j * 4 + 2] += w * mesh.positions[v * 3 + 2];
      sum[j * 4 + 3] += w;
    }
  }
  const out = [];
  for (let j = 0; j < jointCount; j++) {
    const w = sum[j * 4 + 3] || 1;
    out.push([sum[j * 4] / w, sum[j * 4 + 1] / w, sum[j * 4 + 2] / w]);
  }
  return out;
}

/** Column-major T(p) * Rz(a) * T(-p), written at `out[o..o+16]`. */
function rotateAbout(out, o, p, a) {
  const c = Math.cos(a);
  const s = Math.sin(a);
  out[o] = c; out[o + 1] = s; out[o + 2] = 0; out[o + 3] = 0;
  out[o + 4] = -s; out[o + 5] = c; out[o + 6] = 0; out[o + 7] = 0;
  out[o + 8] = 0; out[o + 9] = 0; out[o + 10] = 1; out[o + 11] = 0;
  out[o + 12] = p[0] - (c * p[0] - s * p[1]);
  out[o + 13] = p[1] - (s * p[0] + c * p[1]);
  out[o + 14] = 0;
  out[o + 15] = 1;
}

// ---- scene ----

function makeScene(engine, mesh) {
  const jointCount = 1 + mesh.joints.reduce((m, j) => Math.max(m, j), 0);
  const palette = [[0.9, 0.9, 0.95], [0.95, 0.75, 0.3], [0.2, 0.22, 0.3], [0.85, 0.4, 0.2]];
  const subMeshes = mesh.parts.map((p, i) => ({
    first: p.first,
    count: p.count,
    material: engine.addMaterial({ color: palette[i % palette.length], roughness: 0.6, model: 'pbr' }),
  }));
  const meshId = engine.addMesh({
    positions: mesh.positions,
    normals: mesh.normals,
    indices: mesh.indices,
    subMeshes,
    joints: mesh.joints,
    weights: mesh.weights,
  });
  const pivotList = pivots(mesh, jointCount);
  const nodes = [];
  const instances = [];
  let matrices = new Float32Array(0);
  let handles = new Uint32Array(0);

  function setCount(n) {
    while (nodes.length > n) {
      engine.removeNode(nodes.pop());
      instances.pop();
    }
    while (nodes.length < n) {
      const i = nodes.length;
      const side = Math.ceil(Math.sqrt(200));
      const x = ((i % side) - side / 2) * 0.9;
      const z = (Math.floor(i / side) - side / 2) * 0.9;
      const node = engine.addNode(null, [x, 0, z]);
      nodes.push(node);
      instances.push(engine.addMeshInstance(node, meshId));
    }
    handles = Uint32Array.from(instances);
    matrices = new Float32Array(n * jointCount * 16);
  }

  function pose(nowMs) {
    const t = nowMs / 1000;
    for (let i = 0; i < instances.length; i++) {
      const phase = i * 0.37;
      for (let j = 0; j < jointCount; j++) {
        const a = j === 0 ? 0 : 0.35 * Math.sin(t * 2.4 + phase + j * 0.8);
        rotateAbout(matrices, (i * jointCount + j) * 16, pivotList[j], a);
      }
    }
    if (instances.length) engine.setSkinPoses(handles, matrices);
  }

  return { setCount, pose, jointCount, get count() { return instances.length; } };
}

function counts(engine, n) {
  const stats = engine.stats();
  const skin = engine.skinStats();
  return {
    instances: n,
    drawCalls: stats.drawCalls,
    computeDispatches: skin.dispatches,
    paletteWrites: skin.paletteWrites,
    posed: skin.posed,
  };
}

function format(row) {
  return `N=${String(row.instances).padStart(3)}  draws/pass=${row.drawCalls}  ` +
    `compute dispatches=${row.computeDispatches}  palette writes=${row.paletteWrites}  posed=${row.posed}`;
}

async function sweep(engine, scene) {
  const rows = [];
  for (const n of SWEEP) {
    scene.setCount(n);
    // Two frames: the first registers the new instances, the second draws them posed.
    for (let f = 0; f < 2; f++) {
      const now = performance.now();
      scene.pose(now);
      engine.render(now);
      await nextFrame();
    }
    rows.push(counts(engine, n));
  }
  scene.setCount(Number(slider.value));
  table.textContent = ['Sweep (structural counts, no timing):', ...rows.map(format)].join('\n');
  return rows;
}

let ready;
registerNmcGpu({
  run: async () => {
    const { engine, scene, label } = await ready;
    return { mesh: label, rows: await sweep(engine, scene) };
  },
  screenshot: async () => {
    const { engine, scene } = await ready;
    const now = performance.now();
    scene.pose(now);
    engine.render(now);
    return canvas.toDataURL('image/png').split(',')[1];
  },
});

ready = (async () => {
  const raa = await loadRaa();
  const engine = await raa.createEngine(canvas, {});
  const mesh = await loadMesh();
  sourceText.textContent = mesh.label;
  const scene = makeScene(engine, mesh);
  scene.setCount(Number(slider.value));
  engine.setOrbit([0, 0.4, 0], 12, 0.6, 0.45);
  engine.setLight({ direction: [-0.4, -1, -0.3], color: [1, 0.96, 0.9], intensity: 3 });
  engine.setAmbient({ color: [0.55, 0.6, 0.7], intensity: 0.4 });
  slider.addEventListener('input', () => {
    nOut.textContent = slider.value;
    scene.setCount(Number(slider.value));
  });
  document.getElementById('sweep').addEventListener('click', () => {
    sweep(engine, scene).catch((e) => { errorText.textContent = String(e?.message || e); });
  });
  const loop = (now) => {
    try {
      scene.pose(now);
      engine.render(now);
      live.textContent = `${format(counts(engine, scene.count))}  (joints per instance: ${scene.jointCount})`;
    } catch (error) {
      errorText.textContent = String(error?.message || error);
      return;
    }
    nextFrame().then(loop);
  };
  nextFrame().then(loop);
  return { engine, scene, label: mesh.label };
})();
ready.catch((error) => {
  console.error(error);
  errorText.textContent = `${error?.code ?? 'page.start_failed'}: ${error?.message ?? error}`;
});
