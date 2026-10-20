// Basic example of @etcito/raa: one prims v0 asset built in code and spawned 4 times,
// 200 instanced primitives (25 of each kind), the orbit camera and one light. The engine
// comes ONLY from the packed package (./raa/index.js); ./common/ holds the shared page
// helpers for nomorechat's GPU client (window.nmcGpu.ops, see common/nmc-gpu.js).
import { loadRaa } from './raa/index.js';
import { nextFrame, runMeasured } from './common/run-loop.js';
import { registerNmcGpu } from './common/nmc-gpu.js';

const canvas = document.getElementById('c');
const gpuErrors = [];

// Some Chrome builds (M1 dogfood item 6) return SwiftShader for a core adapter and the
// hardware GPU only at featureLevel 'compatibility'. Prefer that hardware adapter. The
// engine's hardware flag still comes only from the real adapter info.
function installHardwareAdapterWrapper() {
  const gpu = navigator.gpu;
  if (!gpu || typeof gpu.requestAdapter !== 'function') return;
  const original = gpu.requestAdapter.bind(gpu);
  gpu.requestAdapter = async (options = {}) => {
    const core = await original(options);
    if (core && !(core.info && core.info.isFallbackAdapter)) {
      console.info('requestAdapter: using the core adapter');
      return core;
    }
    const compat = await original({ ...options, featureLevel: 'compatibility' });
    if (!compat || (compat.info && compat.info.isFallbackAdapter)) {
      console.info('requestAdapter: no hardware compatibility adapter, using the core result');
      return core;
    }
    // Clamp every required limit to what the compatibility adapter supports.
    const requestDevice = compat.requestDevice.bind(compat);
    compat.requestDevice = async (descriptor = {}) => {
      const required = descriptor.requiredLimits;
      const limits = {};
      for (const [name, value] of Object.entries(required ?? {})) {
        const max = compat.limits[name];
        if (value === undefined || typeof max !== 'number') {
          limits[name] = value;
        } else if (name.startsWith('min')) {
          // Alignment limits are minimums: the adapter value is the tightest allowed.
          limits[name] = Math.max(value, max);
        } else {
          limits[name] = Math.min(value, max);
        }
      }
      const device = await requestDevice({ ...descriptor, requiredLimits: limits });
      // WebGPU validation errors are asynchronous and do not throw from render().
      // Surface them through the client op instead of accepting a black capture.
      device.addEventListener('uncapturederror', (event) => {
        if (gpuErrors.length === 0) gpuErrors.push(event.error.message);
        console.error(event.error.message);
      });
      return device;
    };
    console.info('requestAdapter: using the compatibility adapter (core was missing or a fallback)');
    return compat;
  };
}

function showError(error) {
  const pre = document.createElement('pre');
  const code = error && error.code ? error.code : 'page.start_failed';
  let message = error && error.message ? error.message : String(error);
  // RaaError messages already start with "<code>: ".
  if (message.startsWith(`${code}: `)) message = message.slice(code.length + 2);
  pre.textContent = `${code}: ${message}`;
  document.body.appendChild(pre);
}

const IDENTITY_TRS = { r: [0, 0, 0, 1], s: [1, 1, 1] };
const at = (t) => ({ t, ...IDENTITY_TRS });

// A small lantern: box base, cylinder post, sphere globe, torus ring, cone cap. Every
// child is placed in the base's local space; the base sits on y = 0.
// Fixed LOD0 counts: box 12, cylinder 160, sphere 1520, cone 80, torus 1920 = 3692. A
// sphere or a torus alone exceeds the default budget (1500) at LOD0, so the asset
// declares the smallest explicit budget, as the loader's lamp sample does.
function lanternAsset() {
  const asset = {
    format: 'raa.prims',
    version: 0,
    id: 'lantern',
    name: 'Lantern',
    triangleBudget: 3692,
    nodes: [
      {
        id: 'base',
        kind: 'box',
        params: { halfExtents: [0.3, 0.06, 0.3] },
        transform: at([0, 0.06, 0]),
        material: 'metal',
        children: [
          {
            id: 'post',
            kind: 'cylinder',
            params: { radius: 0.05, halfHeight: 0.25 },
            transform: at([0, 0.31, 0]),
          },
          {
            id: 'globe',
            kind: 'sphere',
            params: { radius: 0.2 },
            transform: at([0, 0.76, 0]),
            material: 'glow',
          },
          {
            id: 'ring',
            kind: 'torus',
            params: { majorRadius: 0.22, minorRadius: 0.035 },
            transform: at([0, 0.76, 0]),
            material: 'trim',
          },
          {
            id: 'cap',
            kind: 'cone',
            params: { radius: 0.27, halfHeight: 0.14 },
            transform: at([0, 1.08, 0]),
            material: 'trim',
          },
        ],
      },
    ],
    // Keys sorted bytewise (canonical form).
    materials: {
      glow: { color: [1.0, 0.85, 0.45], roughness: 0.3, metallic: 0, emissive: [0.6, 0.4, 0.1] },
      metal: { color: [0.25, 0.27, 0.3], roughness: 0.45, metallic: 0.8, emissive: [0, 0, 0] },
      trim: { color: [0.8, 0.55, 0.2], roughness: 0.35, metallic: 0.9, emissive: [0, 0, 0] },
    },
    anchors: { top: at([0, 1.28, 0]) },
    // A torus has no derived collider (collision.torus_needs_override), so the asset
    // lists its colliders explicitly, in asset space; the ring becomes a flat cylinder.
    collision: {
      colliders: [
        { node: 'base', shape: 'cuboid', params: { halfExtents: [0.3, 0.06, 0.3] }, transform: at([0, 0.06, 0]) },
        { node: 'post', shape: 'cylinderY', params: { halfHeight: 0.25, radius: 0.05 }, transform: at([0, 0.37, 0]) },
        { node: 'globe', shape: 'ball', params: { radius: 0.2 }, transform: at([0, 0.82, 0]) },
        { node: 'ring', shape: 'cylinderY', params: { halfHeight: 0.035, radius: 0.255 }, transform: at([0, 0.82, 0]) },
        { node: 'cap', shape: 'coneY', params: { halfHeight: 0.14, radius: 0.27 }, transform: at([0, 1.14, 0]) },
      ],
    },
  };
  return JSON.stringify(asset, null, 2) + '\n';
}

// S3 reference parameter set A, one mesh per kind.
const SET_A = [
  ['sphere', { radius: 0.8 }],
  ['box', { halfExtents: [0.7, 0.6, 0.5] }],
  ['roundBox', { halfExtents: [0.7, 0.6, 0.5], radius: 0.15 }],
  ['capsule', { radius: 0.4, halfHeight: 0.5 }],
  ['cylinder', { radius: 0.6, halfHeight: 0.7 }],
  ['cone', { radius: 0.7, halfHeight: 0.8 }],
  ['torus', { majorRadius: 0.6, minorRadius: 0.22 }],
  ['ellipsoid', { radii: [0.9, 0.6, 0.5] }],
];

const MATERIALS = [
  { color: [0.85, 0.2, 0.2], roughness: 0.5, model: 'pbr' },
  { color: [0.2, 0.7, 0.3], model: 'flat' },
  { color: [0.2, 0.4, 0.9], roughness: 0.25, metallic: 0.6, model: 'pbr' },
  { color: [0.95, 0.8, 0.2], model: 'flat' },
  { color: [0.8, 0.8, 0.82], roughness: 0.2, metallic: 1, model: 'pbr' },
  { color: [0.6, 0.3, 0.8], model: 'flat' },
  { color: [0.9, 0.5, 0.2], roughness: 0.7, model: 'pbr' },
  { color: [0.2, 0.75, 0.8], roughness: 0.4, metallic: 0.3, model: 'pbr' },
];

const PER_KIND = 25;
const INSTANCE_SCALE = 0.3;
const GRID_SPACING = 0.9;
const GRID_HALF = 8; // 16 x 16 cells
const ASSET_OFFSET = 1.4;

// Grid cells around the 2x2 asset block, the central cells left free for the assets.
function gridCells(count) {
  const cells = [];
  for (let iz = -GRID_HALF; iz < GRID_HALF && cells.length < count; iz++) {
    for (let ix = -GRID_HALF; ix < GRID_HALF && cells.length < count; ix++) {
      const x = (ix + 0.5) * GRID_SPACING;
      const z = (iz + 0.5) * GRID_SPACING;
      if (Math.abs(x) < 2.6 && Math.abs(z) < 2.6) continue;
      cells.push([x, z]);
    }
  }
  return cells;
}

function buildScene(engine) {
  const lantern = engine.loadPrimsAsset(lanternAsset());
  const corners = [[-1, -1], [1, -1], [-1, 1], [1, 1]];
  corners.forEach(([sx, sz], i) => {
    const yaw = (i * Math.PI) / 4;
    const node = engine.addNode(
      null,
      [sx * ASSET_OFFSET, 0, sz * ASSET_OFFSET],
      [0, Math.sin(yaw / 2), 0, Math.cos(yaw / 2)],
      [1.2, 1.2, 1.2],
    );
    engine.spawnAsset(lantern.id, node);
  });

  const meshes = SET_A.map(([kind, params]) => engine.primitiveMesh(kind, params));
  const materials = MATERIALS.map((desc) => engine.addMaterial(desc));
  const total = meshes.length * PER_KIND;
  const cells = gridCells(total);
  const nodes = new Uint32Array(total);
  const trs = new Float32Array(total * 10);
  for (let i = 0; i < total; i++) {
    const node = engine.addNode(null);
    engine.addInstance(node, meshes[i % meshes.length], materials[(i * 3 + (i >> 3)) % materials.length]);
    nodes[i] = node;
    const [x, z] = cells[i];
    const yaw = i * 0.7;
    trs.set(
      [x, 0.3, z, 0, Math.sin(yaw / 2), 0, Math.cos(yaw / 2), INSTANCE_SCALE, INSTANCE_SCALE, INSTANCE_SCALE],
      i * 10,
    );
  }
  engine.setTransforms(nodes, trs);

  engine.setOrbit([0, 0.4, 0], 13, 0.6, 0.5);
  engine.setLight({ direction: [-0.4, -1, -0.3], color: [1, 0.96, 0.9], intensity: 3 });
  engine.setAmbient({ color: [0.55, 0.6, 0.7], intensity: 0.35 });
}

// Maps run-loop.js's demo interface onto the engine.
const YAW_RAD_PER_MS = 0.00015;

function demoAdapter(engine) {
  let lastNow = null;
  return {
    frame(now) {
      const dt = lastNow === null ? 0 : Math.min(Math.max(now - lastNow, 0), 100);
      lastNow = now;
      engine.orbit(dt * YAW_RAD_PER_MS, 0, 1);
      engine.render(now);
    },
    resize: (width, height) => engine.resize(width, height),
    probe_non_empty: (now) => engine.probeNonEmpty(now),
    stats: () => engine.stats(),
  };
}

// The ops are registered before the engine starts, so the client never finds them
// missing; they wait for the start to settle and reject with its error when it failed.
let opRunning = false;
const demoReady = main();

registerNmcGpu({
  run: async (args) => {
    const demo = await demoReady;
    if (opRunning) throw new Error('run: an op run is already in progress');
    opRunning = true;
    try {
      const result = await runMeasured(demo, canvas, args);
      if (gpuErrors.length) throw new Error(gpuErrors.join('\n'));
      return result;
    } finally {
      opRunning = false;
    }
  },
  // Draw one frame and read the canvas in the same task, before it is presented.
  screenshot: async () => {
    const demo = await demoReady;
    demo.frame(performance.now());
    return canvas.toDataURL('image/png').split(',')[1];
  },
});

async function main() {
  let demo;
  try {
    installHardwareAdapterWrapper();
    const raa = await loadRaa();
    const engine = await raa.createEngine(canvas, {});
    // Q3: the asset and the 200 instances are cheap to build, so they are added before
    // frame 0, which is then rendered at once (no wait for the next animation frame).
    // Fallback if the capture shows firstFrameMs > 500: render frame 0 on the empty
    // scene first and call buildScene after it, as the M5 demo did (not implemented).
    buildScene(engine);
    demo = demoAdapter(engine);
    demo.frame(performance.now());
  } catch (error) {
    console.error(error);
    showError(error);
    throw error;
  }

  // The idle loop pauses while an op run drives frames itself, so no frame is doubled.
  let loggedFrameError = false;
  const loop = (now) => {
    if (!opRunning) {
      try {
        demo.frame(now);
      } catch (error) {
        if (!loggedFrameError) console.error(error);
        loggedFrameError = true;
      }
    }
    nextFrame().then(loop);
  };
  nextFrame().then(loop);
  return demo;
}
