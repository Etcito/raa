import { setCanvasSize } from '../../../web/common/page-runner.js';
import { interactiveInterior } from './interactive-interior.js';
import { PATH_FIXTURE, digest, loadGalleryFit } from './gallery-path-model.js';

const scene = interactiveInterior.default;
const TIME = 1000;
const TOLERANCE = 0.025;
const CAMERAS = [
  { id: 'home', eye: [0.1, 1.65, -0.35], target: [0, 1.32, -5.8] },
  { id: 'left-aisle', eye: [-1.1, 1.55, -2.1], target: [0.3, 1.28, -6.6] },
];
const HOLDOUT = { id: 'right-aisle', eye: [1.1, 1.6, -1.45], target: [0, 1.4, -6.3] };
const layout = { exposure: 1, rowOrder: 'top-down', byteOrder: 'little-endian', jitter: [0, 0] };

async function create({ target, canvas, raa }, buildId, width, height) {
  if (!/Windows.+Chrome\//.test(target.navigator.userAgent) || typeof buildId !== 'string' || !buildId) {
    throw new Error('Gallery path evidence requires a pinned Windows Chrome build');
  }
  setCanvasSize(canvas, width, height);
  const engine = await raa.createEngine(canvas, { msaa: 1, allowSoftware: false, powerPreference: 'high-performance' });
  try {
    const { hardware, adapter } = engine.stats();
    if (!hardware || adapter?.isFallbackAdapter !== false) throw new Error('Expected hardware WebGPU adapter');
    const preset = scene.variants.find(variant => variant.id === 'performance');
    scene.build(engine, { raa }, {
      ...preset,
      effects: { ...preset.effects, gi: false, ssr: false, dof: false, bloom: false, 'moving-objects': false },
    });
    engine.setPassEnabled('taa', false);
    engine.setCameraJitter(false);
    engine.setAutoExposure({ enabled: false });
    engine.setToneMapping({ exposureEv: 0 });
    const facts = { hardware, adapter, browser: target.navigator.userAgent, devicePixelRatio: target.devicePixelRatio };
    const source = { kind: 'windows-render', buildId, fixture: PATH_FIXTURE, facts };
    return { engine, source, setPose({ eye, target }) { engine.setCamera({ ...preset.selection.camera, eye, target }); } };
  } catch (error) {
    engine.dispose();
    throw error;
  }
}

function unpack({ size, data }) {
  if (!Array.isArray(size) || size.length !== 2 || data.byteLength !== size[0] * size[1] * 32) {
    throw new Error('Invalid mesh path capture byte length');
  }
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const lanes = Float32Array.from({ length: data.byteLength / 4 }, (_, i) => view.getFloat32(i * 4, true));
  if (!lanes.every(Number.isFinite)) throw new Error('Non-finite gallery path sample');
  return { size, lanes };
}

async function plane(files, file, shape, lanes) {
  const bytes = new Uint8Array(lanes.length * 4);
  const view = new DataView(bytes.buffer);
  lanes.forEach((value, i) => view.setFloat32(i * 4, value, true));
  const sha256 = await digest(bytes);
  files.push({ name: file, blob: new Blob([bytes], { type: 'application/octet-stream' }) });
  return { file, shape, sha256, dtype: 'f32le', byteLength: bytes.byteLength };
}

async function capturePose(engine, files, pose, id, checkDeadline) {
  const base = 1 + id * 65536;
  engine.setMeshTracing({ pathMode: 'short', reference: false });
  const inputSeed = base;
  const input = unpack(await engine.captureMeshPathSample(TIME, inputSeed));
  const [width, height] = input.size;
  const count = width * height;
  const raw = new Float32Array(count * 12);
  let pixels = 0;
  for (let i = 0; i < count; i++) {
    const lanes = input.lanes.subarray(i * 8, i * 8 + 8);
    if (lanes[6] <= 0) continue;
    pixels++;
    raw.set(lanes, i * 12);
    raw[i * 12 + 8] = 1;
    raw[i * 12 + 11] = 1;
  }
  if (!pixels) throw new Error(`${pose.id}: no supported specular receivers`);

  engine.setMeshTracing({ pathMode: 'traced' });
  const batches = [[], []];
  const sums = [new Float64Array(count * 3), new Float64Array(count * 3)];
  let rmse;
  for (const samples of [16, 32, 64, 128]) {
    for (let batch = 0; batch < batches.length; batch++) {
      for (let j = batches[batch].length; j < samples; j++) {
        checkDeadline();
        const seed = base + (batch + 1) * 8192 + j;
        const next = unpack(await engine.captureMeshPathSample(TIME, seed));
        if (next.size[0] !== width || next.size[1] !== height) throw new Error('Reference extent changed');
        for (let i = 0; i < count; i++) {
          for (let c = 3; c < 8; c++) {
            if (next.lanes[i * 8 + c] !== input.lanes[i * 8 + c]) throw new Error(`${pose.id}: reference guides moved`);
          }
          for (let c = 0; c < 3; c++) sums[batch][i * 3 + c] += next.lanes[i * 8 + c];
        }
        batches[batch].push(seed);
      }
    }
    rmse = Math.sqrt(sums[0].reduce((sum, value, i) =>
      sum + ((value - sums[1][i]) / samples) ** 2, 0) / (3 * pixels));
    if (rmse <= TOLERANCE) break;
  }
  if (!(rmse <= TOLERANCE)) throw new Error(`${pose.id}: reference batch RMSE ${rmse} exceeds ${TOLERANCE}`);
  const teacher = sums[0].map((value, i) => (value + sums[1][i]) / (2 * batches[0].length));
  const signal = {
    inputSeeds: [inputSeed],
    input: await plane(files, `${pose.id}-input.bin`, [height, width, 12], raw),
    reference: await plane(files, `${pose.id}-reference.bin`, [height, width, 3], teacher),
    referenceMetadata: {
      estimator: 'ggx-first-lobe-mesh-path/1',
      settings: { primary: 'fixed pixel center', lobe: 'GGX', inputBounces: 1, referenceBounces: 2, mirrorBounces: 2 },
      seedBatches: batches, batchRmse: rmse, batchRmseTolerance: TOLERANCE,
    },
  };
  return { id: pose.id, frameId: id, viewId: 'gallery', size: input.size, timeMs: TIME, camera: pose,
    signals: { 'indirect-specular': signal } };
}

export async function captureGalleryPath(context, { buildId, remainingMs = 280000 } = {}) {
  const deadline = context.target.performance.now() + remainingMs;
  const checkDeadline = () => {
    if (context.target.performance.now() >= deadline) throw new Error('Gallery path capture timed out');
  };
  const { engine, source, setPose } = await create(context, buildId, 256, 144);
  try {
    const files = [], frames = [];
    for (const [id, pose] of CAMERAS.entries()) {
      setPose(pose);
      frames.push(await capturePose(engine, files, pose, id, checkDeadline));
    }
    return { json: { format: 'raa.optional-head-capture/1', source, ...layout, frames }, files };
  } finally {
    engine.dispose();
  }
}

export async function measureGalleryPath(context, { buildId, width = 640, height = 360 } = {}) {
  if (![width, height].every(value => Number.isInteger(value) && value > 0 && value <= 1920)) {
    throw new Error('Invalid gallery measurement extent');
  }
  const { canvas, target } = context;
  const { engine, source, setPose } = await create(context, buildId, width, height);
  const files = [], cases = [];
  try {
    const model = await loadGalleryFit();
    if (!model) throw new Error('Build and stage the fixed gallery fit before measuring');
    engine.setMeshPathModel(model.bytes);
    const readback = target.document.createElement('canvas');
    readback.width = width;
    readback.height = height;
    const copy = readback.getContext('2d');
    for (const pose of [CAMERAS[0], HOLDOUT]) {
      setPose(pose);
      for (const mode of ['short', 'traced', 'learned', 'reference']) {
        engine.setMeshTracing({
          pathMode: mode === 'reference' ? 'off' : mode, reflections: true, reference: mode === 'reference',
          pathBounces: 2, reflectionBounces: 2, confidenceThreshold: 0.55,
        });
        for (let i = 0; i < 8; i++) {
          engine.render(TIME);
          await engine.submittedWorkDone();
          engine.passGpuTimings();
        }
        const timings = new Map(), counts = new Map();
        for (let i = 0; i < 72 && timings.size < 20; i++) {
          engine.render(TIME);
          await engine.submittedWorkDone();
          const timing = engine.passGpuTimings();
          if (timing.status === 'unavailable') throw new Error(`GPU clock unavailable: ${timing.reason}`);
          const pass = timing.passes?.find(sample => sample.pass === 'trace.mesh');
          if (pass?.ms != null) timings.set(timing.frame, timing);
          const stats = engine.meshTracingStats();
          if (stats) counts.set(stats.frame, stats);
        }
        if (timings.size < 20 || !counts.size) throw new Error(`${pose.id}/${mode}: insufficient GPU samples`);
        const stats = [...counts.values()].at(-1);
        if (mode === 'learned' && (!stats.modelPixels || !stats.refreshPixels || !stats.diffuseMissingPixels
            || stats.referencePixels || stats.traced + stats.acceptedScreen !== stats.eligible
            || stats.primaryBvhPixels + stats.acceptedScreen !== stats.width * stats.height)) {
          throw new Error(`${pose.id}: learned path coverage or confidence gate failed`);
        }
        if (mode === 'reference' && (stats.modelPixels || stats.referenceSpp !== 4
            || stats.referencePixels !== stats.width * stats.height)) throw new Error('Reference used hybrid estimator');
        const file = `${pose.id}-${mode}.png`;
        await new Promise((resolve, reject) => target.requestAnimationFrame(() => {
          try {
            engine.render(TIME);
            copy.drawImage(canvas, 0, 0);
            files.push({ name: file, base64: readback.toDataURL('image/png').split(',')[1] });
            resolve();
          } catch (error) { reject(error); }
        }));
        await engine.submittedWorkDone();
        cases.push({ pose, mode, file, counts: [...counts.values()], timings: [...timings.values()] });
      }
    }
    return { json: { source, modelSha256: model.fit.weights.sha256, captureSha256: model.fit.captureSha256,
      size: [width, height], ...layout, cases }, files };
  } finally {
    engine.dispose();
  }
}
