import { setCanvasSize } from '../../../web/common/page-runner.js';
import {
  applyBaselineEvent, assertNeuralBaselineReady, BASELINE_FEATURES, BASELINE_FIXTURE,
  BASELINE_FRAME_MS, BASELINE_WARMUP_FRAMES, buildNeuralBaseline,
} from './neural-baseline.js';

const SIZE = [65, 49]; // Odd extent exercises padded readback on every plane.
const EVENT = 16;
const FRAMES = [0, 14, 15, 16, 17, 18];
const BATCH_SIZES = [32, 64, 128, 256];
export const REFERENCE_RMSE = 0.02;

const PLANES = [
  ['raw', 'rgba16float', 8], ['depth', 'r32float', 4],
  ['normal', 'rgba16float', 8], ['motion', 'rgba16float', 8],
];
const CHANNELS = [
  'currentS', 'currentU', 'previousS', 'previousU', 'normalX', 'normalY', 'normalZ',
  'metricDepth', 'motionU', 'motionV', 'historyValid', 'lightChange',
];

function half(view, index) {
  const bits = view.getUint16(index * 2, true);
  const exponent = (bits >> 10) & 31;
  const fraction = bits & 1023;
  if (exponent === 31) throw new Error('non-finite native FP16 capture');
  const magnitude = exponent === 0 ? fraction * 2 ** -24 : (1024 + fraction) * 2 ** (exponent - 25);
  return bits & 0x8000 ? -magnitude : magnitude;
}

function planeViews(sample, descriptors) {
  const [width, height] = sample.size;
  if (sample.planes.length !== descriptors.length) throw new Error('unexpected plane count');
  return descriptors.map(([name, , stride], i) => {
    const bytes = sample.planes[i];
    if (bytes.byteLength !== width * height * stride) throw new Error(`invalid tight ${name} readback`);
    return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  });
}

export function areaInput(sample, previous, reset) {
  const [width, height] = sample.size;
  const [raw, depth, normal, motion] = planeViews(sample, PLANES);
  const history = !reset && previous && previous.lightIndex === sample.lightIndex &&
    previous.lightGeneration === sample.lightGeneration ? planeViews(previous, PLANES) : null;
  const bytes = new Uint8Array(width * height * 12 * 4);
  const out = new DataView(bytes.buffer);
  for (let p = 0; p < width * height; p++) {
    const z = depth.getFloat32(p * 4, true);
    if (!Number.isFinite(z) || z < 0) throw new Error('invalid metric depth');
    if (z === 0) continue;
    const lane = p * 4;
    const du = half(motion, lane), dv = half(motion, lane + 1);
    const x = Math.floor(p % width + 0.5 - du * width);
    const y = Math.floor(Math.floor(p / width) + 0.5 - dv * height);
    const values = [
      half(raw, lane), half(raw, lane + 1), 0, 0, half(normal, lane),
      half(normal, lane + 1), half(normal, lane + 2), z, du, dv, 0, 0,
    ];
    if (history && half(motion, lane + 2) < 0.5 && x >= 0 && x < width && y >= 0 && y < height) {
      const q = y * width + x;
      const previousDepth = history[1].getFloat32(q * 4, true);
      if (previousDepth > 0 && Math.abs(previousDepth - z) <= 0.05 * z) {
        values[2] = half(history[0], q * 4);
        values[3] = half(history[0], q * 4 + 1);
        values[10] = 1;
      }
    }
    for (let c = 0; c < values.length; c++) out.setFloat32((p * 12 + c) * 4, values[c], true);
  }
  return bytes;
}

function accumulate(sums, sample) {
  const [raw] = planeViews(sample, PLANES.slice(0, 1));
  for (let p = 0; p < sums.length / 2; p++) {
    const s = half(raw, p * 4), u = half(raw, p * 4 + 1);
    if (s < 0 || s > u || u < 0) throw new Error('invalid weighted visibility pair');
    sums[p * 2] += s;
    sums[p * 2 + 1] += u;
  }
}

const ratio = (s, u) => u > 0 ? s / u : 1;

export async function areaReference(size, sampleForSeed, batchSizes = BATCH_SIZES) {
  const count = size[0] * size[1];
  const a = new Float64Array(count * 2), b = new Float64Array(count * 2);
  const seeds = [[], []];
  let error;
  for (const total of batchSizes) {
    for (let i = seeds[0].length; i < total; i++) {
      for (const [batch, sums] of [a, b].entries()) {
        // Each disjoint stream sees all four strata for every consecutive four samples.
        const seed = (batch + 1) * 4096 + i;
        const sample = await sampleForSeed(seed);
        if (sample.size[0] !== size[0] || sample.size[1] !== size[1]) throw new Error('teacher extent changed');
        accumulate(sums, sample);
        seeds[batch].push(seed);
      }
    }
    let squared = 0;
    for (let p = 0; p < a.length; p += 2) {
      squared += (ratio(a[p], a[p + 1]) - ratio(b[p], b[p + 1])) ** 2;
    }
    error = Math.sqrt(squared / count);
    if (error <= REFERENCE_RMSE) {
      if (!a.some((value, index) => index % 2 === 1 && value > 0) ||
          !b.some((value, index) => index % 2 === 1 && value > 0)) {
        throw new Error('area teacher returned no weighted receivers');
      }
      const bytes = new Uint8Array(count * 4), result = new DataView(bytes.buffer);
      for (let p = 0; p < a.length; p += 2) {
        result.setFloat32(p * 2, ratio(a[p] + b[p], a[p + 1] + b[p + 1]), true);
      }
      return { bytes, seedBatches: seeds, rmse: error };
    }
  }
  throw new Error(`area teacher batch RMSE ${error} exceeds ${REFERENCE_RMSE}`);
}

async function binary(files, file, bytes, dtype, shape) {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  const sha256 = Array.from(digest, byte => byte.toString(16).padStart(2, '0')).join('');
  files.push({ name: file, blob: new Blob([bytes], { type: 'application/octet-stream' }) });
  return { file, sha256, byteLength: bytes.byteLength, dtype, shape };
}

const timeOf = frame => (BASELINE_WARMUP_FRAMES + frame) * BASELINE_FRAME_MS;
const sameBytes = (a, b) => a.length === b.length && a.every((value, index) => value === b[index]);

export async function captureAreaPilot({ target, canvas, raa }, { buildId, remainingMs = 280000 } = {}) {
  if (!/Windows/.test(target.navigator.userAgent)) throw new Error('area pilot capture requires Windows Chrome');
  if (typeof buildId !== 'string' || !buildId || !Number.isFinite(remainingMs) || remainingMs <= 0)
    throw new Error('invalid area pilot capture options');
  const deadline = target.performance.now() + remainingMs;
  const checkDeadline = () => {
    if (target.performance.now() >= deadline) throw new Error('area pilot capture deadline exceeded');
  };
  const files = [], frames = [];
  const [width, height] = SIZE;
  let facts;
  let engine;

  async function makeEngine() {
    engine = await raa.createEngine(canvas, { msaa: 1, powerPreference: 'high-performance', allowSoftware: false });
    const { hardware, adapter } = engine.stats();
    if (!hardware) throw new Error('area pilot requires a hardware adapter');
    facts = { adapter, hardware, browser: target.navigator.userAgent, devicePixelRatio: target.devicePixelRatio };
    const state = await buildNeuralBaseline(engine);
    engine.setCameraJitter(false);
    for (let i = 0; i < BASELINE_WARMUP_FRAMES; i++) {
      checkDeadline();
      engine.render(i * BASELINE_FRAME_MS);
      if (i % 16 === 15) await new Promise(resolve => target.setTimeout(resolve, 0));
    }
    assertNeuralBaselineReady(engine);
    return state;
  }

  async function capture(time, seed, guides) {
    checkDeadline();
    const sample = await engine.captureAreaSample(time, seed, guides);
    if (sample.size[0] !== width || sample.size[1] !== height) throw new Error('unexpected area pilot extent');
    return sample;
  }

  setCanvasSize(canvas, width, height);
  try {
    const state = await makeEngine();
    let previous = null;
    for (let frameId = 0; frameId <= FRAMES.at(-1); frameId++) {
      const moved = frameId === EVENT;
      if (moved) applyBaselineEvent(engine, state, 'blocker-step');
      const timeMs = timeOf(frameId), inputSeed = frameId;
      const sample = await capture(timeMs, inputSeed, true);
      if (FRAMES.includes(frameId)) {
        const id = `main-${frameId}`;
        const native = {};
        for (const [i, [name, dtype]] of PLANES.entries()) {
          native[name] = await binary(files, `${id}-${name}.bin`, sample.planes[i], dtype,
            dtype === 'r32float' ? [height, width] : [height, width, 4]);
        }
        const input = await binary(files, `${id}-input.bin`, areaInput(sample, previous, moved), 'f32le', [height, width, 12]);
        frames.push({
          id, viewId: 'main', frameId, timeMs, size: SIZE, activeRect: [0, 0, width, height],
          eventId: moved ? 'blocker-step' : null, inputSeed, input, native,
          lightSlot: 0, lightIndex: sample.lightIndex, lightGeneration: sample.lightGeneration,
        });
      }
      previous = sample;
    }
    engine.dispose();
    engine = null;

    // Replay in an isolated renderer so repeated teacher samples cannot advance input history.
    const teacher = await makeEngine();
    let beforeEvent;
    for (const frame of frames) {
      if (frame.eventId) {
        applyBaselineEvent(engine, teacher, frame.eventId);
        // The reference sees the new pose after transient volume rebuild work finishes.
        let settled = false;
        for (let i = 0; i < BASELINE_WARMUP_FRAMES; i++) {
          checkDeadline();
          engine.render(frame.timeMs);
          if (engine.giStats().pendingRebuild === false) {
            settled = true;
            break;
          }
        }
        if (!settled) throw new Error('area teacher volume did not settle after the blocker step');
      }
      const atPose = async seed => {
        const sample = await capture(frame.timeMs, seed, false);
        if (frame.lightIndex !== sample.lightIndex || frame.lightGeneration !== sample.lightGeneration)
          throw new Error('teacher light identity differs from input');
        return sample;
      };
      const first = await atPose(4096);
      const repeat = await atPose(4096);
      if (!sameBytes(first.planes[0], repeat.planes[0])) throw new Error(`teacher pose is unstable at ${frame.id}`);
      if (frame.frameId === EVENT - 1) beforeEvent = first.planes[0];
      if (frame.eventId && sameBytes(beforeEvent, first.planes[0])) {
        throw new Error('blocker step did not change the area trace at a fixed seed');
      }
      frame.identitySeed = 4096;
      const reference = await areaReference(SIZE, atPose);
      frame.referenceSeeds = reference.seedBatches.flat();
      frame.referenceBatches = reference.seedBatches;
      frame.referenceBatchRmse = reference.rmse;
      frame.reference = await binary(files, `${frame.id}-reference.bin`, reference.bytes, 'f32le', [height, width]);
    }
    return {
      json: {
        format: 'raa.area-pilot-capture/1', buildId, fixture: BASELINE_FIXTURE, facts, features: BASELINE_FEATURES,
        channels: CHANNELS, normalizationScales: [1, 1, 1, 1, 1, 1, 1, 0.125, 8, 8, 1, 1],
        jitter: [0, 0], exposure: 1, rowOrder: 'top-down', byteOrder: 'little-endian',
        history: 'nearest previous raw sample with motion/depth reject and blocker-step reset',
        teacher: { estimator: 'stratified-area-same-estimator', batchRmseTolerance: REFERENCE_RMSE },
        frames,
      },
      files,
    };
  } finally {
    engine?.dispose();
  }
}
