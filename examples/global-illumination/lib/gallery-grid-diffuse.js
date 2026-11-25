import { setCanvasSize } from '../../../web/common/page-runner.js';
import { digest } from './gallery-path-model.js';
import { createGridDiffuseFixture } from './grid-diffuse-fixtures.js';

const defaultSize = [256, 144];
const studentModes = new Set(['published-legacy', 'legacy', 'analytic-directional', 'learned']);
const packedChannels = 128;
const rowLimit = 128;
const samplesPerBank = 32;

function selectGraph(engine, mode) {
  const frozen = engine.setFrozenRenderGraph('spherical_gi');
  if (!frozen.ok) throw new Error(`Spherical GI starter refused: ${JSON.stringify(frozen.issues)}`);
  const graph = engine.renderGraph();
  const node = graph.nodes.find(item => item.type === 'mesh.reflection');
  if (!node) throw new Error('Spherical GI starter has no geometry reflection producer');
  if (mode === 'published-legacy') return node.id;
  const spherical = graph.nodes.find(item => item.type === 'pass.gi.spherical');
  if (!spherical) throw new Error('Spherical GI starter has no spherical lookup');
  node.type = 'mesh.grid-diffuse';
  node.params = { ...node.params, inheritGi: false, diffuseMode: mode };
  graph.links = graph.links.filter(link => !['sphericalLookup', 'volumeAtlas'].some(port => link.to === node.id + '.' + port));
  graph.links.push({ from: spherical.id + '.lookup', to: node.id + '.sphericalLookup' });
  const result = engine.setRenderGraphSelection(graph);
  if (!result.ok) throw new Error(`Grid diffuse graph refused: ${JSON.stringify(result.issues)}`);
  return node.id;
}

function uniqueSamples(samples) {
  return [...new Map(samples.filter(sample => Number.isInteger(sample.frame))
    .map(sample => [sample.frame, sample])).values()];
}

function summarizeTimings(samples) {
  const passes = new Map();
  for (const sample of uniqueSamples(samples)) {
    for (const pass of sample.passes ?? []) {
      if (!Number.isFinite(pass.ms)) continue;
      const name = pass.pass;
      const values = passes.get(name) ?? [];
      values.push(pass.ms);
      passes.set(name, values);
    }
  }
  return Object.fromEntries([...passes].map(([name, values]) => {
    values.sort((a, b) => a - b);
    const rank = percentile => values[Math.max(0, Math.ceil(values.length * percentile) - 1)];
    return [name, { samples: values.length, p50Ms: rank(0.50), p95Ms: rank(0.95) }];
  }));
}

async function save(files, file, data, shape, dtype) {
  const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const record = { file, shape, dtype, byteLength: bytes.byteLength, sha256: await digest(bytes) };
  files.push({ name: file, blob: new Blob([bytes], { type: 'application/octet-stream' }) });
  return record;
}

async function render(engine, timeMs) {
  engine.render(timeMs);
  await engine.submittedWorkDone();
}

async function currentCapture(target, engine, nodeId, frameId, deadline) {
  while (target.performance.now() < deadline) {
    const capture = engine.meshGridDiffuseCapture(nodeId);
    if (capture?.frame === frameId) return capture;
    if (capture?.frame > frameId) throw new Error(`Grid readback skipped renderer frame ${frameId}`);
    await new Promise(resolve => target.requestAnimationFrame(resolve));
  }
  throw new Error(`Grid readback timed out at frame ${frameId}`);
}

function coverage(raw) {
  const counts = Array(9).fill(0);
  const float = new Float32Array(raw.buffer, raw.byteOffset, raw.length);
  let fallback = 0;
  for (let row = 0; row < raw.length; row += packedChannels) {
    const neighbors = raw[row + 11];
    if (neighbors > 8) throw new Error('Grid capture returned an invalid neighbor count');
    counts[neighbors]++;
    fallback += Number(float[row + 15] <= 0);
  }
  return { queriedRows: raw.length / packedChannels, neighbors: counts, missing: counts[0],
    oneNeighbor: counts[1], fallback };
}

export async function captureGridDiffuseReplay({ target, engine, fixture }, {
  buildId, mode = 'published-legacy', stride = 64, checkpointSha256, checkpointBase64,
  captureRows = true, size = defaultSize, remainingMs = 280000,
} = {}) {
  const sceneId = fixture.scene.id;
  if (!/Windows.+Chrome\//.test(target.navigator.userAgent) || !buildId || !studentModes.has(mode)
      || !Number.isInteger(stride) || stride < 1 || stride > 4096
      || !Array.isArray(size) || size.length !== 2
      || size.some(value => !Number.isInteger(value) || value < 64 || value > 1920)
      || typeof captureRows !== 'boolean') {
    throw new Error('Grid diffuse capture requires a pinned Windows Chrome build and valid experiment settings');
  }
  if ((fixture.scene.split !== 'train' || mode === 'learned') && !/^[0-9a-f]{64}$/.test(checkpointSha256 ?? '')) {
    throw new Error('Reserve held-out evaluation until the frozen checkpoint hash is recorded');
  }

  const deadline = target.performance.now() + remainingMs;
  const files = [];
  try {
    const { hardware, adapter } = engine.stats();
    if (!hardware || adapter?.isFallbackAdapter !== false) throw new Error('Expected native WebGPU');
    if (mode === 'learned') {
      const weights = Uint8Array.from(target.atob(checkpointBase64 ?? ''), byte => byte.charCodeAt(0));
      if (weights.byteLength !== 81 * 4 || await digest(weights) !== checkpointSha256) {
        throw new Error('Frozen checkpoint bytes do not match their SHA-256');
      }
      engine.setMeshGridDiffuseModel(weights);
    }
    const nodeId = selectGraph(engine, mode);
    engine.setCameraJitter(false);
    engine.setAutoExposure({ enabled: false });
    engine.setToneMapping({ exposureEv: 0 });
    fixture.apply(fixture.tape[0]);

    let warmedFrames = 0;
    for (; warmedFrames < 128; warmedFrames++) {
      if (target.performance.now() >= deadline) throw new Error('Grid diffuse warmup timed out');
      await render(engine, fixture.tape[0].timeMs);
      const grid = engine.sphericalGiStats();
      if (warmedFrames >= 7 && grid?.readyCells > 0 && grid.pendingFaces === 0 && grid.inFlightFaces === 0) break;
    }
    if (!engine.sphericalGiStats()?.readyCells) throw new Error('No occupied spherical cell was ready');

    const frames = [];
    const measuredMode = mode !== 'published-legacy' && captureRows;
    const timingSamples = [];
    for (const frame of fixture.tape) {
      if (target.performance.now() >= deadline) throw new Error('Grid diffuse replay timed out');
      fixture.apply(frame);
      if (measuredMode) engine.setMeshGridDiffuseCapture(frame.seed, stride);
      const beforeSubmit = target.performance.now();
      await render(engine, frame.timeMs);
      const workDoneMs = target.performance.now();
      const frameId = engine.stats().frames - 1;
      const timing = engine.passGpuTimings();
      timingSamples.push(timing);
      const trace = engine.meshTracingStats(nodeId);
      const grid = engine.sphericalGiStats();
      let rowFields = {};
      if (measuredMode) {
        const capture = await currentCapture(target, engine, nodeId, frameId,
          Math.min(deadline, target.performance.now() + 12000));
        const readbackWallMs = target.performance.now() - workDoneMs;
        if (capture.seed !== frame.seed || capture.stride !== stride || !(capture.rows instanceof Uint32Array)
            || capture.rows.length % packedChannels || !Number.isInteger(capture.candidates)) {
          throw new Error(`Invalid or unmatched grid capture for ${frame.id}`);
        }
        const rows = capture.rows.length / packedChannels;
        if (capture.candidates > rowLimit || capture.candidates !== rows) {
          throw new Error(`Truncated grid rows at ${frame.id}: ${capture.candidates} candidates, ${rows} retained. Raise stride and replay all routes.`);
        }
        const indices = Uint32Array.from({ length: rows }, (_, i) => capture.rows[i * packedChannels + 31]);
        if (indices.some(index => index >= capture.width * capture.height) || new Set(indices).size !== rows) {
          throw new Error(`Invalid first-hit pixel identity for ${frame.id}`);
        }
        const order = Array.from(indices.keys()).sort((a, b) => indices[a] - indices[b]);
        // Sort the raw GPU bits by primary pixel so asynchronous GPU append order is immaterial.
        const sorted = new Uint32Array(capture.rows.length);
        for (const [i, item] of order.entries()) {
          sorted.set(capture.rows.subarray(item * packedChannels, (item + 1) * packedChannels), i * packedChannels);
        }
        const raw = await save(files, `${frame.id}-rows.bin`, sorted, [rows, packedChannels], 'u32le');
        const rasterIndices = await save(files, `${frame.id}-pixels.bin`,
          Uint32Array.from(order, item => indices[item]), [rows], 'u32le');
        const hitIds = Uint32Array.from({ length: rows * 5 }, (_, i) => {
          const row = Math.floor(i / 5) * packedChannels;
          const lane = i % 5;
          return sorted[row + (lane === 4 ? 31 : lane)];
        });
        const firstHitSha256 = await digest(new Uint8Array(hitIds.buffer));
        rowFields = { rows, file: raw.file, sha256: raw.sha256, readbackWallMs, rasterIndices, planes: { rows: raw, rasterIndices },
          coverage: coverage(sorted), width: capture.width, height: capture.height,
          reference: { samplesPerBank, independent: true, firstHitSha256 },
          ...(mode === 'learned' && { learnedWeightsSha256: checkpointSha256 }) };
      }
      frames.push({ ...frame, frameId, trajectory: sceneId + '-camera-light',
        lighting: { panels: frame.panels, channelMask: engine.sphericalGi().channelMask },
        trace, grid, renderWallMs: workDoneMs - beforeSubmit, gpuTimings: timing, ...rowFields });
    }
    if (!frames.some(frame => frame.trace?.firstHitHits > 0)) throw new Error('No genuine secondary hit was exercised');

    const tapeSha256 = await digest(new TextEncoder().encode(JSON.stringify(fixture.tape)));
    return { json: {
      format: measuredMode ? 'raa.spherical-diffuse-capture/1' : 'raa.grid-diffuse-native-replay/1',
      units: 'scene-linear irradiance/pi before receiver albedo',
      source: { kind: 'windows-render', platform: 'win32', adapter: adapter.device, build: buildId, buildId,
        facts: { hardware, adapter, browser: target.navigator.userAgent } },
      scene: fixture.scene, tapeSha256, fixedHz: fixture.fixedHz, mode, checkpointSha256,
      ...(mode === 'learned' && { learnedWeightsSha256: checkpointSha256 }), stride, sceneSize: size, captureRows: measuredMode,
      warmedFrames: warmedFrames + 1, grid: engine.sphericalGi(), timing: summarizeTimings(timingSamples), frames,
    }, files };
  } finally {
    if (mode !== 'published-legacy' && captureRows) engine.setMeshGridDiffuseCapture(0, 0);
  }
}

async function captureGalleryMode({ target, canvas, raa }, args) {
  setCanvasSize(canvas, ...(args.size ?? defaultSize));
  const engine = await raa.createEngine(canvas, { msaa: 1, allowSoftware: false, powerPreference: 'high-performance' });
  try {
    const fixture = await createGridDiffuseFixture(engine, args.scene);
    return await captureGridDiffuseReplay({ target, engine, fixture }, args);
  } finally {
    engine.dispose();
  }
}

export async function captureGalleryGridDiffuse(context, args = {}) {
  if (!args.timingStudy) return captureGalleryMode(context, args);
  const modes = ['legacy', 'analytic-directional', 'learned', 'learned', 'analytic-directional', 'legacy'];
  const deadline = context.target.performance.now() + (args.remainingMs ?? 280000);
  const runs = [];
  for (const mode of modes) {
    const remainingMs = deadline - context.target.performance.now();
    if (remainingMs <= 0) throw new Error('Matched grid diffuse timing study timed out');
    const { json } = await captureGalleryMode(context, { ...args, mode, captureRows: false, remainingMs });
    if (runs.some(run => run.tapeSha256 !== json.tapeSha256 || run.scene.geometrySha256 !== json.scene.geometrySha256)) {
      throw new Error('Timing conditions changed across cache routes');
    }
    runs.push(json);
  }
  const first = runs[0];
  return {
    json: { format: 'raa.grid-diffuse-gpu-timing/1', source: first.source, scene: first.scene,
      checkpointSha256: args.checkpointSha256, tapeSha256: first.tapeSha256, size: first.sceneSize,
      condition: 'Warmed interleaved real-frame renderer GPU timestamps, teacher capture disabled',
      simulationHz: first.fixedHz, frames: [], runs },
    files: [],
  };
}
