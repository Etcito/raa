import { interactiveInterior } from './interactive-interior.js';
import { digest } from './gallery-path-model.js';
import { loadGalleryNeural } from './gallery-neural.js';
import { reflectionGraph, reflectionInputs, reflectionTeacher } from './gallery-reflection-teacher.js';
import { jointTrajectory } from './gallery-joint-trajectories.js';

const scene = interactiveInterior.default;
const variant = scene.variants.find(row => row.id === 'performance');
const sceneSize = [160, 90], displaySize = [320, 180];
const frameCount = 12;
const spp = 32;
const tolerance = 0.045;
const guideChannels = ['normalX', 'normalY', 'normalZ', 'log1pDepthOver8', 'roughness',
  'motionU', 'motionV', 'historyValid', 'reactive', 'jitterXpx', 'jitterYpx', 'geometryValid'];
const maskChannels = ['disocclusion', 'outsideHistory', 'reflectionOffscreen', 'lightEvent'];
const camera = variant.selection.camera;
const identity = [0, 0, 0, 1], scale = [1, 1, 1];

function tape(heldout) {
  return Array.from({ length: frameCount }, (_, i) => {
    const u = i / (frameCount - 1);
    const timeMs = 1000 + i * (1000 / 60);
    return {
      timeMs,
      camera: heldout ? {
        eye: [1.23 - 1.15 * u, 1.55 + 0.2 * u, -1.4 - 0.55 * u],
        target: [0.18 - 0.6 * u, 1.37, -6.45],
      } : {
        eye: [0.13 - 1.05 * u, 1.63 + 0.05 * u, -0.37 - 0.72 * u],
        target: [0.55 - 0.9 * u, 1.37 - 0.1 * u, -5.8],
      },
      sphere: heldout ? [-0.9 + 0.55 * u, 0.56, -3.3 - 0.1 * u] : [-1.15 + 0.4 * u, 0.56, -3.2],
      reflector: heldout ? [2.2 - 0.33 * u, 1.72 - 0.3 * u, -0.96] : [2.26 - 0.52 * u, 1.29 + 0.57 * u, -1.12],
      panelScale: heldout ? (i < 7 ? [0.55, 0.42] : [0.55, 1.25]) : (i < 5 ? [1, 1] : [0.12, 1]),
      event: i === (heldout ? 7 : 5) ? (heldout ? 'cool-panel-on' : 'warm-panel-dim') : null,
    };
  });
}

async function create(target, raa, canvas, size, heldout, reference, graph, unjittered = reference) {
  [canvas.width, canvas.height] = size;
  const engine = await raa.createEngine(canvas, { msaa: 1, allowSoftware: false, powerPreference: 'high-performance' });
  try {
    const { hardware, adapter } = engine.stats();
    if (!hardware || adapter?.isFallbackAdapter !== false) throw new Error('Sequence capture requires hardware WebGPU');
    const state = scene.build(engine, { raa }, { ...variant, skylightBlocked: heldout,
      vertexWallColors: heldout, effects: { ...variant.effects, gi: !reference, 'moving-objects': false,
        'geometry-reflections': !reference && !graph, ssr: false, bloom: false, dof: false } });
    engine.setPassEnabled('taa', false);
    engine.setPassEnabled('ao.gtao', false);
    engine.setCameraJitter(!unjittered);
    engine.setAutoExposure({ enabled: false });
    engine.setToneMapping({ exposureEv: 0 });
    engine.setEnvironmentLighting({ diffuse: 0.65, specular: 0 });
    if (graph) {
      const selection = engine.setRenderGraphSelection(graph);
      if (!selection.ok) throw new Error(`SSR training graph: ${JSON.stringify(selection.issues)}`);
    }
    if (reference) engine.setMeshTracing({ reference: true, reflections: true, pathMode: 'off',
      lightingOutput: 'combined', pathBounces: 2, reflectionBounces: 2 });
    return { engine, facts: { hardware, adapter, browser: target.navigator.userAgent,
      devicePixelRatio: target.devicePixelRatio }, apply(frame, previous) {
      engine.setCamera({ ...camera, ...frame.camera });
      engine.setTransform(state.sphereNode, frame.sphere, identity, scale);
      engine.setTransform(state.entryNode, frame.reflector, identity, scale);
      for (const [i, fixture] of state.fixtures.entries()) {
        if (!previous || frame.panelScale[i] !== previous.panelScale[i]) {
          engine.updateLight(fixture.light, { intensity: fixture.intensity * frame.panelScale[i] });
        }
      }
    } };
  } catch (error) {
    engine.dispose();
    throw error;
  }
}

function floats(bytes, count, half) {
  if (!bytes || bytes.byteLength !== count * (half ? 2 : 4)) throw new Error('Unexpected gallery tap byte length');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const lanes = Float32Array.from({ length: count }, (_, i) => {
    if (!half) return view.getFloat32(i * 4, true);
    const bits = view.getUint16(i * 2, true);
    const exponent = (bits >>> 10) & 31, mantissa = bits & 1023;
    if (exponent === 31) return NaN;
    return (bits & 0x8000 ? -1 : 1) * (exponent ? (1024 + mantissa) * 2 ** (exponent - 25) : mantissa * 2 ** -24);
  });
  if (!lanes.every(Number.isFinite)) throw new Error('Nonfinite gallery capture lanes');
  return lanes;
}

const unit = v => v.map(x => x / Math.hypot(...v));
const subtract = (a, b) => a.map((x, i) => x - b[i]);
const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
function basis(pose) {
  const front = unit(subtract(pose.target, pose.eye));
  const right = unit(cross(front, [0, 1, 0]));
  return { front, right, up: cross(right, front) };
}

function realSignals(taps, frame, previous, reflectionMeta) {
  const [w, h] = sceneSize, n = w * h;
  const rawHdr = floats(taps.hdr, n * 4, true), rawDepth = floats(taps.depth, n, false);
  const normal = floats(taps.normal, n * 4, true), rawMotion = floats(taps.motion, n * 4, true);
  const reflectionHit = floats(taps.reflectionHit ?? taps.reflectionTrace, n * 4, true);
  const jitterPx = [taps.jitterNdc[0] * w / 2, -taps.jitterNdc[1] * h / 2];
  const hdr = new Float32Array(n * 3), depth = new Float32Array(n), motion = new Float32Array(n * 3);
  const masks = new Float32Array(n * 4), guide = new Float32Array(n * 12);
  const counts = { geometric: 0, historyValid: 0, moved: 0, disocclusion: 0,
    outsideHistory: 0, reflectionOffscreen: 0, reactive: 0 };
  const current = basis(frame.camera);
  const previousBasis = previous && basis(previous.camera);
  const tanHalf = Math.tan(camera.fovYDeg * Math.PI / 360);
  for (let p = 0; p < n; p++) {
    const lane = p * 4;
    const z = rawDepth[p];
    const valid = z < 1;
    const metric = valid ? camera.near * camera.far / (camera.far - z * (camera.far - camera.near)) : 0;
    const x = p % w, y = Math.floor(p / w);
    const [du, dv, reject, mirror] = rawMotion.subarray(lane, lane + 4);
    const offscreen = reflectionMeta
      ? reflectionMeta[lane + 1] === 0 && reflectionMeta[lane + 2] === -2
      : reflectionHit[lane + 3] < -1.5;
    let accepted = false, outside = false;
    if (previous && valid && reject < 0.5) {
      const u = (x + 0.5 - jitterPx[0]) / w, v = (y + 0.5 - jitterPx[1]) / h;
      const srcX = Math.floor((u - du) * w + previous.jitterPx[0]);
      const srcY = Math.floor((v - dv) * h + previous.jitterPx[1]);
      outside = srcX < 0 || srcX >= w || srcY < 0 || srcY >= h;
      if (!outside) {
        const prior = previous.depth[srcX + srcY * w];
        const world = frame.camera.eye.map((c, i) => c + metric * (current.front[i]
          + current.right[i] * (2 * u - 1) * tanHalf * w / h + current.up[i] * (1 - 2 * v) * tanHalf));
        const expected = dot(subtract(world, previous.camera.eye), previousBasis.front);
        accepted = prior > 0 && expected > 0 && Math.abs(prior - expected) <= Math.max(0.04, expected * 0.025);
      }
    }
    const disocclusion = Boolean(previous && valid && !accepted && !outside);
    const reactive = valid && (frame.event !== null || (mirror > 0.5 && Boolean(previous)));
    hdr.set(rawHdr.subarray(lane, lane + 3), p * 3);
    depth[p] = metric;
    motion.set([du, dv, reject], p * 3);
    masks.set([disocclusion, outside, offscreen, valid && frame.event !== null], lane);
    guide.set([...(valid ? normal.subarray(lane, lane + 3) : [0, 0, 0]), Math.log1p(metric) / 8,
      valid ? normal[lane + 3] : 0, du, dv, accepted, reactive, ...jitterPx, valid], p * 12);
    counts.geometric += Number(valid);
    counts.historyValid += Number(accepted);
    counts.moved += Number(Math.abs(du) + Math.abs(dv) > 1e-4);
    counts.disocclusion += Number(disocclusion);
    counts.outsideHistory += Number(outside);
    counts.reflectionOffscreen += Number(offscreen);
    counts.reactive += Number(reactive);
  }
  return { hdr, guide, depth, motion, masks, reflectionHit, jitterPx, counts };
}

function hdrRgb(bytes, size) {
  const pixels = size[0] * size[1];
  const rgba = floats(bytes, pixels * 4, true);
  return Float32Array.from({ length: pixels * 3 }, (_, i) => rgba[Math.floor(i / 3) * 4 + i % 3]);
}

async function warmSequence(engine, timeMs) {
  for (let i = 0; i < 32; i++) {
    engine.render(timeMs);
    await engine.submittedWorkDone();
  }
}

async function referenceHdr(teacher, timeMs, base, checkTime) {
  const [w, h] = displaySize, n = w * h * 3;
  const sums = [new Float64Array(n), new Float64Array(n)], seeds = [[], []];
  let logRmse, linearRmse, batchCalls;
  for (batchCalls = 1; batchCalls <= 8; batchCalls++) {
    for (let batch = 0; batch < 2; batch++) {
      checkTime();
      const seed = base + batch * 256 + batchCalls;
      const sample = await teacher.captureTrainingReference(timeMs, seed, spp);
      if (sample.size.join() !== displaySize.join()) throw new Error('Reference tracer did not emit the display grid');
      const raw = floats(sample.data, w * h * 4, true);
      const sum = sums[batch];
      for (let p = 0; p < w * h; p++) {
        for (let c = 0; c < 3; c++) sum[p * 3 + c] += raw[p * 4 + c];
      }
      seeds[batch].push(seed);
    }
    if (batchCalls < 2) continue;
    let logError = 0, linearError = 0;
    for (let i = 0; i < n; i++) {
      const a = sums[0][i] / batchCalls, b = sums[1][i] / batchCalls;
      logError += (Math.log1p(Math.max(0, a)) - Math.log1p(Math.max(0, b))) ** 2;
      linearError += (a - b) ** 2;
    }
    logRmse = Math.sqrt(logError / n);
    linearRmse = Math.sqrt(linearError / n);
    if (logRmse <= tolerance) break;
  }
  if (!(logRmse <= tolerance)) throw new Error(`Independent HDR batches did not converge at ${timeMs} ms, log RMSE ${logRmse}`);
  const label = Float32Array.from(sums[0], (sum, i) => (sum + sums[1][i]) / (2 * batchCalls));
  return { label, batches: sums.map(sum => Float32Array.from(sum, v => v / batchCalls)), teacher: { sppPerBatch: spp * batchCalls, seeds, logRmse, linearRmse, tolerance } };
}

async function save(files, prefix, name, size, array) {
  const [w, h] = size, channels = array.length / (w * h);
  const file = `${prefix}-${name.toLowerCase()}.bin`;
  const bytes = new Uint8Array(array.length * 4), view = new DataView(bytes.buffer);
  array.forEach((value, i) => view.setFloat32(i * 4, value, true));
  files.push({ name: file, blob: new Blob([bytes], { type: 'application/octet-stream' }) });
  return { file, shape: [h, w, channels], dtype: 'f32le', byteLength: bytes.length, sha256: await digest(bytes) };
}

export async function captureGallerySequence({ target, canvas, raa }, { buildId, split = 'both', reflection = false, trajectory, remainingMs = 280000 } = {}) {
  if (!/Windows.+Chrome\//.test(target.navigator.userAgent) || !buildId || !['train', 'test', 'both'].includes(split)) {
    throw new Error('Sequence capture requires a pinned Windows Chrome build and a valid split');
  }
  const deadline = target.performance.now() + remainingMs;
  const checkTime = () => { if (target.performance.now() >= deadline) throw new Error('Gallery sequence timed out'); };
  const files = [], sequences = [];
  const joint = trajectory === undefined ? null : jointTrajectory(trajectory);
  reflection ||= Boolean(joint);
  const graph = reflection ? await reflectionGraph() : null;
  const selected = joint ? [joint] : ['train', 'test'].flatMap((partition, index) => {
    if (split !== 'both' && split !== partition) return [];
    const heldout = partition === 'test';
    return [{ partition, heldout, replay: tape(heldout), seedBase: 8192 + index * 1048576,
      id: heldout ? 'gallery-heldout' : 'gallery-overfit',
      groupId: heldout ? 'sealed-skylight-cool-event' : 'open-skylight-warm-event' }];
  });
  let facts;
  for (const { partition, heldout, replay, id, groupId, seedBase } of selected) {
    const student = await create(target, raa, canvas, sceneSize, heldout, false, graph);
    let teacher, raster;
    try {
      teacher = await create(target, raa, target.document.createElement('canvas'), displaySize.map(n => n * 2), heldout, true);
      if (joint) {
        raster = await create(target, raa, target.document.createElement('canvas'), displaySize, heldout, false, graph, true);
      }
      facts = student.facts;
      const frames = [], reflectionGuides = [];
      let previous;
      student.apply(replay[0]);
      // Let the default volume cache settle before the single contiguous recorded run.
      await warmSequence(student.engine, replay[0].timeMs);
      // Record student frames consecutively before independent teacher readbacks.
      for (const [i, frame] of replay.entries()) {
        checkTime();
        student.apply(frame, replay[i - 1]);
        const inputCutoffMs = target.performance.now();
        const frameKey = `${id}-${String(i).padStart(2, '0')}`;
        const timing = student.engine.renderInstrumented(frame.timeMs, {
          viewId: 'primary', trialId: id, presentId: frameKey, inputCutoffMs,
          ...(frame.event && { eventId: frame.event, eventMs: inputCutoffMs, acceptedMs: inputCutoffMs }),
        });
        if (timing.error || !Number.isInteger(timing.frameIndex)
            || (i && timing.frameIndex !== frames.at(-1).frameId + 1)) throw new Error('Nonconsecutive real capture');
        await student.engine.submittedWorkDone();
        const workDoneMs = target.performance.now();
        const taps = await student.engine.captureTrainingTaps(timing.frameIndex, Boolean(joint));
        const meta = reflection ? floats(taps.reflectionMeta, sceneSize[0] * sceneSize[1] * 4, false) : null;
        const signals = realSignals(taps, frame, previous, joint ? meta : null);
        const readbackDoneMs = target.performance.now();
        const gpuTimings = student.engine.passGpuTimings();
        const { jitterPx, counts } = signals;
        previous = { camera: frame.camera, depth: signals.depth, jitterPx };
        const planes = {};
        for (const name of ['hdr', 'guide', 'depth', 'motion', 'masks', 'reflectionHit']) {
          planes[name] = await save(files, frameKey, name, sceneSize, signals[name]);
        }
        if (reflection) {
          const [w, h] = sceneSize;
          const incident = floats(taps.reflectionIncident, w * h * 4, false);
          const guides = reflectionInputs(incident, meta, signals.reflectionHit, joint ? 0 : undefined);
          counts.reflectionAccepted = guides.accepted;
          reflectionGuides.push(guides);
          planes.reflection = await save(files, frameKey, 'reflection', sceneSize, guides.input);
          planes.reflectionMeta = await save(files, frameKey, 'reflectionmeta', sceneSize, meta);
          planes.reflectionIncident = await save(files, frameKey, 'reflectionincident', sceneSize, incident);
        }
        if (joint) {
          planes.compositeHdr = await save(files, frameKey, 'compositehdr', sceneSize, hdrRgb(taps.compositeHdr, sceneSize));
        }
        frames.push({ id: frameKey, frameId: timing.frameIndex, simulationTick: i, viewId: 'primary', reset: i === 0,
          ...frame, jitterPx, counts, timing: { ...timing, workDoneMs, readbackDoneMs, gpuTimings }, planes });
      }
      if (!frames.some(frame => frame.counts.moved > 300 && frame.counts.historyValid > 100
          && frame.counts.disocclusion > 20)
          || ((!reflection || joint) && !frames.some(frame => frame.counts.reflectionOffscreen > 20))
          || !frames.some(frame => frame.event && frame.counts.reactive > 100)
          || (joint && frames.filter(frame => frame.counts.reflectionAccepted > 100).length < 4)) {
        throw new Error(`${id} is missing an effective motion, disocclusion, offscreen reflection or lighting event`);
      }
      if (raster) {
        raster.apply(replay[0]);
        await warmSequence(raster.engine, replay[0].timeMs);
        for (const [i, frame] of frames.entries()) {
          checkTime();
          raster.apply(frame, replay[i - 1]);
          raster.engine.render(frame.timeMs);
          await raster.engine.submittedWorkDone();
          const realFrame = raster.engine.stats().frames - 1;
          const taps = await raster.engine.captureTrainingTaps(realFrame, true);
          if (taps.jitterNdc.some(value => value !== 0)) throw new Error('Raster teacher used camera jitter');
          frame.planes.rasterLabel = await save(files, frame.id, 'rasterlabel', displaySize,
            hdrRgb(taps.compositeHdr, displaySize));
          frame.rasterTeacher = { realFrame, timeMs: frame.timeMs, jitterNdc: taps.jitterNdc };
        }
      }
      for (const [i, frame] of frames.entries()) {
        teacher.apply(frame, replay[i - 1]);
        const reference = await referenceHdr(teacher.engine, frame.timeMs, seedBase + i * 1024, checkTime);
        frame.planes.label = await save(files, frame.id, 'label', displaySize, reference.label);
        frame.teacher = reference.teacher;
        if (reflection) {
          const { target: incident, ...stats } = reflectionTeacher(reflectionGuides[i], reference.batches,
            sceneSize, displaySize, frame.jitterPx, tolerance);
          frame.planes.reflectionTarget = await save(files, frame.id, 'reflectiontarget', sceneSize, incident);
          frame.inputSeeds = [];
          frame.teacher.reflection = stats;
        }
      }
      sequences.push({ id, split: partition, groupId,
        sceneSize, displaySize, replaySha256: await digest(new TextEncoder().encode(JSON.stringify(replay))), frames });
    } finally {
      raster?.engine.dispose();
      teacher?.engine.dispose();
      student.engine.dispose();
    }
  }
  return { json: { format: 'raa.gallery-temporal-sequence/1', source: { buildId, facts },
    guideChannels, maskChannels, rowOrder: 'top-down', colorSpace: 'scene-linear', exposure: 1,
    target: 'current real frame HDR from independent full mesh path tracing at the exact pose and time',
    ...(joint && { rasterTarget: 'native display-resolution scene-linear SSR composite from the same raster graph and replay, no camera jitter',
      studentTargetDomain: 'hdr is forward HDR, compositeHdr is the typed SSR composite from that same submission' }),
    reference: { samplesPerDispatch: spp, pathBounces: 2, reflectionBounces: 2, jitter: [0, 0], seedBatches: 2 },
    optionalHeads: { area: 'separate area pilot', reflection: reflection ? {
      channels: ['radianceR', 'radianceG', 'radianceB', 'valid', 'class', 'age'],
      supplier: 'typed SSR incident and hitMeta from the same real submission',
      target: 'accepted screen hits conditioned on independent path HDR and matching screen cone',
      eligibility: 'class 1 with geometric validity and positive screen confidence',
      misses: 'zero-filled and masked; BVH and volume fallback outside this capture',
    } : 'hit guides only' },
    reactiveSource: 'authored current panel revision for visible surfaces and conservative mirror coverage on moving-actor frames',
    sequences }, files };
}

export async function captureGallerySrReplay({ target, canvas, raa }, { buildId, trajectory } = {}) {
  if (!buildId || !/Windows.+Chrome\//.test(target.navigator.userAgent)) {
    throw new Error('An actual Windows Chrome GPU is required for SR replay');
  }
  const { id, partition, heldout, replay, groupId } = jointTrajectory(trajectory);
  const replaySha256 = await digest(new TextEncoder().encode(JSON.stringify(replay)));
  const sequences = [], files = [];
  const guideGraph = await reflectionGraph();
  let facts, weightsSha256;
  for (const arm of ['analytic', 'joint']) {
    const view = await create(target, raa, canvas, displaySize, heldout, false, guideGraph);
    const { engine } = view;
    try {
      facts = view.facts;
      const model = await loadGalleryNeural(engine);
      if (!model?.manifest || !['area', 'reflection', 'display'].every(head =>
        model.manifest.trainedHeads.includes(head))) throw new Error('SR replay needs an installed joint model package');
      weightsSha256 = model.manifest.weightsSha256;
      const selected = engine.setRenderGraphSelection(model.graph, undefined, ...sceneSize);
      if (!selected.ok) throw new Error('SR replay graph refused: ' + JSON.stringify(selected.issues));
      engine.setNeuralReconstruction(arm === 'joint'
        ? { model: model.id, area: true, reflection: true, sr: true } : null);
      view.apply(replay[0]);
      await warmSequence(engine, replay[0].timeMs);
      const frames = [];
      for (const [i, frame] of replay.entries()) {
        view.apply(frame, replay[i - 1]);
        const real = engine.renderInstrumented(frame.timeMs, {
          viewId: 'primary', trialId: id + '-' + arm, presentId: id + '-' + arm + '-' + i,
          inputCutoffMs: target.performance.now(),
        });
        if (real.error || !Number.isInteger(real.frameIndex)) throw new Error('SR replay real submission failed');
        const dispatch = engine.neuralReconstructionStats();
        if (arm === 'joint'
          ? !dispatch?.ran.area || !dispatch.ran.reflection || !dispatch.ran.sr
            || dispatch.realFrame !== real.frameIndex || dispatch.weightsSha256 !== weightsSha256
          : dispatch !== null) {
          throw new Error('SR replay actual dispatch differs from the requested mode: ' + JSON.stringify(dispatch));
        }
        const taps = await engine.captureTrainingTaps(real.frameIndex, true);
        const frameId = id + '-' + String(i).padStart(2, '0');
        const plane = await save(files, frameId, arm + 'resolved', displaySize, hdrRgb(taps.resolvedHdr, displaySize));
        frames.push({ id: frameId, frameId: real.frameIndex, arm, timeMs: frame.timeMs, event: frame.event,
          jitterNdc: taps.jitterNdc, dispatch, planes: { resolved: plane } });
      }
      sequences.push({ id, split: partition, groupId, arm, sceneSize, displaySize, replaySha256, frames });
    } finally {
      engine.dispose();
    }
  }
  return { json: { format: 'raa.gallery-sr-resolved-replay/1', source: { buildId, facts },
    rowOrder: 'top-down', colorSpace: 'scene-linear', exposure: 1, weightsSha256, sequences }, files };
}

export async function verifyGalleryReflection({ target, canvas, raa }, { buildId } = {}) {
  if (!/Windows.+Chrome\//.test(target.navigator.userAgent) || !buildId) throw new Error('Windows Chrome build required');
  const student = await create(target, raa, canvas, sceneSize, false, false, await reflectionGraph());
  const { engine, facts } = student;
  try {
    const model = await loadGalleryNeural(engine);
    if (!model.manifest.trainedHeads.includes('reflection')) throw new Error('Gallery reflection fit required');
    const frame = tape(false)[0];
    student.apply(frame);
    engine.setCameraJitter(false);
    const [w, h] = sceneSize;
    const cases = [];
    for (const [index, selected] of [false, true, false].entries()) {
      const graph = structuredClone(model.graph);
      if (selected) graph.nodes.find(node => node.type === 'neural.reflection-reconstruction').params = { model: model.id };
      const result = engine.setRenderGraphSelection(graph);
      if (!result.ok) throw new Error(`Reflection A/B graph: ${JSON.stringify(result.issues)}`);
      const samples = new Map();
      let timing;
      for (let i = 0; i < 48 && samples.size < 8; i++) {
        timing = engine.renderInstrumented(frame.timeMs, { viewId: 'primary', trialId: 'reflection-model',
          presentId: `reflection-${index}-${i}`, inputCutoffMs: target.performance.now() });
        if (timing.error || !Number.isInteger(timing.frameIndex)) throw new Error(`Reflection frame failed: ${JSON.stringify(timing.error)}`);
        await engine.submittedWorkDone();
        const gpu = engine.passGpuTimings();
        if (gpu.status === 'unavailable') throw new Error(`Reflection GPU timestamps unavailable: ${gpu.reason}`);
        if (i >= 6 && gpu.frame > timing.frameIndex - 5) {
          const ms = gpu.passes?.find(pass => pass.pass === 'ssr.replace#replace')?.ms;
          if (Number.isFinite(ms)) samples.set(gpu.frame, ms);
        }
      }
      if (samples.size < 8) throw new Error('Insufficient warmed reflection GPU samples');
      const taps = await engine.captureTrainingTaps(timing.frameIndex);
      const raw = floats(taps.reflectionIncident, w * h * 4, false);
      const resolved = floats(taps.reflectionReconstruction, w * h * 4, false);
      const meta = floats(taps.reflectionMeta, w * h * 4, false);
      let eligible = 0, changed = 0, masked = 0, alphaIdentical = true, maxDelta = 0;
      for (let p = 0; p < w * h; p++) {
        const lane = p * 4;
        const active = meta[lane] === 1 && meta[lane + 1] === 1 && raw[lane + 3] > 0;
        if (active) eligible++;
        else masked++;
        alphaIdentical &&= raw[lane + 3] === resolved[lane + 3];
        let delta = 0;
        for (let c = 0; c < 3; c++) delta = Math.max(delta, Math.abs(raw[lane + c] - resolved[lane + c]));
        if (delta > 0 && !active) throw new Error(`Reflection model changed masked pixel ${p}`);
        if (delta > 0) changed++;
        maxDelta = Math.max(delta, maxDelta);
      }
      const stats = engine.neuralReconstructionStats();
      if (!(eligible > 100 && masked > 100 && alphaIdentical
          && (selected ? changed > 0 && stats?.ran.reflection && stats.dispatches > 0
            && stats.weightsSha256 === model.manifest.weightsSha256 && stats.realFrame === timing.frameIndex
            : changed === 0 && stats === null))) throw new Error(`Reflection A/B proof failed: ${JSON.stringify({ selected, eligible, changed, masked, alphaIdentical, stats, timing })}`);
      cases.push({ selected, eligible, changed, masked, alphaIdentical, maxDelta, frameId: timing.frameIndex, stats,
        gpuTimings: [...samples].map(([frame, ms]) => ({ frame, pass: 'ssr.replace#replace', ms })) });
    }
    return { json: { source: { buildId, facts }, size: sceneSize, pose: frame.camera, timeMs: frame.timeMs,
      eligible: 'typed screen class 1, hitMeta.valid 1 and positive incident confidence', cases }, files: [] };
  } finally {
    engine.dispose();
  }
}

export { create as createGalleryReplay, floats as decodeGalleryTap, warmSequence as warmGalleryReplay };
