import { setCanvasSize } from '../../../web/common/page-runner.js';
import { volumeRoom } from './content.js';
import { interactiveInterior } from './interactive-interior.js';
import { PATH_FIXTURE, digest, loadGalleryFit } from './gallery-path-model.js';
import { runGallerySpherical, sphericalEvidenceSize } from './gallery-spherical-evidence.js';

const scene = interactiveInterior.default;
const TIME = 1000;
const TOLERANCE = 0.025;
const CAMERAS = [
  { id: 'home', eye: [0.1, 1.65, -0.35], target: [0, 1.32, -5.8] },
  { id: 'left-aisle', eye: [-1.1, 1.55, -2.1], target: [0.3, 1.28, -6.6] },
];
const HOLDOUT = { id: 'right-aisle', eye: [1.1, 1.6, -1.45], target: [0, 1.4, -6.3] };
const layout = { exposure: 1, rowOrder: 'top-down', byteOrder: 'little-endian', jitter: [0, 0] };

async function create({ target, canvas, raa }, buildId, width, height, { vertexWallColors = false } = {}) {
  if (!/Windows.+Chrome\//.test(target.navigator.userAgent) || typeof buildId !== 'string' || !buildId) {
    throw new Error('Gallery path evidence requires a pinned Windows Chrome build');
  }
  setCanvasSize(canvas, width, height);
  const engine = await raa.createEngine(canvas, { msaa: 1, allowSoftware: false, powerPreference: 'high-performance' });
  try {
    const { hardware, adapter } = engine.stats();
    if (!hardware || adapter?.isFallbackAdapter !== false) throw new Error('Expected hardware WebGPU adapter');
    const preset = scene.variants.find(variant => variant.id === 'performance');
    const state = scene.build(engine, { raa }, {
      ...preset, vertexWallColors,
      effects: { ...preset.effects, gi: false, ssr: false, dof: false, bloom: false, 'moving-objects': false },
    });
    engine.setPassEnabled('taa', false);
    engine.setCameraJitter(false);
    engine.setAutoExposure({ enabled: false });
    engine.setToneMapping({ exposureEv: 0 });
    const facts = { hardware, adapter, browser: target.navigator.userAgent, devicePixelRatio: target.devicePixelRatio };
    const source = { kind: 'windows-render', buildId, fixture: PATH_FIXTURE, facts };
    return { engine, source, state, setPose({ eye, target }) { engine.setCamera({ ...preset.selection.camera, eye, target }); } };
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

const REGIONS = {
  redWall: [28, 115, 88, 152], greenWall: [548, 108, 610, 148],
  panelLitFloor: [252, 307, 302, 340], warmPanel: [72, 12, 155, 28],
};

function half(view, offset) {
  const bits = view.getUint16(offset, true);
  const sign = (bits & 0x8000) ? -1 : 1;
  const exp = (bits >> 10) & 31;
  const fraction = bits & 1023;
  if (exp === 31) throw new Error('Non-finite linear diffuse pixel');
  return sign * (exp ? (1024 + fraction) * 2 ** (exp - 25) : fraction * 2 ** -24);
}

function accumulateRegions(sums, sample, width, height) {
  if (sample.size[0] !== width || sample.size[1] !== height || sample.data.byteLength !== width * height * 8) {
    throw new Error('Diffuse capture extent or RGBA16F byte length differs');
  }
  const view = new DataView(sample.data.buffer, sample.data.byteOffset, sample.data.byteLength);
  for (const [name, [x0, y0, x1, y1]] of Object.entries(REGIONS)) {
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        for (let c = 0; c < 3; c++) sums[name][c] += half(view, (y * width + x) * 8 + c * 2);
      }
    }
  }
}

function verifyDiffuseCases(cases) {
  const get = (color, mode, panels = true) => cases.find(c => c.color === color && c.mode === mode && c.panels === panels).regions;
  for (const mode of ['raster-diffuse', 'learned-direct', 'reference-direct', 'learned-diffuse', 'reference-diffuse']) {
    const material = get('material', mode), vertex = get('vertex', mode);
    for (const [region, dominant] of [['redWall', 0], ['greenWall', 1]]) {
      const a = material[region], b = vertex[region];
      if (a[dominant] < 0.004 || b[dominant] < 0.004
          || b[dominant] < 1.6 * Math.max(...b.filter((_, c) => c !== dominant))
          || a.some((value, c) => Math.abs(value - b[c]) > Math.max(0.0002, 0.035 * value))) {
        throw new Error(`${mode}/${region}: vertex-tinted diffuse disagrees with material diffuse ${JSON.stringify({ a, b })}`);
      }
      if (mode.endsWith('direct') || mode === 'raster-diffuse') {
        if (get('vertex', mode, false)[region].some(value => Math.abs(value) > 0.0001)) {
          throw new Error(`${mode}/${region}: disabled panels still shade the wall`);
        }
      }
    }
  }
  const warmPanel = get('material', 'raster-rgb').warmPanel;
  const darkPanel = get('material', 'raster-rgb', false).warmPanel;
  if (warmPanel[0] <= warmPanel[2] || warmPanel.some((value, c) => value - darkPanel[c] < 0.25)) {
    throw new Error('The warm panel lost its colored emission response');
  }
  for (const name of ['redWall', 'greenWall', 'panelLitFloor']) {
    const raster = get('material', 'raster-diffuse')[name];
    const learned = get('material', 'learned-direct')[name];
    const reference = get('material', 'reference-direct')[name];
    for (let c = 0; c < 3; c++) {
      // Raster form-factor clipping and traced visibility differ at occluder edges.
      if (Math.abs(learned[c] - reference[c]) > Math.max(0.001, 0.12 * reference[c])
          || Math.abs(learned[c] - raster[c]) > Math.max(0.003, 0.17 * raster[c])) {
        throw new Error(`${name}: panel diffuse RGB mismatch ${JSON.stringify({ raster, learned, reference })}`);
      }
    }
  }
}

export async function measureGalleryDiffuse(context, { buildId, requireFixed = true } = {}) {
  const width = 640, height = 360, samples = 8;
  const cases = [];
  let captureSource;
  let modelSha256;
  for (const color of ['material', 'vertex']) {
    const { engine, source, state, setPose } = await create(context, buildId, width, height, {
      vertexWallColors: color === 'vertex',
    });
    captureSource = source;
    try {
      const model = await loadGalleryFit();
      if (!model) throw new Error('Expected fitted gallery weights for diffuse regression');
      modelSha256 = model.fit.weights.sha256;
      engine.setMeshPathModel(model.bytes);
      setPose(CAMERAS[0]);
      engine.setLight({ ...volumeRoom().sun, intensity: 0 });
      engine.setPassEnabled('ao.gtao', false);
      const modes = [
        ['raster-rgb', 'off', false, 'off'],
        ['raster-diffuse', 'off', false, 'off'],
        ['learned-direct', 'learned', false, 'direct'],
        ['reference-direct', 'off', true, 'direct'],
        ['learned-diffuse', 'learned', false, 'direct-and-indirect'],
        ['reference-diffuse', 'off', true, 'direct-and-indirect'],
      ];
      for (const [mode, pathMode, reference, diffuseOutput] of modes) {
        engine.setAreaLightShading({ mode: mode === 'raster-rgb' ? 'ltc' : 'formFactor' });
        engine.setMeshTracing({ pathMode, reference, diffuseOutput, reflections: reference, pathBounces: 2 });
        for (const panels of [true, false]) {
          scene.setOption(engine, state, 'area-lights', panels);
          const sums = Object.fromEntries(Object.keys(REGIONS).map(name => [name, [0, 0, 0]]));
          for (let i = 0; i < samples; i++) {
            accumulateRegions(sums, await engine.captureMeshLighting(TIME, pathMode !== 'off' || reference), width, height);
          }
          const regions = Object.fromEntries(Object.entries(sums).map(([name, sum]) => {
            const [x0, y0, x1, y1] = REGIONS[name];
            return [name, sum.map(value => value / (samples * (x1 - x0) * (y1 - y0)))];
          }));
          const stats = engine.meshTracingStats();
          if (pathMode === 'learned' && (!stats?.modelPixels || stats.referencePixels)) throw new Error('No model path ran');
          if (reference && (!stats?.referencePixels || stats.modelPixels)) throw new Error('Reference used the model');
          cases.push({ color, mode, panels, regions, stats });
        }
      }
    } finally {
      engine.dispose();
    }
  }
  if (requireFixed) verifyDiffuseCases(cases);
  return { json: {
    source: captureSource, modelSha256, size: [width, height], camera: CAMERAS[0], samples, ...layout,
    space: 'linear-scene-rgb-from-rgba16f', regions: REGIONS, requireFixed, cases,
  }, files: [] };
}

export async function measureGallerySpherical(context, { buildId, remainingMs = 280000 } = {}) {
  const [width, height] = sphericalEvidenceSize;
  const { engine, source, state, setPose } = await create(context, buildId, width, height);
  try {
    return await runGallerySpherical({ engine, source, state, setPose, scene, camera: CAMERAS[0],
      target: context.target, half, digest, timeMs: TIME, remainingMs });
  } finally {
    engine.dispose();
  }
}

const PANEL_ROI = [132, 67, 206, 100];
const PANEL_ESTIMATORS = ['single', 'stable', 'referenceA', 'referenceB'];

function reflectionCrop(sample) {
  const [width, height] = sample.size;
  if (width !== 320 || height !== 180 || sample.data.byteLength !== width * height * 8) {
    throw new Error('Expected half-resolution linear RGBA16F mesh reflection samples');
  }
  const [x0, y0, x1, y1] = PANEL_ROI;
  const result = new Float32Array((x1 - x0) * (y1 - y0) * 4);
  const view = new DataView(sample.data.buffer, sample.data.byteOffset, sample.data.byteLength);
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const src = (y * width + x) * 8, dst = ((y - y0) * (x1 - x0) + x - x0) * 4;
      for (let c = 0; c < 4; c++) result[dst + c] = half(view, src + c * 2);
    }
  }
  return result;
}

function panelContribution(on, off) {
  return on.map((value, i) => i % 4 === 3 ? Number(value === 1 && off[i] === 1) : value - off[i]);
}

function panelErrors(frames) {
  const rowWidth = PANEL_ROI[2] - PANEL_ROI[0];
  const rms = (squares, count) => Math.sqrt(squares / count);
  const reports = {};
  for (const name of ['single', 'stable']) {
    let spatial = 0, fine = 0, temporal = 0, referenceBatch = 0;
    let pixels = 0, edges = 0, pairs = 0;
    const energy = [0, 0, 0], referenceEnergy = [0, 0, 0];
    let previous;
    for (const frame of frames) {
      const { referenceA: a, referenceB: b } = frame;
      const estimate = frame[name], residual = new Float32Array(a.length);
      for (let i = 0; i < a.length; i += 4) {
        if (a[i + 3] !== 1 || b[i + 3] !== 1 || estimate[i + 3] !== 1) continue;
        residual[i + 3] = 1;
        pixels++;
        for (let c = 0; c < 3; c++) {
          const j = i + c, target = (a[j] + b[j]) / 2;
          residual[j] = estimate[j] - target;
          spatial += residual[j] ** 2;
          referenceBatch += (a[j] - b[j]) ** 2;
          energy[c] += estimate[j];
          referenceEnergy[c] += target;
        }
      }
      for (let i = 0; i < residual.length; i += 4) {
        if (residual[i + 3] !== 1) continue;
        if (previous?.[i + 3] === 1) {
          pairs++;
          for (let c = 0; c < 3; c++) temporal += (residual[i + c] - previous[i + c]) ** 2;
        }
        const right = (i / 4) % rowWidth < rowWidth - 1 ? i + 4 : -4;
        for (const next of [right, i + rowWidth * 4]) {
          if (residual[next + 3] !== 1) continue;
          edges++;
          for (let c = 0; c < 3; c++) fine += (residual[i + c] - residual[next + c]) ** 2;
        }
      }
      previous = residual;
    }
    if (!pixels || !edges || !pairs || referenceEnergy.some(value => value <= 0)) {
      throw new Error('Missing reflected direct panel signal');
    }
    reports[name] = {
      pixels, edges, pairs, spatialRmse: rms(spatial, 3 * pixels), fineSpatialRmse: rms(fine, 3 * edges),
      temporalRmse: rms(temporal, 3 * pairs), referenceBatchRmse: rms(referenceBatch, 3 * pixels),
      meanRgb: energy.map(value => value / pixels), referenceMeanRgb: referenceEnergy.map(value => value / pixels),
    };
  }
  return reports;
}

export async function measureGalleryPanelNoise(context, { buildId, requireFixed = true } = {}) {
  const { engine, source, state, setPose } = await create(context, buildId, 640, 360);
  const files = [], frames = [], samples = [];
  try {
    engine.setLight({ ...volumeRoom().sun, intensity: 0 });
    engine.setPassEnabled('ao.gtao', false);
    const configure = panelSampling => {
      engine.setMeshTracing({
        reflections: true, reference: false, pathMode: 'off', diffuseOutput: 'off',
        reflectionBounces: 2, confidenceThreshold: 1, panelSampling,
      });
    };
    const render = async panelSampling => {
      configure(panelSampling);
      return reflectionCrop(await engine.captureMeshLighting(TIME, true, true));
    };
    const camera = CAMERAS[0];
    for (let i = 0; i < 9; i++) {
      const shift = (i - 4) * 0.022;
      const pose = {
        eye: [camera.eye[0] + shift, ...camera.eye.slice(1)],
        target: [camera.target[0] + shift, ...camera.target.slice(1)],
      };
      setPose(pose);
      // Change the light intensity alone. Keep emissive mesh color, geometry,
      // sky and all other shading fixed so the linear difference is direct panel light.
      for (const light of state.fixtures) engine.updateLight(light.light, { intensity: 0 });
      const off = await render('single');
      for (const light of state.fixtures) engine.updateLight(light.light, { intensity: light.intensity });
      const data = {};
      const planes = {};
      for (const name of PANEL_ESTIMATORS) {
        const sampler = name.startsWith('reference') ? 'independent' : name;
        data[name] = panelContribution(await render(sampler), off);
        const stats = engine.meshTracingStats();
        if (!stats?.traced || stats.acceptedScreen !== 0 || stats.eligible !== stats.traced) {
          throw new Error('Panel comparison needs actual mirror-traced pixels without screen reuse');
        }
        const [x0, y0, x1, y1] = PANEL_ROI;
        planes[name] = await plane(files, `pan-${i}-${name.toLowerCase()}.bin`, [y1 - y0, x1 - x0, 4], data[name]);
      }
      frames.push({ pose, planes });
      samples.push(data);
    }
    const errors = panelErrors(samples);
    const { stable, single } = errors;
    if (requireFixed && (
      stable.referenceBatchRmse >= single.spatialRmse * 0.25
      || stable.spatialRmse >= single.spatialRmse * 0.65
      || stable.fineSpatialRmse >= single.fineSpatialRmse * 0.65
      || stable.temporalRmse >= single.temporalRmse * 0.7
      || stable.meanRgb.some((value, c) => Math.abs(value / stable.referenceMeanRgb[c] - 1) > 0.05)
    )) throw new Error(`Panel reflection stability regression: ${JSON.stringify(errors)}`);
    setPose(camera);
    const gpuTimings = {};
    for (const mode of ['single', 'stable']) {
      configure(mode);
      for (let i = 0; i < 6; i++) {
        engine.render(TIME);
        await engine.submittedWorkDone();
        engine.passGpuTimings();
      }
      const timings = new Map();
      for (let i = 0; i < 48 && timings.size < 16; i++) {
        engine.render(TIME);
        await engine.submittedWorkDone();
        const sample = engine.passGpuTimings();
        if (sample.status === 'unavailable') throw new Error(`GPU timestamps unavailable: ${sample.reason}`);
        const pass = sample.passes?.find(stage => stage.pass === 'trace.mesh');
        if (pass?.ms != null) timings.set(sample.frame, pass.ms);
      }
      if (timings.size < 16) throw new Error(`Missing mesh pass timestamps for ${mode}`);
      gpuTimings[mode] = [...timings].map(([frame, ms]) => ({ frame, ms }));
    }
    return { json: {
      source, size: [640, 360], reflectionSize: [320, 180], roi: PANEL_ROI, requireFixed, ...layout,
      case: 'static gallery with laterally moving camera and geometry-traced rear mirror',
      signal: 'linear scene RGB, panel intensities on minus off; alpha marks mirror coverage',
      referenceSamplesPerPanel: [128, 128], frames, errors, gpuTimings,
    }, files };
  } finally {
    engine.dispose();
  }
}
