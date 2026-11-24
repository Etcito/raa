import { setCanvasSize } from '../../../web/common/page-runner.js';
import { interactiveInterior } from './interactive-interior.js';
import { panel } from './content.js';
import { inspectMirrorCoverage, measureLearnedSecondary, measureSharpSecondary } from './gallery-secondary-evidence.js';
import { digest } from './gallery-path-model.js';

const scene = interactiveInterior.default;
const preset = scene.variants[0];
const timeMs = 1000;
const extent = [320, 180];

function requireEvidence(condition, message, detail) {
  if (!condition) throw new Error(`${message}${detail === undefined ? '' : `: ${JSON.stringify(detail)}`}`);
}

function rgbPlane(data, size) {
  const pixels = size[0] * size[1];
  requireEvidence(data.byteLength === pixels * 8, 'Invalid linear RGBA16F extent', size);
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const rgb = Float32Array.from({ length: pixels * 3 }, (_, i) =>
    view.getFloat16((Math.floor(i / 3) * 4 + i % 3) * 2, true));
  requireEvidence(rgb.every(Number.isFinite), 'Nonfinite linear reflection HDR');
  return rgb;
}

function compare(a, b, accept = () => true) {
  let squared = 0, peak = 0, changedPixels = 0, pixels = 0;
  for (let p = 0; p < a.length / 3; p++) {
    if (!accept(p)) continue;
    pixels++;
    let changed = false;
    for (let c = 0; c < 3; c++) {
      const delta = Math.abs(a[p * 3 + c] - b[p * 3 + c]);
      squared += delta * delta;
      peak = Math.max(peak, delta);
      changed ||= delta > 0.0001;
    }
    changedPixels += Number(changed);
  }
  return { pixels, changedPixels, peak, rmse: pixels ? Math.sqrt(squared / (3 * pixels)) : null };
}

export async function measureGalleryMirror({ target, canvas, raa }, {
  buildId, remainingMs = 280000, baseline = false, coverageOnly = false, sharpOnly = false,
} = {}) {
  requireEvidence(/Windows.+Chrome\//.test(target.navigator.userAgent) && buildId, 'Pinned Windows Chrome build required');
  const deadline = target.performance.now() + remainingMs;
  const checkTime = () => requireEvidence(target.performance.now() < deadline, 'Mirror evidence timed out');
  setCanvasSize(canvas, ...extent);
  const engine = await raa.createEngine(canvas, { ...scene.engine, ...preset.engine,
    allowSoftware: false, powerPreference: 'high-performance' });
  try {
    const { hardware, adapter } = engine.stats();
    requireEvidence(hardware && adapter?.isFallbackAdapter === false, 'Hardware WebGPU is required');
    engine.resize(...extent);
    const state = scene.build(engine, { raa }, preset);
    engine.setCamera(preset.selection.camera);
    const initial = { ...state.enabled, ...state.values };
    const set = (id, value) => scene.setOption(engine, state, id, value);
    const files = [], captures = [];
    engine.setCameraJitter(false);
    engine.setAutoExposure({ enabled: false });
    engine.setToneMapping({ exposureEv: 0 });

    async function settle(grid) {
      for (let i = 0; i < 160; i++) {
        checkTime();
        engine.render(timeMs);
        await engine.submittedWorkDone();
        const volume = engine.giStats();
        const spherical = engine.sphericalGiStats();
        if (i >= 7 && !volume.pendingRebuild && (grid === 'volume' ||
          (spherical?.active && spherical.readyCells > 0 && !spherical.pendingFaces
            && !spherical.inFlightFaces && !spherical.lastCellQueries))) return spherical;
      }
      throw new Error(`${grid} did not settle at the fixed camera`);
    }

    async function save(id, data, format) {
      const file = `${id}.${format}`;
      files.push({ name: file, blob: new Blob([data], { type: 'application/octet-stream' }) });
      return { file, byteLength: data.byteLength, sha256: await digest(data) };
    }

    async function capture(id, traced, raw = false, evidenceSeed, reference = false) {
      checkTime();
      const { data, size, roughEvidence } = reference
        ? await engine.captureTrainingReference(timeMs, evidenceSeed, 1)
        : await engine.captureMeshLighting(timeMs, traced, raw, evidenceSeed);
      const expected = raw && !reference ? extent.map(value => value / 2) : extent;
      requireEvidence(size.join() === expected.join(), 'Capture extent changed', { id, size, expected });
      const frame = engine.stats().frames - 1;
      let trace = engine.meshTracingStats();
      for (let i = 0; traced && trace?.frame !== frame && i < 80; i++) {
        await new Promise(resolve => setTimeout(resolve, 5));
        trace = engine.meshTracingStats();
      }
      if (traced) requireEvidence(trace?.frame === frame, 'Mesh counters were not recorded for the captured frame', { id, frame, trace });
      const entry = { id, frame, size, ...await save(id, data, 'rgba16f'), trace: traced ? trace : null,
        cache: engine.sphericalGiStats(), grid: engine.sphericalGi() };
      captures.push(entry);
      if (roughEvidence) captures.push({ id: id + '-rough-hits', size: roughEvidence.size,
        stride: 48, dtype: 'f32le/u32le', ...await save(id + '-rough-hits', roughEvidence.data, 'bin') });
      return { ...entry, rgb: rgbPlane(data, size), roughEvidence };
    }

    requireEvidence(initial['reflection-bounces'] === 2 && initial['gi-grid'] === 'volume'
      && initial['spherical-debug'] === 'off' && initial.gi, 'Unexpected gallery defaults', initial);
    await settle('volume');
    const mirrorOff = await capture('volume-mirror-off', false);
    set('geometry-reflections', true);
    await settle('volume');
    const volume = await capture('volume-default-mirror', true);
    const two = await capture('volume-two-raw', true, true);
    set('reflection-bounces', 1);
    const one = await capture('volume-one-raw', true, true);
    set('reflection-bounces', initial['reflection-bounces']);
    const mirrorChange = compare(volume.rgb, mirrorOff.rgb);
    const bounceChange = compare(two.rgb, one.rgb);
    const baselineFacts = {
      default: two.trace, firstRayMisses: two.trace.traced - two.trace.geometryHit,
      secondMirrorRays: two.trace.mirrorRays - two.trace.traced, mirrorChange, bounceChange,
    };
    requireEvidence(two.trace.offscreen > 0 && two.trace.geometryHit > 0
      && baselineFacts.secondMirrorRays > 0 && one.trace.mirrorRays === one.trace.traced
      && mirrorChange.changedPixels > 10 && bounceChange.changedPixels > 10,
    'Gallery mirror fixture failed to exercise BVH geometry, offscreen, misses and the second ray', baselineFacts);
    set('gi-grid', 'spherical');
    const cold = baseline ? null : await capture('default-spherical-cold', true, true);
    const residency = await settle('spherical');
    const sphere = await capture('spherical-default-mirror', true);
    const comparison = compare(sphere.rgb, volume.rgb);
    const coverage = baseline ? null : inspectMirrorCoverage(cold,
      await capture('default-spherical-warm', true, true), residency, requireEvidence);
    const facts = { hardware, adapter, browser: target.navigator.userAgent, devicePixelRatio: target.devicePixelRatio };
    const manifest = { source: { kind: 'windows-render', buildId, facts }, size: extent, camera: preset.selection.camera,
      timeMs, preset: preset.id, initial, baseline, layout: 'linear-scene RGBA16F little-endian top-down',
      baselineFacts, residency, comparison, captures, coverage };
    if (coverageOnly || coverage?.zeroCoverage) return { json: { ...manifest, coverageOnly }, files };
    const sharp = baseline ? null : await measureSharpSecondary({ engine, capture, check: requireEvidence, compare,
      defaultConfidence: initial['reflection-confidence'] });
    if (sharpOnly) return { json: { ...manifest, sharp, sharpOnly }, files };
    const learned = baseline ? null : await measureLearnedSecondary({ engine, state, set, settle, capture,
      check: requireEvidence, compare });
    let reference;
    if (!baseline) {
      set('model-assisted-path', false);
      // Keep the cache resident while asking the independent reference to render.
      engine.setMeshTracing({ reference: true, pathMode: 'off', reflections: true, lightingOutput: 'combined',
        pathBounces: 2, reflectionBounces: 2 });
      const attached = await capture('reference-cache-resident', true, true, 73030, true);
      requireEvidence(attached.grid.enabled && attached.cache?.readyCells > 0,
        'Reference exclusion requires a populated spherical cache', attached.cache);
      engine.setSphericalGi({ enabled: false });
      const detached = await capture('reference-cache-released', true, true, 73030, true);
      const neutral = compare(attached.rgb, detached.rgb);
      requireEvidence(neutral.peak === 0 && [attached, detached].every(sample =>
        sample.trace.referencePixels > 0 && sample.trace.modelPixels === 0 && !sample.roughEvidence
        && ['sphericalDiffuseHits', 'sphericalSpecularHits', 'volumeDiffuseHits', 'volumeSpecularHits']
          .every(key => sample.trace[key] === 0)),
      'Independent reference output moved when the resident spherical cache was released', { neutral, attached: attached.trace, detached: detached.trace });
      reference = { seed: 73030, residentCells: attached.cache.readyCells, ...neutral };
      engine.setMeshTracing({ reference: false, pathMode: 'off' });
    }

    // The home mirror is enclosed and all its first rays hit geometry. Give the
    // miss control an isolated mirror that reflects into empty space.
    set('gi-grid', 'volume');
    const missCamera = { ...preset.selection.camera, eye: [64, 2, 0], target: [64, 2, -4] };
    const surface = engine.addSurfaceMaterial({
      color: [0.22, 0.28, 0.33], roughness: 0.42, metallic: 0.72, traceMirror: true,
    });
    const node = engine.addNode(null, [64, 2, -4], [0, 0, 0, 1], [1, 1, 1]);
    const mesh = panel(3, 2);
    engine.addMeshInstance(node, engine.addMesh({
      ...mesh, subMeshes: [{ first: 0, count: mesh.indices.length, material: surface }],
    }));
    engine.setCamera(missCamera);
    await settle('volume');
    const miss = await capture('isolated-mirror-sky-miss', true, true);
    requireEvidence(miss.trace.traced > 100 && miss.trace.geometryHit === 0,
      'The isolated mirror did not reflect into empty space', miss.trace);
    return { json: { ...manifest, sharp, learned, reference, missCamera, miss: miss.trace }, files };
  } finally {
    engine.dispose();
  }
}

export async function verifyGalleryMirrorUi({ fitted = false } = {}) {
  const option = id => document.querySelector(`[data-option="${id}"]`);
  const status = () => document.getElementById('status').textContent;
  const stats = () => document.getElementById('live-stats').textContent;
  async function wait(condition, description) {
    const deadline = performance.now() + 25000;
    while (!condition()) {
      requireEvidence(performance.now() < deadline && !status().startsWith('failed:'), description, status());
      await new Promise(resolve => setTimeout(resolve, 50));
    }
  }
  const change = (id, value) => {
    const input = option(id);
    requireEvidence(input && !input.disabled, `Gallery control unavailable: ${id}`);
    if (input.type === 'checkbox') input.checked = value;
    else input.value = value;
    input.dispatchEvent(new Event('change', { bubbles: true }));
  };
  document.getElementById('run').click();
  await wait(() => option('geometry-reflections'), 'Gallery mirror controls did not mount');
  const initial = Object.fromEntries(['geometry-reflections', 'gi-grid', 'reflection-bounces',
    'reflection-confidence', 'spherical-subdivision', 'spherical-exponent', 'spherical-clip-start',
    'spherical-z-max', 'spherical-debug'].map(id => [id, option(id).type === 'checkbox' ? option(id).checked : option(id).value]));
  requireEvidence(initial['geometry-reflections'] === false && initial['gi-grid'] === 'volume'
    && initial['reflection-bounces'] === '2' && initial['spherical-debug'] === 'off', 'Unexpected gallery mirror UI defaults', initial);
  change('geometry-reflections', true);
  await wait(() => stats().includes('Hybrid reflections'), 'Default mirrors did not report traced reflection work');
  const volume = stats();
  change('gi-grid', 'spherical');
  await wait(() => stats().includes('Hybrid reflections') && /Spherical GI · ready cells [1-9]/.test(stats()),
    'Default spherical mirror UI did not reach occupied captures');
  const spherical = stats();
  let learned;
  if (fitted) {
    const cache = option('spherical-mirror-cache');
    requireEvidence(cache?.value === 'auto' && !cache.disabled, 'Spherical secondary cache must start in automatic mode');
    change('model-assisted-path', true);
    const ran = () => /Hybrid path · model [1-9]/.test(stats());
    await wait(() => ran() && !option('gi-grid').disabled, 'Fitted dispatch did not run with the spherical grid');
    const sphericalFit = stats();
    for (const lobe of ['diffuse-indirect', 'rough-specular']) {
      requireEvidence(![...option('lighting-view').options].find(row => row.value === lobe)?.disabled,
        `Fitted lobe control is disabled: ${lobe}`);
      change('lighting-view', lobe);
      await wait(ran, `Fitted dispatch disappeared in ${lobe}`);
    }
    change('spherical-mirror-lighting', '1');
    await new Promise(resolve => requestAnimationFrame(resolve));
    change('spherical-mirror-lighting', '7');
    change('spherical-mirror-cache', 'off');
    await wait(() => option('spherical-mirror-lighting').disabled, 'Cache bypass kept group controls active');
    change('spherical-mirror-cache', 'auto');
    change('gi-grid', 'volume');
    await wait(() => ran() && option('spherical-mirror-cache').disabled && !stats().includes('Spherical GI'),
      'Fitted dispatch did not run after switching to the volume grid');
    const volumeFit = stats();
    change('lighting-view', 'combined');
    change('model-assisted-path', false);
    learned = { spherical: sphericalFit, volume: volumeFit };
  }
  return { status: status(), initial, volume, spherical, learned,
    resolution: document.getElementById('resolution').value };
}
