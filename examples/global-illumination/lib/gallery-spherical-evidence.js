import { box } from './content.js';

const groups = [['sky', 1], ['sun', 2], ['other', 4]];
const debugViews = ['boundaries', 'occupancy', 'age', 'light-hash'];
export const sphericalEvidenceSize = [320, 180];

function check(condition, description, detail) {
  if (!condition) throw new Error(`Spherical GI ${description}${detail === undefined ? '' : ': ' + JSON.stringify(detail)}`);
}

function difference(a, b) {
  let squares = 0, peak = 0, changed = 0;
  const meanRgb = [0, 0, 0];
  for (let i = 0; i < a.length; i++) {
    const delta = a[i] - b[i], magnitude = Math.abs(delta);
    meanRgb[i % 3] += delta;
    squares += delta * delta;
    peak = Math.max(peak, magnitude);
    if (magnitude > 1e-4) changed++;
  }
  return { rmse: Math.sqrt(squares / a.length), peak, changedChannels: changed,
    meanRgb: meanRgb.map(sum => sum / (a.length / 3)) };
}

export async function runGallerySpherical({ engine, source, state, scene, setPose, camera, target,
  half, digest, timeMs, remainingMs }) {
  const [width, height] = sphericalEvidenceSize;
  const deadline = target.performance.now() + remainingMs;
  const files = [], captures = [], comparisons = [], transitions = [];
  const stats = () => engine.sphericalGiStats();
  const set = (id, value) => scene.setOption(engine, state, id, value);
  const configure = options => engine.setSphericalGi(options);
  const pass = () => engine.passes().some(row => row.name === 'gi.spherical' && row.enabled);
  const live = () => {
    const sample = stats();
    check(sample?.active && pass() && sample.gpuBytes > 0, 'capture pass is inactive', sample);
    check(sample.readyFaces >= 6 * sample.readyCells && sample.cells === sample.nearCells + sample.farCells,
      'scheduler counters disagree', sample);
    return sample;
  };
  const record = (id, sample = stats()) => {
    const timing = engine.passGpuTimings();
    const sphericalGpuMs = timing.passes?.find(stage => stage.pass === 'gi.spherical')?.ms ?? null;
    transitions.push({ id, frame: engine.stats().frames, settings: engine.sphericalGi(), stats: sample,
      pass: pass(), sphericalGpuMs });
    return sample;
  };
  const frame = async () => {
    check(target.performance.now() < deadline, 'evidence timed out');
    engine.render(timeMs);
    await engine.submittedWorkDone();
  };
  async function warm(minCells = 8) {
    for (let i = 0; i < 120; i++) {
      await frame();
      const sample = live();
      if (i >= 7 && sample.readyCells >= minCells && sample.pendingFaces === 0 && sample.inFlightFaces === 0
          && sample.lightingMetadataReadyCells === sample.readyCells) return sample;
    }
    check(false, 'occupied gallery grid did not become ready', stats());
  }
  async function readProbe() {
    let probe = engine.sphericalGiProbe();
    for (let i = 0; i < 20 && !probe; i++) {
      await frame();
      probe = engine.sphericalGiProbe();
    }
    check(probe?.faceHits.length === 6 && probe.faceHits.some(count => count > 0)
      && probe.faceHits.every((count, i) => count === probe.faceValidGeometry[i] && count + probe.faceValidMisses[i] === 64)
      && probe.channelSums.length === 3 && probe.channelSums.flat().every(Number.isFinite)
      && /^[a-f0-9]{16}$/.test(probe.lightHash) && probe.contributingLightIds.length > 0
      && probe.contributingLightIds.every(id => /^[a-f0-9]{16}$/.test(id))
      && new Set(probe.contributingLightIds).size === probe.contributingLightIds.length,
    'occupied probe has invalid GPU position, normal, or lighting data', probe);
    return probe;
  }
  function lastShell(sample) {
    const { inner, center, outer } = sample.lastShell;
    const zMax = engine.sphericalGi().zMax;
    check(outer === zMax && Math.abs(center - (zMax - (outer - inner) / 2)) < 2e-5,
      'terminal shell does not end at zMax', sample.lastShell);
  }
  async function capture(id) {
    check(target.performance.now() < deadline, 'evidence timed out');
    const { data, size } = await engine.captureMeshLighting(timeMs, false);
    check(size[0] === width && size[1] === height && data.byteLength === width * height * 8,
      'unexpected linear HDR layout', size);
    const bytes = new DataView(data.buffer, data.byteOffset, data.byteLength);
    const rgb = new Float32Array(width * height * 3);
    for (let p = 0; p < width * height; p++) {
      for (let c = 0; c < 3; c++) rgb[p * 3 + c] = half(bytes, p * 8 + c * 2);
    }
    const file = `${id}.rgba16f`;
    files.push({ name: file, blob: new Blob([data], { type: 'application/octet-stream' }) });
    captures.push({ id, file, byteLength: data.byteLength, sha256: await digest(data), stats: stats() });
    return rgb;
  }

  setPose(camera);
  engine.setPassEnabled('ao.gtao', false);
  set('lighting-view', 'indirect');
  check(!engine.sphericalGi().enabled && stats() === null && !pass(), 'feature is enabled on a default engine');
  set('gi', true);
  // Isolate the spherical lighting groups from both direct light and volume fallback.
  engine.setGlobalIllumination({ enabled: false, screenSpace: false });
  for (let i = 0; i < 8; i++) await frame();
  record('never-enabled');
  const defaultOff = await capture('default-off');
  set('gi-grid', 'spherical');
  lastShell(record('occupied', await warm()));
  const probe = await readProbe();
  configure({ channelMask: 0 });
  const noGroups = await capture('groups-off');
  const sum = new Float32Array(noGroups.length);
  for (const [id, channelMask] of groups) {
    configure({ channelMask });
    const signal = await capture(`${id}-on`);
    const delta = difference(signal, noGroups);
    check(delta.changedChannels > 20 && delta.rmse > 1e-5, `${id} group has no linear HDR response`, delta);
    comparisons.push({ id, on: `${id}-on`, off: 'groups-off', ...delta });
    for (let i = 0; i < sum.length; i++) sum[i] += signal[i] - noGroups[i];
  }
  configure({ channelMask: 7 });
  const combined = await capture('groups-on');
  const groupTotal = difference(combined, noGroups);
  const additiveError = difference(combined, sum.map((value, i) => value + noGroups[i]));
  check(additiveError.rmse <= 0.03 * groupTotal.rmse + 1e-4, 'groups are not additive in linear HDR',
    { additiveError, groupTotal });

  const occupied = live();
  const debug = [];
  for (const id of debugViews) {
    set('spherical-debug', id);
    const signal = await capture(`debug-${id}`);
    const delta = difference(signal, combined);
    check(delta.changedChannels > 20, `${id} debug has no output`, delta);
    for (const previous of debug) check(difference(signal, previous.signal).changedChannels > 20,
      `${id} and ${previous.id} debug outputs match`);
    debug.push({ id, signal });
    const after = live();
    check(after.originEpoch === occupied.originEpoch && after.readyCells >= occupied.readyCells,
      `${id} debug discarded the capture cache`, after);
  }
  set('spherical-debug', 'off');

  const frameBefore = engine.stats().frames;
  for (const [option, key, value] of [
    ['spherical-subdivision', 'sphereSubdivision', 4],
    ['spherical-exponent', 'exponentialPower', 1],
    ['spherical-clip-start', 'clipStart', 0.25],
    ['spherical-z-max', 'zMax', 16],
  ]) {
    const before = live();
    set(option, value);
    await frame();
    const after = record(`hot-${key}`, live());
    check(engine.sphericalGi()[key] === value && engine.stats().frames > frameBefore,
      `hot ${key} did not apply on the current engine`);
    check(after.readyFaces < before.readyFaces, `hot ${key} retained incompatible faces`, { before, after });
    lastShell(after);
    await warm();
  }
  // A refused multi-field patch must keep both the accepted settings and buffers.
  const beforePatch = JSON.stringify(engine.sphericalGi());
  for (const patch of [{ sphereSubdivision: 5, zMax: 0 }, { sphereSubdivision: 2.5 },
    { exponentialPower: Infinity }, { clipStart: -1 }, { debug: 'unknown' }]) {
    let refusal;
    try { configure(patch); } catch (error) { refusal = error.code; }
    check(refusal === 'gi.spherical_invalid' && JSON.stringify(engine.sphericalGi()) === beforePatch,
      'invalid patch was not refused atomically', refusal);
  }

  const anchor = live();
  const translate = x => setPose({
    eye: [camera.eye[0] + x, ...camera.eye.slice(1)],
    target: [camera.target[0] + x, ...camera.target.slice(1)],
  });
  for (const x of [0.04, 0.08, 0.12, 0.16, 0.2]) {
    translate(x);
    await frame();
    const current = live();
    check(current.originEpoch === anchor.originEpoch
      && current.origin.every((value, i) => value === anchor.origin[i]), 'small camera steps moved the grid', current);
  }
  translate(1.25);
  await frame();
  const moved = record('camera-recentered', live());
  check(moved.originEpoch > anchor.originEpoch && moved.readyFaces < anchor.readyFaces,
    'recenter kept faces from the previous origin', { anchor, moved });
  await warm();

  const lit = live();
  const remote = engine.addLight({ kind: 'point', position: [250, 250, 250], range: 1, intensity: 15 });
  for (let i = 0; i < 4; i++) await frame();
  const distant = record('remote-light-added', live());
  check(distant.lightInvalidatedFaces === lit.lightInvalidatedFaces && distant.capturedFaces === lit.capturedFaces,
    'remote light invalidated occupied cells', { lit, distant });
  engine.removeLight(remote);
  for (let i = 0; i < 4; i++) await frame();
  const removedRemote = record('remote-light-removed', live());
  check(removedRemote.lightInvalidatedFaces === lit.lightInvalidatedFaces
    && removedRemote.capturedFaces === lit.capturedFaces, 'remote light removal invalidated occupied cells', removedRemote);

  const local = engine.addLight({ kind: 'point', position: [0.2, 0.35, -2.8], range: 0.7, intensity: 15 });
  let beforeLightChange = record('local-light-added', await warm());
  check(beforeLightChange.lightInvalidatedFaces > removedRemote.lightInvalidatedFaces
    && beforeLightChange.capturedFaces > removedRemote.capturedFaces,
  'adding a light over the open floor did not schedule its captures', beforeLightChange);
  for (const [id, change] of [
    ['local-light-edited', () => engine.updateLight(local, { intensity: 4 })],
    ['local-light-moved', () => engine.updateLight(local, { position: [0.25, 0.35, -2.8] })],
    ['local-light-removed', () => engine.removeLight(local)],
  ]) {
    change();
    const after = record(id, await warm());
    const invalidated = after.lightInvalidatedFaces - beforeLightChange.lightInvalidatedFaces;
    check(invalidated > 0 && invalidated < beforeLightChange.readyFaces
      && after.capturedFaces > beforeLightChange.capturedFaces,
    `${id} failed to target the contributing cells`, { beforeLightChange, after });
    beforeLightChange = after;
  }

  translate(48);
  await frame();
  const empty = record('empty', live());
  check(empty.cells === 0 && empty.readyFaces === 0 && empty.pendingFaces === 0,
    'empty origin retained occupied cells', empty);
  await capture('empty');
  const cube = engine.addMesh(box([0, 0, 0], [0.3, 0.3, 0.3]));
  const cubeX = camera.eye[0] + 48;
  const cubeNode = engine.addNode(null, [cubeX, 1.25, -2.1], [0, 0, 0, 1], [1, 1, 1]);
  const cubeInstance = engine.addMeshInstance(cubeNode, cube);
  const beforeGeometry = record('isolated-geometry', await warm(1));
  const isolatedProbe = await readProbe();
  check(isolatedProbe.faceValidMisses.some(count => count > 0), 'isolated geometry did not retain valid sky misses', isolatedProbe);
  engine.setTransform(cubeNode, [cubeX, 1.25, -4.2], [0, 0, 0, 1], [1, 1, 1]);
  await frame();
  const geometry = record('geometry-moved', live());
  check(geometry.originEpoch === beforeGeometry.originEpoch && geometry.lightRevision === beforeGeometry.lightRevision
    && (geometry.readyFaces < beforeGeometry.readyFaces || geometry.capturedFaces > beforeGeometry.capturedFaces),
  'moving occupied geometry did not refresh its captures', { beforeGeometry, geometry });
  await warm(1);
  engine.removeInstance(cubeInstance);
  engine.removeNode(cubeNode);
  engine.removeMesh(cube);
  await frame();
  const cleared = record('geometry-removed', live());
  check(cleared.originEpoch === geometry.originEpoch && cleared.cells === 0 && cleared.readyFaces === 0,
    'removed geometry retained occupied cells', cleared);
  translate(0);
  record('occupied-again', await warm());

  set('gi-grid', 'volume');
  await frame();
  const off = record('disabled');
  check(off?.active === false && off.gpuBytes === 0 && off.readyFaces === 0 && off.pendingFaces === 0
    && !pass(), 'did not release capture resources when disabled', off);
  const restoredOff = await capture('restored-off');
  const offEquivalence = difference(restoredOff, defaultOff);
  check(offEquivalence.changedChannels === 0, 'disabled HDR differs from the default-off renderer', offEquivalence);
  set('gi-grid', 'spherical');
  await frame();
  const restarted = record('reenabled-first-frame', live());
  check(restarted.readyFaces < occupied.readyFaces, 're-enable retained ready faces', restarted);
  record('reenabled-ready', await warm());

  for (const method of ['probes', 'restir']) {
    set('gi-grid', 'volume');
    set('indirect-solver', method);
    engine.setGlobalIllumination({ enabled: true, indirect: method, screenSpace: false });
    for (let i = 0; i < 48; i++) {
      await frame();
      if (i >= 7 && !engine.giStats().pendingRebuild) break;
    }
    const off = await capture(`${method}-off`);
    set('gi-grid', 'spherical');
    await warm();
    const on = await capture(`${method}-on`);
    const delta = difference(on, off);
    check(delta.changedChannels > 20, `${method} comparison shows no spherical response`, delta);
    comparisons.push({ id: method, on: `${method}-on`, off: `${method}-off`, ...delta });
  }
  return { json: { source, size: [width, height], camera, timeMs, probe, isolatedProbe, additiveError, groupTotal, offEquivalence,
    layout: 'linear scene RGBA16F, little-endian, top-down', captures, comparisons, transitions }, files };
}

export async function verifyGallerySphericalUi() {
  const select = id => document.querySelector(`[data-option="${id}"]`);
  const status = () => document.getElementById('status').textContent;
  const stats = () => document.getElementById('live-stats').textContent;
  const change = (input, value) => {
    if (input.type === 'checkbox') input.checked = value;
    else input.value = value;
    input.dispatchEvent(new Event('change', { bubbles: true }));
  };
  async function until(condition, description) {
    const deadline = performance.now() + 20000;
    while (!condition()) {
      check(performance.now() < deadline && !status().startsWith('failed:'), description, status());
      await new Promise(resolve => setTimeout(resolve, 50));
    }
  }
  change(document.getElementById('resolution'), '640x360');
  document.getElementById('run').click();
  const ready = () => /Spherical GI · ready cells [1-9]/.test(stats());
  await until(() => select('gi-grid'), 'gallery grid control was not mounted');
  check(select('gi-grid').value === 'volume' && select('spherical-debug').disabled, 'unexpected initial gallery controls');
  change(select('moving-objects'), false);
  change(select('gi-grid'), 'spherical');
  await until(ready, 'gallery did not show occupied spherical stats');
  const first = stats();
  for (const [id, value] of [
    ['spherical-subdivision', 4], ['spherical-exponent', 0.9],
    ['spherical-clip-start', 0.15], ['spherical-z-max', 14], ['spherical-debug', 'occupancy'],
  ]) {
    check(!select(id).disabled, `gallery ${id} was disabled`);
    change(select(id), value);
  }
  await until(() => ready() && stats() !== first, 'gallery stopped updating after hot controls');
  change(select('gi'), false);
  await until(() => select('gi-grid').disabled && !stats().includes('Spherical GI'), 'GI off left spherical active');
  change(select('gi'), true);
  await until(ready, 'GI on did not restore spherical captures');
  change(select('indirect-solver'), 'restir');
  change(select('gi-grid'), 'volume');
  await until(() => select('spherical-debug').disabled && !stats().includes('Spherical GI'), 'volume mode retained spherical controls');
  return { status: status(), first, final: stats() };
}
