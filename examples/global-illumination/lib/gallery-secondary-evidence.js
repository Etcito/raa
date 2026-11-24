import { loadGalleryFit } from './gallery-path-model.js';

const seed = 73029;
const steps = [['sky', 1], ['sun', 2], ['other', 4]];
const lobes = ['diffuse-indirect', 'rough-specular'];
const bytesPerRoughHit = 48;
const miss = 0xffffffff;

function roughTap(sample, check) {
  const { roughEvidence: evidence, size } = sample;
  check(evidence?.size.join() === size.join() && evidence.data instanceof Uint8Array
    && evidence.frame === sample.frame && evidence.data.byteLength === size[0] * size[1] * bytesPerRoughHit,
  'Missing fixed-seed rough secondary evidence', { id: sample.id, size });
  return new DataView(evidence.data.buffer, evidence.data.byteOffset, evidence.data.byteLength);
}

function roughSourceCounts(view, check) {
  const counts = { hit: 0, miss: 0, none: 0, volume: 0, spherical: 0 };
  for (let offset = 0; offset < view.byteLength; offset += bytesPerRoughHit) {
    const triangle = view.getUint32(offset + 12, true);
    const source = view.getUint32(offset + 44, true);
    const trust = view.getFloat32(offset + 28, true);
    check(source <= 2 && Number.isFinite(trust) && trust >= 0 && trust <= 1, 'Invalid rough cache source or trust',
      { offset, source, trust });
    counts[triangle === miss ? 'miss' : 'hit']++;
    counts[['none', 'volume', 'spherical'][source]]++;
    if (source) check(triangle !== miss, 'Cache reported a source before geometry was hit', offset);
  }
  return counts;
}

function proveRoughFirstHit(first, second, check) {
  let hits = 0, identicalDirect = 0, changedTail = 0;
  for (let offset = 0; offset < first.byteLength; offset += bytesPerRoughHit) {
    for (let lane = 0; lane < 16; lane += 4) {
      check(first.getUint32(offset + lane, true) === second.getUint32(offset + lane, true),
        'Rough cache changed the first actual BVH intersection', { pixel: offset / bytesPerRoughHit, lane });
    }
    if (first.getUint32(offset + 12, true) === miss) continue;
    hits++;
    let changed = false;
    for (let lane = 0; lane < 12; lane += 4) {
      check(first.getUint32(offset + 16 + lane, true) === second.getUint32(offset + 16 + lane, true),
        'Rough cache changed material-weighted first-hit direct light', { pixel: offset / bytesPerRoughHit, lane });
      changed ||= Math.abs(first.getFloat32(offset + 32 + lane, true)
        - second.getFloat32(offset + 32 + lane, true)) > 1e-6;
    }
    identicalDirect++;
    changedTail += Number(changed);
  }
  return { hits, identicalDirect, changedTail };
}

function assertLearned(sample, grid, check) {
  const stats = sample.trace;
  check(stats.modelPixels > 0 && stats.refreshPixels > 0 && stats.specularRays > 0
    && stats.diffuseRays > 0 && stats.referencePixels === 0 && stats.hybridLobeSpp > 0,
  `No fitted secondary dispatch on ${grid}`, stats);
}


const firstRayKeys = ['eligible', 'acceptedScreen', 'traced', 'geometryHit', 'mirrorRays',
  'firstHitRays', 'firstHitHits', 'firstHitHash'];

function sameFirstRay(a, b, check) {
  const left = a.trace, right = b.trace;
  check(left.firstHitHits > 0 && firstRayKeys.every(key => left[key] === right[key]),
    'Cache policy altered exact first-hit counts, triangle/world-position digest, or confidence decisions',
    { first: a.id, second: b.id, left, right });
}

export function inspectMirrorCoverage(cold, warm, residency, check) {
  const stats = warm.trace;
  check(cold.trace && Number.isInteger(stats.sphericalDiffuseHits) && Number.isInteger(stats.firstHitHash),
    'Renderer lacks reflected geometry cache diagnostics', stats);
  const supported = stats.sphericalDiffuseOffscreenSupported, inGrid = stats.sphericalDiffuseOffscreenInGrid;
  const readiness = { cells: residency.cells, readyCells: residency.readyCells, readyFaces: residency.readyFaces,
    pendingFaces: residency.pendingFaces, inFlightFaces: residency.inFlightFaces };
  check(readiness.cells <= 128 && readiness.readyCells === readiness.cells && readiness.readyCells > 0,
    'Unexpected default spherical residency', readiness);
  return { readiness, cold: cold.trace, warm: stats, inGrid, supported,
    zeroCoverage: stats.sphericalDiffuseHits === 0 || (inGrid > 0 && supported === 0) };
}

export async function measureSharpSecondary({ engine, capture, check, compare }) {
  const policy = indirectGrid => {
    engine.setMeshTracing({ indirectGrid });
    // The default UI continues to use screen confidence. Only this bracket
    // bypasses it, and the captured counters must show all eligible rays traced.
    engine.setPassEnabled('ssr.confidence', false);
  };
  const groups = channelMask => engine.setSphericalGi({ channelMask });
  try {
    policy('auto');
    const active = await capture('sharp-bvh-spherical', true, true);
    check(active.trace.eligible > 0 && active.trace.acceptedScreen === 0
      && active.trace.traced === active.trace.eligible && active.trace.firstHitRays === active.trace.traced,
    'Forced-BVH bracket accepted a screen hit or skipped a mirror first ray', active.trace);
    policy('off');
    const bypass = await capture('sharp-bvh-cache-bypassed', true, true);
    sameFirstRay(active, bypass, check);
    check(bypass.trace.sphericalDiffuseHits === 0 && bypass.trace.volumeDiffuseHits === 0,
      'Forced-BVH bypass sampled a cache', bypass.trace);
    const effect = compare(active.rgb, bypass.rgb);
    check(effect.changedPixels > 0 && effect.rmse > 0,
      'Spherical cache supplied no physical reflection contribution', { effect, active: active.trace });

    policy('volume');
    const volume = await capture('sharp-bvh-forced-volume', true, true);
    sameFirstRay(active, volume, check);
    const volumeEffect = compare(active.rgb, volume.rgb);
    check(volume.trace.sphericalDiffuseHits === 0 && volumeEffect.changedPixels > 0,
      'Volume and spherical terminal shading were not isolated', { volume: volume.trace, volumeEffect });
    policy('auto');
    groups(0);
    const zero = await capture('sharp-bvh-groups-off', true, true);
    const responses = [];
    for (const [name, mask] of steps) {
      groups(mask);
      const isolated = await capture(`sharp-bvh-${name}`, true, true);
      sameFirstRay(active, isolated, check);
      const response = compare(isolated.rgb, zero.rgb);
      check(response.changedPixels > 0 && response.rmse > 0, `Shot did not exercise physical mirror ${name} lighting`, response);
      responses.push({ name, ...response });
    }
    groups(7);
    const physical = await capture('sharp-bvh-debug-off', true, true);
    const forward = await capture('sharp-bvh-overlay-off', false);
    engine.setSphericalGi({ debug: 'occupancy' });
    const debug = await capture('sharp-bvh-debug-occupancy', true, true);
    const overlay = await capture('sharp-bvh-overlay-occupancy', false);
    sameFirstRay(physical, debug, check);
    const sceneLinear = compare(physical.rgb, debug.rgb), presentation = compare(forward.rgb, overlay.rgb);
    check(sceneLinear.peak === 0 && presentation.changedPixels > 0,
      'Debug overlay changed physical reflected lighting or was invisible', { sceneLinear, presentation });
    return { active: active.trace, bypass: bypass.trace, volume: volume.trace, effect, volumeEffect,
      groups: responses, debug: { sceneLinear, presentation } };
  } finally {
    engine.setSphericalGi({ channelMask: 7, debug: 'off' });
    engine.setMeshTracing({ indirectGrid: 'auto' });
  }
}
export async function measureLearnedSecondary({ engine, state, set, settle, capture, check, compare }) {
  const model = await loadGalleryFit();
  check(model, 'Gallery fitted specular package required');
  engine.setMeshPathModel(model.bytes);
  set('model-assisted-path', true);
  const configure = settings => engine.setSphericalGi(settings);
  const policy = indirectGrid => engine.setMeshTracing({ indirectGrid });
  const cases = [];
  const byGrid = {};
  const fit = async (id, grid) => {
    const sample = await capture(id, true, true, seed);
    assertLearned(sample, grid, check);
    return sample;
  };

  for (const grid of ['volume', 'spherical']) {
    set('gi-grid', grid);
    await settle(grid);
    byGrid[grid] = {};
    set('lighting-view', 'combined');
    const combined = await fit(`learned-${grid}-combined`, grid);
    if (grid === 'spherical') {
      set('lighting-view', 'direct');
      policy('auto');
      const direct = await fit('learned-spherical-direct-auto', grid);
      policy('off');
      const unchanged = await fit('learned-spherical-direct-bypass', grid);
      policy('auto');
      sameFirstRay(direct, unchanged, check);
      const comparison = compare(direct.rgb, unchanged.rgb);
      check(comparison.peak === 0, 'Cache policy changed the isolated direct result', comparison);
      cases.push({ grid, lobe: 'direct', comparison });
    }
    for (const lobe of lobes) {
      set('lighting-view', lobe);
      if (grid === 'spherical') {
        policy('auto');
        configure({ channelMask: 7, debug: 'off' });
      }
      const active = await fit(`learned-${grid}-${lobe}`, grid);
      const compared = compare(active.rgb, combined.rgb);
      check(compared.changedPixels > 20, `Learned ${grid} isolation did not change HDR`, { lobe, compared });
      if (grid === 'volume') {
        byGrid[grid][lobe] = active;
        cases.push({ grid, lobe, combined: compared, stats: active.trace });
        continue;
      }

      policy('off');
      const bypass = await fit(`learned-${grid}-${lobe}-off`, grid);
      policy('auto');
      configure({ channelMask: 0 });
      const zero = await fit(`learned-${grid}-${lobe}-groups-off`, grid);
      const groups = [];
      for (const [name, mask] of steps) {
        configure({ channelMask: mask });
        const isolated = await fit(`learned-${grid}-${lobe}-${name}`, grid);
        const response = compare(isolated.rgb, zero.rgb);
        check(response.changedPixels > 0 && response.rmse > 0, `Learned ${lobe} has no ${name} cache response`, response);
        groups.push({ name, ...response });
      }
      configure({ channelMask: 7 });
      sameFirstRay(active, bypass, check);
      const physical = compare(active.rgb, bypass.rgb);
      const volume = compare(active.rgb, byGrid.volume[lobe].rgb);
      check(physical.changedPixels > 10 && volume.changedPixels > 10,
        `Learned spherical ${lobe} cache did not change downstream HDR`, { physical, volume });
      let rough;
      if (lobe === 'rough-specular') {
        const a = roughTap(active, check), b = roughTap(bypass, check);
        rough = { active: roughSourceCounts(a, check), bypass: roughSourceCounts(b, check),
          invariance: proveRoughFirstHit(a, b, check) };
        check(rough.active.spherical > 0 && rough.bypass.spherical === 0
          && rough.invariance.hits > 0 && rough.invariance.changedTail > 0,
        'Rough fitted specular did not sample a spherical tail after its first hit', rough);
      }
      cases.push({ grid, lobe, combined: compared, physical, volume, groups, stats: active.trace, rough });
    }
  }

  set('lighting-view', 'rough-grid');
  const roughnessSweep = [];
  for (const roughness of [0.08, 0.28, 0.55, 0.8]) {
    state.setEvidenceFloorRoughness(roughness);
    const label = String(Math.round(roughness * 100)).padStart(2, '0');
    const outputs = {};
    for (const grid of ['volume', 'spherical']) {
      set('gi-grid', grid);
      await settle(grid);
      const sample = await fit(`roughness-${label}-${grid}`, grid);
      outputs[grid] = { sample, count: roughSourceCounts(roughTap(sample, check), check) };
    }
    const comparison = proveRoughFirstHit(roughTap(outputs.spherical.sample, check),
      roughTap(outputs.volume.sample, check), check);
    roughnessSweep.push({ roughness, volume: outputs.volume.count, spherical: outputs.spherical.count, ...comparison });
  }
  state.setEvidenceFloorRoughness(0.28);
  set('lighting-view', 'combined');
  return { modelSha256: model.fit.weights.sha256, seed, cases, roughnessSweep };
}
