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
