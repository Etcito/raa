import { test } from 'node:test';
import assert from 'node:assert/strict';
import scene from '../scenes/m14-camera-effects.js';
import * as geometry from '../lib/geometry.js';
import * as objects from '../lib/objects.js';

async function build(id) {
  const record = { materials: [], meshes: [], nodes: [], instances: [], env: null };
  const engine = new Proxy({}, { get(_, name) {
    if (name === 'addSurfaceMaterial') return m => record.materials.push(m) - 1;
    if (name === 'addMesh') return m => record.meshes.push(m) - 1;
    if (name === 'addNode') return (_, t, r, s) => record.nodes.push({ t, r, s }) - 1;
    if (name === 'addMeshInstance') return (node, mesh) => record.instances.push({ node, mesh });
    if (name === 'applyEnvironment') return e => { record.env = e; };
    return () => {};
  } });
  const variant = scene.variants.find(v => v.id === id);
  await scene.build(engine, { geometry, objects }, variant);
  return { ...record, settings: variant.settings };
}

test('jungle sky and shaft radiance stay below the white-wash budget', async () => {
  const { env, settings } = await build('world-jungle');
  // Conservative unshadowed upper bound from shafts.wgsl (HG g=0.7), with
  // height_factor <= 1. Unlike fog, this additive integral has no extinction.
  const g = 0.7;
  const peakPhase = (1 + g) / (4 * Math.PI * (1 - g) ** 2);
  const s = settings.sunShafts;
  const maxShaft = env.sun.intensity * s.intensity * s.density * s.maxDistance * peakPhase;
  assert.ok(maxShaft < 1.1, `unattenuated shafts add ${maxShaft} HDR units`);
  const linear = c => c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  const luma = c => c.map(linear).reduce((v, c, i) => v + c * [0.2126, 0.7152, 0.0722][i], 0);
  assert.ok(luma(env.sky.horizon) * env.sky.intensity < 0.3);
  assert.ok(env.sky.zenith[2] > env.sky.zenith[0] * 4);
  assert.ok(settings.fog.start >= 16);
  assert.ok(env.post.toneMapping.exposureEv < 0);
});

test('jungle strip and raised splashes use opaque brown, and meshes are finite', async () => {
  const r = await build('world-jungle');
  const mud = r.materials.findIndex(m => m.transmission === 0 && m.roughness >= 0.6
    && m.color[0] > 1.5 * m.color[1] && m.color[1] > m.color[2]);
  assert.ok(mud >= 0, 'opaque low-gloss brown material');
  const meshes = r.meshes.map((m, i) => m.subMeshes[0].material === mud ? i : -1);
  const uses = r.instances.filter(i => meshes.includes(i.mesh));
  assert.equal(uses.length, 13, 'one strip, six raised splashes and six drops');
  const strip = r.meshes[uses[0].mesh];
  assert.ok(strip.positions.length > 1000, 'brown material is assigned to the strip grid');
  for (const m of r.meshes) {
    assert.ok(m.positions.every(Number.isFinite));
    assert.ok(m.normals.every(Number.isFinite));
  }
});
