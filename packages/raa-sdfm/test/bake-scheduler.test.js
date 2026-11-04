import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BAKE_RUN_SHARE, bakeTextureCount, createBakeScheduler } from '../src/bake-scheduler.js';
import { BAKED_FUNCTION, BAKED_NO_TEXTURE, BAKE_ATLAS_BUDGET } from '../src/field-bake.js';

const quad = {
  positions: Float32Array.of(0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0),
  normals: Float32Array.of(0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1),
  uvs: Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1),
  indices: Uint32Array.of(0, 1, 2, 0, 2, 3),
};
const CONSTANT = { color: [1, 0, 0], metallic: 0, roughness: 0.5, emissive: [0, 0, 0] };
const shading = { program: [], single: true, look: { roughness: 0.3, metallic: 0.2 }, operands: [{ color: [1, 0, 0], material: null, toNode: [], constant: CONSTANT }] };

function engine() {
  const calls = { functions: [], data: new Map(), written: [], surfaces: [], textures: new Map(), removed: [], freed: [] };
  let next = 1;
  return {
    calls,
    addMaterialFunction(name) { if (calls.functions.includes(name)) { const e = new Error('dup'); e.code = 'material.fn_duplicate'; throw e; } calls.functions.push(name); },
    addMaterialData(words) { const id = next++; calls.data.set(id, Uint32Array.from(words)); return id; },
    setMaterialData(id, words) { calls.written.push([id, Uint32Array.from(words)]); calls.data.set(id, Uint32Array.from(words)); },
    removeMaterialData(id) { calls.freed.push(id); calls.data.delete(id); },
    addSurfaceMaterial(desc) { calls.surfaces.push(desc); return next++; },
    addTexture(t) { const id = next++; calls.textures.set(id, t); return id; },
    removeTexture(id) { calls.removed.push(id); calls.textures.delete(id); },
  };
}

// A baker that answers what it was asked, after a tick (fails parts whose label says so).
function baker(log = []) {
  return {
    pipelines: 1,
    async bake(input) {
      log.push(input.size);
      await new Promise((r) => setTimeout(r, 1));
      if (input.shading.fail) throw new Error('GPU lost');
      const img = new Uint8Array(input.size * input.size * 4);
      return { size: input.size, normal: img, color: img, emissive: null, missing: [], pipeline: 'sdfb.x' };
    },
  };
}

const make = (e, extra = {}) => createBakeScheduler(e, { device: () => ({}), field: 'fn pmf_eval() {}', textureSize: () => 256, baker: baker(extra.log), ...extra });

test('bakeTextureCount: emissive texture only when an operand can glow', () => {
  assert.equal(bakeTextureCount(shading), 2);
  assert.equal(bakeTextureCount({ operands: [{ constant: { ...CONSTANT, emissive: [0, 1, 0] } }] }), 3);
  assert.equal(bakeTextureCount({ operands: [{ material: 'graph' }] }), 3);
});

test('prepare: the shared baked function once, a data block naming no texture, tangents; nothing baked', () => {
  const e = engine();
  const s = make(e);
  const a = s.prepare({ shading, materials: [], ...quad, label: 'a' });
  const b = s.prepare({ shading, materials: [], ...quad, label: 'b' });
  assert.deepEqual(e.calls.functions, [BAKED_FUNCTION]);
  assert.equal(a.state, 'idle');
  assert.deepEqual(Array.from(e.calls.data.get(a.data)).slice(1), [BAKED_NO_TEXTURE, BAKED_NO_TEXTURE, BAKED_NO_TEXTURE]);
  assert.deepEqual(e.calls.surfaces[0], { function: BAKED_FUNCTION, data: a.data, roughness: 0.3, metallic: 0.2 });
  assert.equal(a.tangents.length, 16);
  assert.notEqual(a.material, b.material);
  assert.equal(e.calls.textures.size, 0);
  assert.throws(() => s.prepare({ shading, materials: [], ...quad, uvs: undefined }), /UV atlas/);
});

test('queue: one run bakes what a scene queued, writes the texture ids, reports the changes; a second queue bakes nothing', async () => {
  const e = engine();
  const changed = [];
  const s = make(e, { onChange: (parts) => changed.push(...parts.map((p) => p.label)) });
  const parts = ['a', 'b', 'c'].map((label) => s.prepare({ shading, materials: [], ...quad, label }));
  s.queue(parts.slice(0, 2));
  s.queue(parts.slice(2));
  assert.ok(parts.every((p) => p.state === 'queued'));
  await s.done();
  assert.ok(parts.every((p) => p.state === 'done' && p.textures.length === 2 && p.size === 256));
  assert.deepEqual(changed.sort(), ['a', 'b', 'c']);
  for (const p of parts) {
    const words = e.calls.data.get(p.data);
    assert.deepEqual([words[1], words[2], words[3]], [p.textures[0], p.textures[1], BAKED_NO_TEXTURE]);
  }
  const st = s.stats();
  assert.equal(st.parts, 3);
  assert.equal(st.resident.textures, 6);
  assert.equal(st.used, 3 * 256 * 256 * 2);
  s.queue(parts);
  await s.done();
  assert.equal(s.stats().parts, 3, 'done parts are never baked again');
});

test('a run takes at most BAKE_RUN_SHARE of the atlas left: the sides are fitted down', async () => {
  const e = engine();
  const log = [];
  // 2 parts x 2 textures x 1024^2 asked with a budget of 1M texels: 750k for this run.
  const s = make(e, { log, textureSize: () => 1024, budget: 1 << 20 });
  const parts = [0, 1].map(() => s.prepare({ shading, materials: [], ...quad }));
  s.queue(parts);
  await s.done();
  const total = parts.reduce((n, p) => n + p.texels, 0);
  assert.ok(total <= (1 << 20) * BAKE_RUN_SHARE, `${total} texels`);
  assert.ok(log.every((size) => size < 1024));
  assert.equal(BAKE_RUN_SHARE, 0.75);
  assert.ok(BAKE_ATLAS_BUDGET > 0);
});

test('a failed bake keeps the part failed with one report; an atlas-full upload fails the rest of the run', async () => {
  const e = engine();
  const failed = [];
  const s = make(e, { onFailed: (p, why) => failed.push(`${p.label}: ${why}`) });
  const bad = s.prepare({ shading: { ...shading, fail: true }, materials: [], ...quad, label: 'bad' });
  const good = s.prepare({ shading, materials: [], ...quad, label: 'good' });
  s.queue([bad, good]);
  await s.done();
  assert.equal(bad.state, 'failed');
  assert.equal(good.state, 'done');
  assert.deepEqual(failed, ['bad: bake failed: GPU lost']);
  const full = engine();
  full.addTexture = () => { const err = new Error('texture.atlas_full: no room'); err.code = 'texture.atlas_full'; throw err; };
  const f = make(full);
  const parts = [0, 1].map(() => f.prepare({ shading, materials: [], ...quad }));
  f.queue(parts);
  await f.done();
  assert.ok(parts.every((p) => p.state === 'failed'));
  assert.equal(f.stats().failed, 2);
});

test('release: textures and texels go back, the data block is freed; a part released while baking uploads nothing', async () => {
  const e = engine();
  const s = make(e);
  const a = s.prepare({ shading, materials: [], ...quad });
  s.queue([a]);
  await s.done();
  const ids = [...a.textures];
  s.release(a);
  assert.deepEqual(e.calls.removed, ids);
  assert.deepEqual(e.calls.freed, [a.data]);
  assert.equal(s.stats().used, 0);
  s.release(a);
  assert.equal(e.calls.freed.length, 1, 'release is idempotent');
  const b = s.prepare({ shading, materials: [], ...quad });
  s.queue([b]);
  await new Promise((r) => setTimeout(r, 0));
  s.release(b);
  await s.done();
  assert.equal(b.state, 'released');
  assert.equal(e.calls.textures.size, 0);
});

test('no GPU device: every queued part fails with the reason', async () => {
  const e = engine();
  const s = make(e, { device: () => undefined });
  const a = s.prepare({ shading, materials: [], ...quad });
  s.queue([a]);
  await s.done();
  assert.equal(a.state, 'failed');
  assert.match(a.error, /no GPU device/);
});
