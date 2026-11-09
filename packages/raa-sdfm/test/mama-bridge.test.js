import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';

import * as fakeMama from './fake-mama.js';
import { createBrowserPool, MATERIAL_FULL_FIELDS, meshDocument, toMamaLight } from '../src/mama-bridge.js';
import { canonicalJson } from '../src/canonical.js';
import { SdfmError } from '../src/errors.js';

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const SHIFTED = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.5, 1, 0, 1];

// Body in the shape compileMaterialGraph emits: constants read from binding 7 at `extra_words[b + K]`.
const PAINT_BODY = [
  '  let n0_color = vec3f(bitcast<f32>(extra_words[b + 0u]), bitcast<f32>(extra_words[b + 1u]), bitcast<f32>(extra_words[b + 2u]));',
  '  let n1_value = mat_pat_stripes(extra_words[b + 3u], p, bitcast<f32>(extra_words[b + 4u]));',
  '  return MaterialFull(n0_color * n1_value, 0.0, 0.5, vec3f(0.0), 0.0);',
].join('\n');

function doc({ paintConstants = [1061997773, 1048576000, 1040187392, 1, 1090519040], paintHelpers = ['pat_stripes'] } = {}) {
  return {
    nodes: {
      body: {
        kind: 'group', name: 'Body', asset: {},
        fake: {
          parts: [
            {
              partId: 'shell', name: 'Shell', frame: IDENTITY,
              subMeshes: [{ material: 'paint', vertices: 6, uvs: true }, { material: 'metal', vertices: 3 }],
            },
            { partId: 'lid', name: 'Lid', frame: SHIFTED, subMeshes: [{ material: 'paint', vertices: 3 }] },
          ],
        },
      },
      wheel: {
        kind: 'group', name: 'Wheel', asset: {},
        fake: { parts: [{ partId: '__root__', name: 'Wheel', frame: IDENTITY, subMeshes: [{ material: null, vertices: 3 }] }] },
      },
    },
    materials: {
      paint: { name: 'Paint', fake: { wgslBody: PAINT_BODY, helpers: paintHelpers, constants: paintConstants } },
      metal: {
        name: 'Metal',
        fake: { wgslBody: '  return MaterialFull(vec3f(0.8), 1.0, 0.3, vec3f(0.0), 0.0);', helpers: [], constants: [] },
      },
    },
  };
}

function rejectsWith(code) {
  return (e) => e instanceof SdfmError && e.code === code;
}

const light = (kind, fields = {}) => ({
  id: 'lamp', node: 'node', key: 7, t: [1, 2, 3], r: [0, 0, 0, 1], s: [2, 3, 4],
  kind, color: [1, 0.21404114048223255, 0.0031308049535603713], intensity: 3,
  range: 8, shadows: true, ...fields,
});

test('MAMA lights map all kinds, preserve metres and full authored spot angles', () => {
  const point = toMamaLight(light('point'));
  assert.deepEqual({ ...point, color: undefined }, {
    kind: 'point', position: [1, 2, 3], direction: [0, 0, -1], color: undefined,
    intensity: 3, range: 8, shadows: true,
  });
  point.color.forEach((c, i) => assert.ok(Math.abs(c - [1, 0.5, 0.04045][i]) < 1e-6));
  assert.deepEqual(toMamaLight(light('directional')).direction, [0, 0, -1]);
  const spot = toMamaLight(light('spot', { angleDeg: 70, innerAngleDeg: 40 }));
  assert.equal(spot.angleDeg, 70);
  assert.equal(spot.innerAngleDeg, 40);
  const omitted = toMamaLight(light('spot', { angleDeg: 35 }));
  assert.equal(Object.hasOwn(omitted, 'innerAngleDeg'), false);
  assert.equal(point.range, 8);
  assert.deepEqual(toMamaLight(light('area', { size: [2.5, 1.25] })).size, [2.5, 1.25]);
});

test('MAMA light placement rotates and normalizes travel direction without scaling metres', () => {
  const half = Math.sqrt(0.5);
  const mapped = toMamaLight(light('area', {
    t: [4, 5, 6], r: [0, half, 0, half], s: [2, 3, 7], size: [4, 6], range: 12,
  }));
  assert.ok(Math.abs(mapped.direction[0] + 1) < 1e-12);
  assert.ok(Math.abs(mapped.direction[2]) < 1e-12);
  assert.deepEqual(mapped.position, [4, 5, 6]);
  assert.equal(mapped.range, 12);
  assert.deepEqual(mapped.size, [4, 6]);
});

test('MAMA light mapping returns fresh arrays and sRGB API color round-trips to linear once', () => {
  const source = light('spot', { angleDeg: 90, innerAngleDeg: 45 });
  const first = toMamaLight(source);
  const second = toMamaLight(source);
  for (const field of ['position', 'direction', 'color']) assert.notEqual(first[field], second[field]);
  const sourceArea = light('area', { size: [2, 3] });
  const area = toMamaLight(sourceArea);
  assert.notEqual(area.size, sourceArea.size);
  assert.deepEqual(area.size, [2, 3]);
  assert.notEqual(first.position, source.t);
  assert.notEqual(first.color, source.color);
  const apiToLinear = (c) => c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  first.color.forEach((c, i) => assert.ok(Math.abs(apiToLinear(c) - source.color[i]) < 1e-7));
  first.direction[0] = 99;
  first.position[0] = 99;
  assert.deepEqual(source.t, [1, 2, 3]);
  assert.equal(source.r[0], 0);
});

test('malformed and degenerate MAMA light mappings refuse with sdfm.scene_format paths', () => {
  assert.throws(() => toMamaLight(light('point', { t: [0, 1] })), (e) =>
    rejectsWith('sdfm.scene_format')(e) && e.path === '/lights/t');
  assert.throws(() => toMamaLight(light('point', { s: [1, 1, 0] })), (e) =>
    rejectsWith('sdfm.scene_format')(e) && e.path === '/lights/s');
});

test('a group value refusal becomes sdfm.mesher_refused naming the group, message and fix', async () => {
  const d = doc();
  d.nodes.wheel.fake = { refuse: { message: 'the mesher produced no triangle' } };
  d.nodes.spoiler = { kind: 'group', name: 'Spoiler', asset: {}, fake: { refuse: { message: 'second refusal' } } };
  await assert.rejects(meshDocument(fakeMama, d, { pool: fakeMama.fakePool() }), (e) => {
    assert.ok(e instanceof SdfmError);
    assert.equal(e.code, 'sdfm.mesher_refused');
    assert.equal(e.path, '/document');
    assert.match(e.message, /^sdfm\.mesher_refused: wheel: group 'Wheel' cannot be meshed/);
    assert.match(e.message, /the mesher produced no triangle/);
    assert.match(e.message, /Pick another mesher in the Mesher section\./);
    assert.doesNotMatch(e.message, /second refusal/);
    return true;
  });
});

test('a requested id that is not a group is refused too', async () => {
  await assert.rejects(
    meshDocument(fakeMama, doc(), { pool: fakeMama.fakePool(), groups: ['body', 'nope'] }),
    (e) => rejectsWith('sdfm.mesher_refused')(e) && /nope: 'nope' is not a node of the scene/.test(e.message),
  );
});

test('any other MAMA exception becomes sdfm.worker', async () => {
  const d = doc();
  d.nodes.wheel.fake = { throw: 'mesher crashed' };
  await assert.rejects(meshDocument(fakeMama, d, { pool: fakeMama.fakePool() }),
    (e) => rejectsWith('sdfm.worker')(e) && /mesher crashed/.test(e.message));
  await assert.rejects(meshDocument(fakeMama, { bad: true }, { pool: fakeMama.fakePool() }), rejectsWith('sdfm.worker'));
});

test('meshAssets gets the caller pool and groups, and parts are flat with their MAMA frame', async () => {
  const pool = fakeMama.fakePool(3);
  const out = await meshDocument(fakeMama, doc(), { pool, groups: ['wheel', 'body'] });
  const call = fakeMama.calls.findLast((c) => c.fn === 'meshAssets');
  assert.equal(call.options.pool, pool);
  assert.deepEqual(call.options.groups, ['wheel', 'body']);
  assert.deepEqual(Object.keys(call.options).sort(), ['groups', 'pool']);

  assert.deepEqual(out.groups.map((g) => g.group), ['wheel', 'body']);
  const body = out.groups[1];
  assert.deepEqual(body.parts.map((p) => [p.partId, p.name]), [['shell', 'Shell'], ['lid', 'Lid']]);
  for (const part of body.parts) {
    assert.deepEqual(Object.keys(part), ['partId', 'name', 'frame', 'meshes']);
    assert.ok(part.frame instanceof Float64Array);
  }
  assert.deepEqual(Array.from(body.parts[1].frame), SHIFTED);
});

test('each MAMA sub-mesh becomes one mesh with copied typed arrays', async () => {
  const out = await meshDocument(fakeMama, doc(), { pool: fakeMama.fakePool() });
  const shell = out.groups[0].parts[0];
  assert.equal(shell.meshes.length, 2);
  assert.deepEqual(shell.meshes.map((m) => m.material), ['paint', 'metal']);
  const [paint, metal] = shell.meshes;
  assert.deepEqual(Object.keys(paint), ['material', 'positions', 'normals', 'uvs', 'indices']);
  assert.ok(paint.positions instanceof Float32Array && paint.positions.length === 18);
  assert.ok(paint.normals instanceof Float32Array && paint.normals.length === 18);
  assert.ok(paint.uvs instanceof Float32Array && paint.uvs.length === 12);
  assert.ok(paint.indices instanceof Uint32Array && paint.indices.length === 6);
  assert.equal(metal.uvs, null);
  assert.equal(out.groups[1].parts[0].meshes[0].material, null);
  assert.deepEqual(out.materials.map((m) => m.name), ['metal', 'paint']);
});

test('the material key is the sha256 of its canonical record and is stable', async () => {
  const a = await meshDocument(fakeMama, doc(), { pool: fakeMama.fakePool() });
  const b = await meshDocument(fakeMama, doc(), { pool: fakeMama.fakePool() });
  const paint = a.materials.find((m) => m.name === 'paint');
  assert.deepEqual(Object.keys(paint), ['name', 'key', 'body', 'helpers', 'constants', 'fields']);
  assert.equal(paint.body, PAINT_BODY);
  assert.deepEqual(paint.helpers.map((h) => h.id), ['base', 'axes', 'tri', 'pat_stripes']);
  assert.equal(paint.helpers[0].wgsl, fakeMama.BASE_WGSL);
  assert.equal(paint.helpers[3].wgsl, fakeMama.HELPERS.pat_stripes.code);
  assert.deepEqual(paint.constants, [1061997773, 1048576000, 1040187392, 1, 1090519040]);
  assert.deepEqual(paint.fields, [...MATERIAL_FULL_FIELDS]);
  assert.deepEqual(paint.fields, ['base_color', 'metallic', 'roughness', 'emissive', 'height']);
  const { body, helpers, constants, fields } = paint;
  const expected = createHash('sha256').update(canonicalJson({ body, helpers, constants, fields })).digest('hex');
  assert.equal(paint.key, expected);
  assert.match(paint.key, /^[0-9a-f]{64}$/);
  assert.deepEqual(a.materials.map((m) => m.key), b.materials.map((m) => m.key));

  const edited = await meshDocument(fakeMama, doc({ paintConstants: [0, 0, 0, 1, 1090519040] }), { pool: fakeMama.fakePool() });
  assert.notEqual(edited.materials.find((m) => m.name === 'paint').key, paint.key);
});

test('an unresolvable helper throws sdfm.material_unmappable naming it', async () => {
  await assert.rejects(
    meshDocument(fakeMama, doc({ paintHelpers: ['pat_missing'] }), { pool: fakeMama.fakePool() }),
    (e) => rejectsWith('sdfm.material_unmappable')(e) && /paint/.test(e.message) && /'pat_missing'/.test(e.message),
  );
});

test('a material MAMA failed to build throws sdfm.material_unmappable', async () => {
  const d = doc();
  d.materials.metal.fake = { error: 'graph has a cycle' };
  await assert.rejects(meshDocument(fakeMama, d, { pool: fakeMama.fakePool() }),
    (e) => rejectsWith('sdfm.material_unmappable')(e) && /metal/.test(e.message) && /graph has a cycle/.test(e.message));
});

test('the browser pool comes from setupBrowserHost', () => {
  const pool = createBrowserPool(fakeMama, { workerUrl: 'w.js', manifoldWasmUrl: 'manifold.wasm', size: 2 });
  assert.equal(pool.size, 2);
  const call = fakeMama.calls.findLast((c) => c.fn === 'setupBrowserHost');
  assert.deepEqual(call.options, { workerUrl: 'w.js', manifoldWasmUrl: 'manifold.wasm', size: 2 });
  assert.throws(() => createBrowserPool({}, { workerUrl: 'w.js', manifoldWasmUrl: 'm.wasm' }), rejectsWith('sdfm.worker'));
});

test('the browser pool forwards workerMode blob (opaque-origin pages)', () => {
  createBrowserPool(fakeMama, { workerUrl: 'w.js', manifoldWasmUrl: 'manifold.wasm', workerMode: 'blob' });
  const call = fakeMama.calls.findLast((c) => c.fn === 'setupBrowserHost');
  assert.deepEqual(call.options, { workerUrl: 'w.js', manifoldWasmUrl: 'manifold.wasm', workerMode: 'blob' });
});

test('MAMA BASE_WGSL is the first helper, so a body calling mat_clamp01 maps to WGSL defining it', async () => {
  const out = await meshDocument(fakeMama, doc(), { pool: fakeMama.fakePool() });
  for (const m of out.materials) {
    assert.equal(m.helpers[0].id, 'base');
    assert.equal(m.helpers[0].wgsl, fakeMama.BASE_WGSL);
  }
  const { BASE_WGSL, ...noBase } = fakeMama;
  assert.ok(BASE_WGSL.includes('fn mat_clamp01('));
  await assert.rejects(meshDocument(noBase, doc(), { pool: fakeMama.fakePool() }),
    (e) => rejectsWith('sdfm.material_unmappable')(e) && /BASE_WGSL/.test(e.message));
});

test('importMama exposes the root serializer for URL-based asset loading', async () => {
  const { mkdtemp, mkdir, writeFile, rm } = await import('node:fs/promises');
  const { tmpdir } = await import('node:os');
  const { join } = await import('node:path');
  const { pathToFileURL } = await import('node:url');
  const { importMama } = await import('../src/mama-bridge.js');
  const dir = await mkdtemp(join(tmpdir(), 'raa-mama-test-'));
  try {
    await mkdir(join(dir, 'core/game'), { recursive: true });
    await mkdir(join(dir, 'core/material-graph'), { recursive: true });
    await writeFile(join(dir, 'package.json'), '{"type":"module"}');
    await writeFile(join(dir, 'index.js'), 'export const serializeDocument = d => JSON.stringify(d); export const readSdfm = d => d;');
    await writeFile(join(dir, 'core/game/mesh-assets.js'), 'export const meshAssets = async () => [];');
    await writeFile(join(dir, 'core/material-graph/helpers.js'), 'export const HELPERS = {}; export const BASE_WGSL = "";');
    const mama = await importMama(pathToFileURL(dir + '/'), { browser: false });
    assert.equal(mama.serializeDocument({ test: 1 }), '{"test":1}');
  } finally { await rm(dir, { recursive: true, force: true }); }
});
