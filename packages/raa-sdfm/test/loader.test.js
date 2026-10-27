import { serializeDocument } from './fake-mama.js';
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import * as fakeMama from './fake-mama.js';
import { createFakeEngine, FakeRaaError } from './fake-engine.js';
import { createSdfmLoader, SdfmError } from '../index.js';
import { documentHash } from '../src/canonical.js';

// Fixture rigid.asset.json: group 'cart' meshed by the fake MAMA into two rigid parts, Body (sub-meshes
// paint, metal, paint) and Wheel (metal, rotated 90 deg about Y). MAMA has no parent link, so the
// loader's parts are flat under the spawn root; a parent between parts is exercised through spawn.
const fixtureText = readFileSync(new URL('./fixtures/rigid.asset.json', import.meta.url), 'utf8');
const fixture = () => JSON.parse(fixtureText);

function setup() {
  const engine = createFakeEngine();
  let mamaCalls = 0;
  const loader = createSdfmLoader({
    engine,
    mama: async () => {
      mamaCalls += 1;
      return fakeMama;
    },
  });
  return { engine, loader, mamaCalls: () => mamaCalls };
}

const meshCalls = () => fakeMama.calls.filter((c) => c.fn === 'meshAssets').length;

/** A copy of the fixture with its document changed by `edit` and a matching hash. */
async function variant(id, edit) {
  const asset = fixture();
  asset.id = id;
  edit(asset.document);
  asset.hash = await documentHash(asset.document, serializeDocument);
  return asset;
}

function code(c) {
  return (e) => {
    assert.ok(e instanceof SdfmError, `SdfmError expected, got ${e}`);
    assert.equal(e.code, c);
    assert.ok(e.message.length > c.length + 2, 'non-empty message');
    return true;
  };
}

test('load returns an SdfmAsset with parts, and anchors, bounds and collision untouched', async () => {
  const { engine, loader } = setup();
  const input = fixture();
  const asset = await loader.load(input);
  assert.equal(asset.id, 'raatest.cart');
  assert.equal(asset.hash, input.hash);
  assert.equal(asset.name, 'Cart');
  assert.equal(asset.category, 'prop');
  assert.equal(asset.world, 'test');
  assert.deepEqual(asset.anchors, input.anchors);
  assert.deepEqual(asset.bounds, input.bounds);
  assert.deepEqual(asset.collision, input.collision);
  assert.equal(asset.parts.length, 2);

  const [body, wheel] = asset.parts;
  assert.deepEqual(
    { name: body.name, groupId: body.groupId, rigid: body.rigid, parent: body.parent, t: body.t, r: body.r },
    { name: 'Body', groupId: 'cart', rigid: true, parent: null, t: [0, 0.5, 0], r: [0, 0, 0, 1] },
  );
  assert.equal(wheel.name, 'Wheel');
  assert.deepEqual(wheel.t, [0.4, -0.2, 0.3]);
  const h = Math.SQRT1_2;
  wheel.r.forEach((v, i) => assert.ok(Math.abs(v - [0, h, 0, h][i]) < 1e-12, `r = ${wheel.r}`));

  // Body: one mesh, one sub-mesh per material (paint 6 + 3 indices, then metal 3), materials in sub-mesh order.
  // MAMA lists a group's materials sorted by id: metal, then paint.
  const metal = engine.calls.addSurfaceMaterial[0].id;
  const paint = engine.calls.addSurfaceMaterial[1].id;
  const desc = engine.meshes.get(body.mesh);
  assert.deepEqual(desc.subMeshes, [{ first: 0, count: 9, material: paint }, { first: 9, count: 3, material: metal }]);
  assert.deepEqual(body.materials, [paint, metal]);
  assert.equal(desc.positions.length, 12 * 3);
  assert.equal(desc.uvs.length, 12 * 2);
  assert.deepEqual(wheel.materials, [metal]);
  assert.equal(engine.meshes.get(wheel.mesh).uvs, undefined);
  for (const call of engine.calls.addSurfaceMaterial) {
    assert.equal(call.desc.blend, 'opaque');
    assert.ok(engine.functions.has(call.desc.function));
  }
});

test('load accepts the asset as JSON text', async () => {
  const { loader } = setup();
  const asset = await loader.load(fixtureText);
  assert.equal(asset.parts.length, 2);
});

test('cache by hash: a second load of the same asset meshes and uploads nothing more', async () => {
  const { engine, loader, mamaCalls } = setup();
  const before = meshCalls();
  const [a, b] = await Promise.all([loader.load(fixture()), loader.load(fixture())]);
  const c = await loader.load(fixtureText);
  assert.equal(meshCalls() - before, 1);
  assert.equal(mamaCalls(), 1);
  assert.equal(engine.calls.addMesh.length, 2, 'addMesh once per part');
  assert.equal(engine.calls.addSurfaceMaterial.length, 2);
  assert.equal(a, b);
  assert.equal(a, c);
});

test('rigid parts: spawn creates a root and one node and one instance per part', async () => {
  const { engine, loader } = setup();
  const asset = await loader.load(fixture());
  const parent = engine.addNode(null);
  const spawned = loader.spawn(asset, parent);
  assert.equal(engine.nodes.get(spawned.root).parent, parent);
  assert.deepEqual(Object.keys(spawned.parts).sort(), ['Body', 'Wheel']);
  for (const part of asset.parts) {
    const { node, instance } = spawned.parts[part.name];
    const n = engine.nodes.get(node);
    assert.equal(n.parent, spawned.root, 'flat hierarchy: MAMA parts hang under the root');
    assert.deepEqual(n.t, part.t);
    assert.deepEqual(n.r, part.r);
    assert.deepEqual(engine.instances.get(instance), { node, mesh: part.mesh });
  }
  assert.equal(engine.calls.addMeshInstance.length, 2);
  assert.equal(engine.nodes.get(loader.spawn(asset).root).parent, null, 'default parent is null');
});

test('rigid parts: a part with a parent part gets that part node as parent, parents first', async () => {
  const { engine, loader } = setup();
  const loaded = await loader.load(fixture());
  const [body, wheel] = loaded.parts;
  // The wheel listed before its parent: spawn must create Body's node first.
  const asset = { ...loaded, parts: [{ ...wheel, parent: 'Body' }, body] };
  const nodesBefore = engine.calls.addNode.length;
  const spawned = loader.spawn(asset);
  assert.equal(engine.calls.addNode.length - nodesBefore, 3, 'root + one node per part');
  assert.equal(engine.nodes.get(spawned.parts.Body.node).parent, spawned.root);
  assert.equal(engine.nodes.get(spawned.parts.Wheel.node).parent, spawned.parts.Body.node);
  assert.ok(spawned.parts.Body.node < spawned.parts.Wheel.node);
  assert.equal(engine.calls.addMeshInstance.length, 2, 'one instance per part');
});

test('progress reports validate, mesh and upload with done <= total', async () => {
  const { loader } = setup();
  const seen = [];
  await loader.load(fixture(), { onProgress: (p) => seen.push(p) });
  const stages = [...new Set(seen.map((p) => p.stage))];
  assert.deepEqual(stages, ['validate', 'mesh', 'upload']);
  for (const p of seen) {
    assert.ok(Number.isInteger(p.done) && Number.isInteger(p.total) && p.total > 0, JSON.stringify(p));
    assert.ok(p.done <= p.total, JSON.stringify(p));
  }
  for (const stage of stages) {
    const last = seen.filter((p) => p.stage === stage).at(-1);
    assert.equal(last.done, last.total, `${stage} completes`);
  }
});

test('an invalid asset is refused with its S15 code and path, without calling MAMA', async () => {
  const { engine, loader, mamaCalls } = setup();
  const cases = [
    [(a) => { a.format = 'other'; }, 'sdfm.format'],
    [(a) => { a.version = 2; }, 'sdfm.version'],
    [(a) => { delete a.bounds; }, 'sdfm.schema'],
    [(a) => { a.id = 'Bad Id'; }, 'sdfm.id'],
    [(a) => { a.anchors.hitch.t[0] = Number.NaN; }, 'sdfm.not_finite'],
    [(a) => { a.document.pad = 'x'.repeat(2 * 1024 * 1024); }, 'sdfm.cap_document'],
  ];
  for (const [edit, c] of cases) {
    const asset = fixture();
    edit(asset);
    await assert.rejects(loader.load(asset), code(c));
  }
  await assert.rejects(loader.load('{ not json'), code('sdfm.schema'));
  await assert.rejects(loader.load(fixture(), { onProgress: () => {} }).then(() => loader.load({ ...fixture(), game: 1 })),
    (e) => code('sdfm.schema')(e) && e.path === '/game');
  assert.equal(mamaCalls(), 1, 'only the valid load reached MAMA');
  assert.equal(engine.calls.addMesh.length, 2);
});

test('a tampered document is sdfm.hash_mismatch before meshing or upload', async () => {
  const { loader, mamaCalls } = setup();
  const asset = fixture();
  asset.document.nodes.cart.name = 'Tampered';
  await assert.rejects(loader.load(asset), code('sdfm.hash_mismatch'));
  assert.equal(mamaCalls(), 1, 'MAMA supplies the serializer');
});

test('a MAMA group refusal surfaces as sdfm.mesher_refused naming the group', async () => {
  const { engine, loader } = setup();
  const asset = await variant('raatest.refused', (d) => {
    d.nodes.cart.fake = { refuse: { message: 'the mesher produced no triangle' } };
  });
  await assert.rejects(loader.load(asset), (e) => {
    code('sdfm.mesher_refused')(e);
    assert.equal(e.path, '/document');
    assert.match(e.message, /cart: group 'Cart' cannot be meshed/);
    assert.match(e.message, /the mesher produced no triangle/);
    return true;
  });
  assert.equal(engine.calls.addMesh.length, 0);
});

test('a refused load is not cached: the next load of the same hash meshes again', async () => {
  const { loader } = setup();
  const asset = await variant('raatest.throws', (d) => {
    d.nodes.cart.fake = { throw: 'mesher crashed' };
  });
  const before = meshCalls();
  await assert.rejects(loader.load(asset), code('sdfm.worker'));
  await assert.rejects(loader.load(asset), code('sdfm.worker'));
  assert.equal(meshCalls() - before, 2);
});

test('an unmappable compiled graph surfaces as sdfm.material_unmappable', async () => {
  const { engine, loader } = setup();
  const asset = await variant('raatest.unmappable', (d) => {
    d.materials.paint.fake.wgslBody = '  return MaterialFull(raa_secret(), 0.0, 0.4, vec3f(0.0), 0.0);\n';
  });
  await assert.rejects(loader.load(asset), (e) => code('sdfm.material_unmappable')(e) && /raa_/.test(e.message));
  assert.equal(engine.calls.addMesh.length, 0);
});

test('a mama() failure surfaces as sdfm.worker and an engine refusal propagates unchanged', async () => {
  const failing = createSdfmLoader({ engine: createFakeEngine(), mama: async () => { throw new Error('no MAMA'); } });
  await assert.rejects(failing.load(fixture()), (e) => code('sdfm.worker')(e) && /no MAMA/.test(e.message));

  const engine = createFakeEngine();
  engine.addMesh = () => { throw new FakeRaaError('mesh.invalid_normal', 'normals[0] has zero length'); };
  const loader = createSdfmLoader({ engine, mama: async () => fakeMama });
  await assert.rejects(loader.load(fixture()), (e) => e instanceof FakeRaaError && e.code === 'mesh.invalid_normal');
});

test('after dispose load rejects and spawn throws sdfm.disposed', async () => {
  const { loader } = setup();
  const asset = await loader.load(fixture());
  loader.dispose();
  loader.dispose();
  await assert.rejects(loader.load(fixture()), code('sdfm.disposed'));
  assert.throws(() => loader.spawn(asset), code('sdfm.disposed'));
  assert.equal(loader.validate(fixture()).ok, true, 'validate needs no live loader');
});

test('a load in flight when dispose is called rejects sdfm.disposed', async () => {
  const { engine, loader } = setup();
  const pending = loader.load(fixture());
  loader.dispose();
  await assert.rejects(pending, code('sdfm.disposed'));
  assert.equal(engine.calls.addMesh.length, 0);
});

test('addMaterialFunction is called once per graph hash across two assets sharing a material', async () => {
  const { engine, loader } = setup();
  const other = await variant('raatest.trolley', (d) => {
    d.nodes.cart.name = 'Trolley';
    d.nodes.cart.fake.parts = [{ partId: 'deck', name: 'Deck', frame: d.nodes.cart.fake.parts[0].frame,
      subMeshes: [{ material: 'metal', vertices: 3 }] }];
  });
  const a = await loader.load(fixture());
  const b = await loader.load(other);
  assert.notEqual(a.hash, b.hash);
  const names = engine.calls.addMaterialFunction.map((c) => c.name);
  assert.equal(names.length, 2, `paint and metal once each: ${names}`);
  assert.equal(new Set(names).size, 2);
  assert.ok(names.every((n) => /^sdfm\.[0-9a-f]{16}$/.test(n)));
  assert.equal(engine.calls.addSurfaceMaterial.length, 3);
  assert.equal(engine.materials.get(b.parts[0].materials[0]).function, engine.materials.get(a.parts[1].materials[0]).function);
});

test('a function the engine already has (material.fn_duplicate) counts as registered', async () => {
  const engine = createFakeEngine();
  const first = createSdfmLoader({ engine, mama: async () => fakeMama });
  await first.load(fixture());
  const second = createSdfmLoader({ engine, mama: async () => fakeMama });
  const asset = await second.load(fixture());
  assert.equal(engine.calls.addMaterialFunction.length, 2);
  assert.equal(asset.parts.length, 2);
});


test('polygon edges follow material grouping and vertex offsets; absent and empty stay distinct', async () => {
  const { engine, loader } = setup();
  const input = await variant('test.edges', (doc) => {
    const subs = doc.nodes.cart.fake.parts[0].subMeshes;
    subs[0].edges = [0, 1, 1, 2];
    subs[1].edges = [];
    subs[2].edges = [0, 2];
  });
  const asset = await loader.load(input);
  const desc = engine.meshes.get(asset.parts[0].mesh);
  assert.deepEqual(desc.edges, new Uint32Array([0, 1, 1, 2, 6, 8]));
  assert.equal(engine.meshes.get(asset.parts[1].mesh).edges, undefined);
});

test('mixed polygon and legacy submeshes retain legacy triangle edges', async () => {
  const { engine, loader } = setup();
  const input = await variant('test.mixed-edges', (doc) => {
    doc.nodes.cart.fake.parts[0].subMeshes[0].edges = [];
  });
  const asset = await loader.load(input);
  assert.deepEqual(engine.meshes.get(asset.parts[0].mesh).edges,
    new Uint32Array([6, 7, 7, 8, 8, 6, 9, 10, 10, 11, 11, 9]));
});


test('loader hashes serializer bytes rather than adapter pretty JSON', async () => {
  const { sha256Hex, canonicalJson } = await import('../src/canonical.js');
  const { loader } = setup();
  const input = fixture();
  assert.notEqual(input.hash, await sha256Hex(canonicalJson(input.document)));
  await loader.load(input);
  input.hash = await sha256Hex(canonicalJson(input.document));
  await assert.rejects(loader.load(input), (e) =>
    code('sdfm.hash_mismatch')(e) && e.path === '/hash' && /sha256/.test(e.message));
});
