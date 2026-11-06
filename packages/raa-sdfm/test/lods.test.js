import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { LOD_LEVELS_DEFAULT, lodDocument, attachLods, createSdfmLoader } from '../index.js';
import { documentHash } from '../src/canonical.js';
import { createFakeEngine } from './fake-engine.js';
import * as fakeMama from './fake-mama.js';

const fixture = JSON.parse(readFileSync(new URL('./fixtures/rigid.asset.json', import.meta.url), 'utf8'));

async function assetFor(edit = () => {}) {
  const asset = structuredClone(fixture);
  asset.document.settings = { paramesh: { subdivisions: 16, fieldResolution: 96, normals: 'geometry' } };
  edit(asset.document);
  asset.hash = await documentHash(asset.document, fakeMama.serializeDocument);
  return asset;
}

function lodEngine() {
  const engine = createFakeEngine();
  Object.assign(engine.calls, { meshLodError: [], setMeshLods: [], removeMesh: [] });
  engine.meshLodError = (full, level) => {
    const error = (engine.calls.meshLodError.length + 1) / 10;
    engine.calls.meshLodError.push({ full, level, error });
    return error;
  };
  engine.setMeshLods = (mesh, levels) => engine.calls.setMeshLods.push({ mesh, levels });
  engine.removeMesh = (mesh) => {
    engine.calls.removeMesh.push(mesh);
    engine.meshes.delete(mesh);
  };
  return engine;
}

function setup({ engine = lodEngine(), lods, mutate = () => {} } = {}) {
  const documents = [];
  const mama = {
    ...fakeMama,
    async meshAssets(doc, options) {
      documents.push(structuredClone(doc));
      const results = await fakeMama.meshAssets(doc, options);
      mutate(results, documents.length - 1);
      return results;
    },
  };
  const loader = createSdfmLoader({ engine, mama: async () => mama, ...(lods === undefined ? {} : { lods }) });
  return { engine, loader, documents };
}

test('lodDocument copies the document, halves quality per level and clamps at 1 and 16', () => {
  const document = { settings: { paramesh: { subdivisions: 16, fieldResolution: 96, precision: 'f16' } }, nodes: { root: {} } };
  const before = structuredClone(document);
  const first = lodDocument(document, 1);
  const second = lodDocument(document, 2);
  assert.deepEqual([first.settings.paramesh.subdivisions, first.settings.paramesh.fieldResolution], [8, 48]);
  assert.deepEqual([second.settings.paramesh.subdivisions, second.settings.paramesh.fieldResolution], [4, 24]);
  assert.deepEqual(lodDocument(first, 1), second);
  assert.deepEqual(lodDocument(document, 10).settings.paramesh, { subdivisions: 1, fieldResolution: 16, precision: 'f16' });
  first.nodes.root.changed = true;
  assert.deepEqual(document, before);
  assert.deepEqual(lodDocument(document, 0), document);
  assert.deepEqual(lodDocument({}, 1).settings.paramesh, { subdivisions: 8, fieldResolution: 48 });
  assert.deepEqual(lodDocument({ settings: { paramesh: { subdivisions: 3, fieldResolution: 33 } } }, 1)
    .settings.paramesh, { subdivisions: 1, fieldResolution: 16 });
});

test('attachLods returns Rust errors and registers all handle/error pairs once', () => {
  const engine = lodEngine();
  const full = { positions: new Float32Array(9) };
  const levels = [{ positions: new Float32Array(6) }, { positions: new Float32Array(3) }];
  assert.deepEqual(attachLods(engine, 10, full, [11, 12], levels), [0.1, 0.2]);
  assert.deepEqual(engine.calls.meshLodError.map(({ full: f, level }) => [f, level]), [[full, levels[0]], [full, levels[1]]]);
  assert.equal(engine.calls.meshLodError[0].full, full);
  assert.equal(engine.calls.meshLodError[1].level, levels[1]);
  assert.deepEqual(engine.calls.setMeshLods, [{ mesh: 10, levels: [{ mesh: 11, error: 0.1 }, { mesh: 12, error: 0.2 }] }]);
});

test('attachLods without setMeshLods returns [] without measuring', () => {
  const engine = lodEngine();
  delete engine.setMeshLods;
  assert.deepEqual(attachLods(engine, 10, {}, [11], [{}]), []);
  assert.deepEqual(engine.calls.meshLodError, []);
});

test('lods 2 meshes three documents per asset and attaches two levels per part with shared materials', async () => {
  const { engine, loader, documents } = setup({ lods: 2, mutate(results, k) {
    if (k > 0) {
      results[0].parts.reverse();
      for (const part of results[0].parts) part.name = 'Renamed';
    }
  } });
  const input = await assetFor();
  const before = structuredClone(input);
  const progress = [];
  const loaded = await loader.load(input, { onProgress: (p) => progress.push(p) });
  assert.equal(documents.length, 3);
  assert.deepEqual(documents.map((d) => d.settings.paramesh.subdivisions), [16, 8, 4]);
  assert.deepEqual(documents.map((d) => d.settings.paramesh.fieldResolution), [96, 48, 24]);
  assert.deepEqual(input, before);
  assert.equal(engine.calls.setMeshLods.length, 2);
  assert.equal(engine.calls.addMesh.length, 6);
  assert.equal(engine.calls.addSurfaceMaterial.length, 2);
  for (const part of loaded.parts) {
    assert.equal(part.lods.length, 2);
    const chain = engine.calls.setMeshLods.find((call) => call.mesh === part.mesh);
    assert.deepEqual(chain.levels.map((l) => l.mesh), part.lods);
    for (const level of part.lods) {
      assert.deepEqual(engine.meshes.get(level).subMeshes, engine.meshes.get(part.mesh).subMeshes);
    }
  }
  assert.deepEqual(engine.calls.removeMesh, []);
  const lodProgress = progress.filter((p) => p.stage === 'lod');
  assert.equal(lodProgress[0].done, 0);
  assert.equal(lodProgress.at(-1).done, lodProgress.at(-1).total);
  assert.ok(progress.every((p) => p.done >= 0 && p.done <= p.total));
  assert.equal(await loader.load(input), loaded);
  assert.equal(documents.length, 3, 'cached loads make no new levels');
  await loader.load(await assetFor((doc) => { doc.name = 'Other'; }));
  assert.equal(documents.length, 6);
  assert.equal(engine.calls.setMeshLods.length, 4);
});

test('default level count is two; disabled or unsupported LODs mesh only the full document', async () => {
  assert.equal(LOD_LEVELS_DEFAULT, 2);
  const defaults = setup();
  await defaults.loader.load(await assetFor());
  assert.equal(defaults.documents.length, 3);
  for (const options of [{ lods: 0 }, { engine: createFakeEngine() }]) {
    const { loader, documents, engine } = setup(options);
    const loaded = await loader.load(await assetFor());
    assert.equal(documents.length, 1);
    assert.equal(engine.calls.addMesh.length, 2);
    assert.ok(loaded.parts.every((part) => part.lods.length === 0));
  }
});

test('a failed level resolves the load with empty LODs and removes prior level uploads', async () => {
  const { loader, engine, documents } = setup({ mutate(_results, k) {
    if (k === 2) throw new Error('level mesher failed');
  } });
  const loaded = await loader.load(await assetFor());
  assert.equal(documents.length, 3);
  assert.ok(loaded.parts.every((part) => part.lods.length === 0));
  assert.deepEqual(engine.calls.setMeshLods, []);
  assert.deepEqual(engine.calls.removeMesh, engine.calls.addMesh.slice(2).map(({ id }) => id));
  assert.deepEqual([...engine.meshes.keys()], loaded.parts.map((part) => part.mesh));
  assert.equal(Object.keys(loader.spawn(loaded).parts).length, 2);
});

test('a missing part at one level discards only that part chain and removes its unused mesh', async () => {
  const { loader, engine } = setup({ mutate(results, k) {
    if (k === 2) results[0].parts = results[0].parts.filter((part) => part.partId !== 'body');
  } });
  const loaded = await loader.load(await assetFor());
  assert.deepEqual(loaded.parts[0].lods, []);
  assert.equal(loaded.parts[1].lods.length, 2);
  assert.equal(engine.calls.setMeshLods.length, 1);
  assert.equal(engine.calls.removeMesh.length, 1);
  assert.equal(engine.meshes.size, 4);
});

test('LOD upload, measurement and registration failures resolve and clean every unattached mesh', async () => {
  for (const failure of ['addMesh', 'meshLodError', 'setMeshLods']) {
    const engine = lodEngine();
    const original = engine[failure];
    engine[failure] = (...args) => {
      if (failure !== 'addMesh' || engine.calls.addMesh.length >= 3) throw new Error(`${failure} failed`);
      return original(...args);
    };
    const { loader } = setup({ engine });
    const loaded = await loader.load(await assetFor());
    assert.ok(loaded.parts.every((part) => part.lods.length === 0), failure);
    assert.deepEqual(engine.calls.removeMesh, engine.calls.addMesh.slice(2).map(({ id }) => id), failure);
    assert.equal(engine.meshes.size, 2, failure);
  }
});

test('an engine missing meshLodError leaves no uploaded LOD meshes', async () => {
  const engine = lodEngine();
  delete engine.meshLodError;
  const { loader } = setup({ engine });
  const loaded = await loader.load(await assetFor());
  assert.ok(loaded.parts.every((part) => part.lods.length === 0));
  assert.equal(engine.meshes.size, 2);
  assert.deepEqual(engine.calls.setMeshLods, []);
});

test('skinned parts keep no LOD while unskinned parts of the same asset attach levels', async () => {
  const input = await assetFor((doc) => {
    const group = doc.nodes.cart;
    group.rig = { format: 'raa.rig', version: 1, name: 'cart', bones: [] };
    for (const mesh of group.fake.parts[0].subMeshes) {
      const count = mesh.vertices;
      mesh.skin = { joints: Array(count * 4).fill(0), weights: Array.from({ length: count * 4 }, (_, i) => i % 4 === 0 ? 1 : 0) };
    }
  });
  const { loader, engine } = setup();
  const loaded = await loader.load(input);
  assert.ok(loaded.parts[0].skin);
  assert.deepEqual(loaded.parts[0].lods, []);
  assert.equal(loaded.parts[1].lods.length, 2);
  assert.deepEqual(engine.calls.setMeshLods.map((c) => c.mesh), [loaded.parts[1].mesh]);
  assert.equal(engine.calls.addMesh.length, 4);
});

test('matching includes the group when parts share a partId', async () => {
  const input = await assetFor((doc) => {
    doc.nodes.other = structuredClone(doc.nodes.cart);
    doc.nodes.other.fake.parts = [doc.nodes.other.fake.parts[1]];
    doc.nodes.other.fake.parts[0].partId = 'body';
  });
  const { loader, engine } = setup({ mutate(results, k) {
    if (k > 0) results.reverse();
  } });
  const loaded = await loader.load(input);
  assert.equal(loaded.parts.length, 3);
  for (const part of loaded.parts) {
    assert.equal(part.lods.length, 2);
    for (const handle of part.lods) {
      assert.deepEqual(engine.meshes.get(handle).subMeshes, engine.meshes.get(part.mesh).subMeshes);
    }
  }
});

test('lods outside integer 0..3 throws TypeError naming options.lods', () => {
  for (const lods of [4, -1, 1.5, '2', null, NaN, Infinity, true]) {
    assert.throws(() => createSdfmLoader({ engine: lodEngine(), mama: async () => fakeMama, lods }),
      (error) => error instanceof TypeError && error.message.includes('options.lods'));
  }
});
