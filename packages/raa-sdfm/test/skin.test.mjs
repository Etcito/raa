// S18: MAMA skin arrays, the group rig and clips survive the adapter path unchanged.
// Fixture conventions follow MAMA's two-group flipper (src/core/game/skin.test.ts at M5): bones in
// authored order root, flipperL, flipperR, tipL; the rigged group's Left part uses flipperL and
// tipL, its Right part falls back to root, and the asset's second (Plain) group stays unrigged.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fakeMama from './fake-mama.js';
import { createFakeEngine } from './fake-engine.js';
import { createSdfmLoader, SdfmError } from '../index.js';
import { meshDocument } from '../src/mama-bridge.js';
import { documentHash } from '../src/canonical.js';
import { createWorkerHandler } from '../worker.js';
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const bone = (name, parent) => ({ name, parent, rest: { t: [0, 0, 0], r: [0, 0, 0, 1], s: [1, 1, 1] } });
const material = (name, rgb) => ({
  name, fake: { wgslBody: `  return MaterialFull(vec3f(${rgb}), 0.0, 0.5, vec3f(0.0), 0.0);\n`, helpers: [], constants: [] },
});
// Four influences per vertex; the left part's vertices belong to flipperL/tipL, the fallback ones to root.
const RED_A = { joints: [1, 3, 0, 0, 1, 3, 0, 0, 3, 1, 0, 0], weights: [0.75, 0.25, 0, 0, 0.5, 0.5, 0, 0, 1, 0, 0, 0] };
const BLUE_B = { joints: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], weights: [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0] };
const RED_C = { joints: [3, 1, 0, 0, 1, 3, 0, 0, 1, 1, 1, 0], weights: [0.8, 0.2, 0, 0, 0.6, 0.4, 0, 0, 0.5, 0.3, 0.2, 0] };
const rig = () => ({
  format: 'raa.rig', version: 1, name: 'flipper-test',
  bones: [bone('root', null), bone('flipperL', 'root'), bone('flipperR', 'root'), bone('tipL', 'flipperL')],
  roles: { root: 'root', 'upperArm.l': 'flipperL', 'lowerArm.l': 'tipL' },
  bindings: [
    { source: { group: 'left' }, bones: [{ bone: 'flipperL', factor: 1 }] },
    { source: { shape: 'tipL' }, bones: [{ bone: 'tipL', factor: 1 }] },
  ],
  smooth: 0,
});
const clip = (name = 'wave') => ({
  format: 'raa.clip', version: 2, name, rig: 'flipper-test', lengthTicks: 8, loop: true,
  rootMotion: 'none',
  tracks: [{ bone: 'flipperL', channel: 'rotation', keys: [
    { tick: 0, v: [0, 0, 0, 1], interp: 'LINEAR' }, { tick: 8, v: [0, 0, 0.2, 1], interp: 'LINEAR' },
  ] }],
  events: [{ tick: 4, name: 'wave' }],
});
/** One rigged asset group with its two authored groups and one unrigged group, meshed by the fake MAMA. */
function flipperDocument() {
  return {
    nodes: {
      flipper: {
        kind: 'group', name: 'Flipper', asset: { game: 'fixture-game', id: 'prop.flipper', category: 'prop' },
        rig: rig(), clips: [clip(), clip('dance')], children: ['left', 'right'],
        fake: { parts: [
          { partId: 'left', name: 'Left', frame: IDENTITY, subMeshes: [
            { material: 'red', vertices: 3, skin: RED_A },
            { material: 'blue', vertices: 3, skin: BLUE_B },
            { material: 'red', vertices: 3, skin: RED_C },
          ] },
          { partId: 'right', name: 'Right', frame: IDENTITY, subMeshes: [{ material: 'blue', vertices: 3, skin: BLUE_B }] },
        ] },
      },
      left: { kind: 'group', name: 'Left', children: ['tipL', 'paddleL'] },
      right: { kind: 'group', name: 'Right', children: ['tipR'] },
      tipL: { kind: 'box', name: 'Tip L', material: 'red' },
      paddleL: { kind: 'box', name: 'Paddle L', material: 'red' },
      tipR: { kind: 'box', name: 'Tip R', material: 'blue' },
      plain: {
        kind: 'group', name: 'Plain', asset: { game: 'fixture-game', id: 'prop.plain', category: 'prop' },
        fake: { parts: [{ partId: '__root__', name: 'Plain', frame: IDENTITY, subMeshes: [{ material: 'blue', vertices: 3 }] }] },
      },
    },
    materials: { red: material('Red', '1.0, 0.0, 0.0'), blue: material('Blue', '0.0, 0.0, 1.0') },
  };
}
async function assetFor(document) {
  return {
    format: 'sdfm.game-asset', version: 1, game: 'fixture-game', id: 'raatest.flipper',
    name: 'Flipper', category: 'prop', world: 'test', mesher: { output: 'quad' }, document,
    collision: [], anchors: {}, bounds: { min: [-1, -1, -1], max: [1, 1, 1] },
    hash: await documentHash(document, fakeMama.serializeDocument),
  };
}
async function flipperAsset(edit = () => {}) {
  const document = flipperDocument();
  edit(document);
  return assetFor(document);
}
const runtime = () => {
  const engine = createFakeEngine();
  const loader = createSdfmLoader({ engine, mama: async () => fakeMama });
  return { engine, loader };
};
const setup = async (edit = () => {}) => {
  const document = flipperDocument();
  edit(document);
  const { engine, loader } = runtime();
  return { engine, loader, document, asset: await loader.load(await assetFor(document)) };
};
test('the bridge copies skin joint order, typed arrays and the group rig association', async () => {
  const document = flipperDocument();
  const out = await meshDocument(fakeMama, document, { pool: fakeMama.fakePool() });
  const flipper = out.groups.find((g) => g.group === 'flipper');
  assert.deepEqual(flipper.clips, document.nodes.flipper.clips);
  assert.deepEqual(flipper.rig, document.nodes.flipper.rig);
  const meshes = flipper.parts.flatMap((p) => p.meshes);
  const red = meshes.filter((m) => m.material === 'red');
  assert.equal(red.length, 2);
  assert.ok(red[0].skin.joints instanceof Uint16Array && red[0].skin.weights instanceof Float32Array);
  assert.deepEqual(Array.from(red[0].skin.joints), RED_A.joints);
  assert.deepEqual(Array.from(red[1].skin.weights), Array.from(Float32Array.from(RED_C.weights)));
  for (const mesh of meshes) assert.equal(mesh.skin.rig, flipper.rig, 'every skin keeps the group rig');
  const plain = out.groups.find((g) => g.group === 'plain');
  assert.equal(plain.rig, undefined);
  assert.ok(plain.parts.every((p) => p.meshes.every((m) => m.skin === undefined)), 'unrigged meshes stay S13 meshes');
});
test('load uploads joints and weights in material order and exposes the asset rig and clips', async () => {
  const { engine, document, asset } = await setup();
  assert.deepEqual(asset.rig, document.nodes.flipper.rig);
  assert.deepEqual(asset.rig.bones.map((b) => b.name), ['root', 'flipperL', 'flipperR', 'tipL']);
  assert.deepEqual(asset.clips, [clip(), clip('dance')]);
  assert.deepEqual(asset.clips.map((c) => c.name), ['wave', 'dance'], 'clips keep their authored order');
  assert.deepEqual(asset.clips, document.nodes.flipper.clips, 'clip values survive the MAMA bridge and the loader unchanged');
  const left = asset.parts.find((p) => p.name === 'Left');
  const desc = engine.meshes.get(left.mesh);
  assert.ok(desc.joints instanceof Uint16Array && desc.weights instanceof Float32Array);
  assert.equal(desc.joints.length, desc.positions.length / 3 * 4);
  assert.equal(desc.weights.length, desc.joints.length);
  // Material grouping puts the second red sub-mesh before the blue one: skin follows the vertices.
  assert.deepEqual(Array.from(desc.joints), [...RED_A.joints, ...RED_C.joints, ...BLUE_B.joints]);
  assert.deepEqual(Array.from(desc.weights), Array.from(Float32Array.from([...RED_A.weights, ...RED_C.weights, ...BLUE_B.weights])));
  assert.deepEqual(desc.subMeshes.map((s) => s.count), [6, 3]);
  assert.equal(left.skin.joints, desc.joints);
  assert.equal(left.skin.weights, desc.weights);
  assert.equal(left.skin.rig, asset.rig);
  assert.ok(Array.from(desc.joints).every((joint) => joint < asset.rig.bones.length));
  const right = asset.parts.find((p) => p.name === 'Right');
  assert.deepEqual([...new Set(Array.from(engine.meshes.get(right.mesh).joints))], [0], 'unbound geometry keeps the root bone');
  assert.equal(right.skin.rig, asset.rig);
  const plain = asset.parts.find((p) => p.name === 'Plain');
  assert.equal(engine.meshes.get(plain.mesh).joints, undefined);
  assert.equal(plain.skin, undefined);
  assert.equal(engine.calls.addSurfaceMaterial.length, 2, 'materials still upload once per graph');
});
test('an invalid rig surfaces as sdfm.rig with the S16 code and no mesh upload', async () => {
  const engine = createFakeEngine();
  const loader = createSdfmLoader({ engine, mama: async () => fakeMama });
  const asset = await flipperAsset((doc) => {
    doc.nodes.flipper.fake.rigError = 'rig.parent_order: bone 1 is listed before its parent';
  });
  await assert.rejects(loader.load(asset), (e) => {
    assert.ok(e instanceof SdfmError);
    assert.equal(e.code, 'sdfm.rig');
    assert.equal(e.path, '/document');
    assert.match(e.message, /flipper: rig\.parent_order/);
    return true;
  });
  assert.equal(engine.calls.addMesh.length, 0);
});
test('the worker reply carries and transfers the skin arrays', async () => {
  const posted = [];
  const handle = createWorkerHandler({
    post: (message, transfer) => posted.push({ message, transfer }),
    loadMama: async () => fakeMama,
  });
  await handle({ type: 'init', mamaUrl: undefined });
  await handle({ type: 'mesh', id: 7, document: flipperDocument() });
  const [{ message, transfer }] = posted;
  assert.equal(message.type, 'meshed');
  const flipper = message.groups.find((g) => g.group === 'flipper');
  assert.ok(flipper.rig && flipper.clips.length === 2);
  assert.deepEqual(flipper.rig.bones.map((b) => b.name), ['root', 'flipperL', 'flipperR', 'tipL']);
  assert.deepEqual(flipper.clips.map((c) => c.name), ['wave', 'dance']);
  const meshes = flipper.parts.flatMap((p) => p.meshes);
  assert.ok(meshes.every((m) => m.skin && transfer.includes(m.skin.joints.buffer)
    && transfer.includes(m.skin.weights.buffer)));
});
test('a rigged two-group asset leaves its unrigged group at the S15 result', async () => {
  const rigged = await setup();
  const unrigged = await setup((doc) => {
    delete doc.nodes.flipper.rig;
    delete doc.nodes.flipper.clips;
    // Without a rig MAMA exports no skin at all (S18): drop the fixture's skin rows too.
    for (const part of doc.nodes.flipper.fake.parts) for (const sub of part.subMeshes) delete sub.skin;
  });
  assert.deepEqual(Object.keys(unrigged.asset).sort(),
    ['anchors', 'bounds', 'category', 'collision', 'hash', 'id', 'name', 'parts', 'world']);
  assert.equal(unrigged.asset.rig, undefined);
  assert.equal(unrigged.asset.clips, undefined);
  const geometry = (desc) => ({
    positions: desc.positions, normals: desc.normals, indices: desc.indices,
    subMeshes: desc.subMeshes, uvs: desc.uvs, edges: desc.edges,
  });
  assert.equal(unrigged.asset.parts.length, rigged.asset.parts.length);
  for (const [i, part] of unrigged.asset.parts.entries()) {
    const other = rigged.asset.parts[i];
    assert.deepEqual([part.name, part.groupId, part.rigid, part.parent, part.t, part.r],
      [other.name, other.groupId, other.rigid, other.parent, other.t, other.r]);
    assert.equal(part.skin, undefined);
    const desc = unrigged.engine.meshes.get(part.mesh);
    assert.equal(desc.joints, undefined);
    assert.equal(desc.weights, undefined);
    assert.deepEqual(geometry(desc), geometry(rigged.engine.meshes.get(other.mesh)), 'only skin is additive');
  }
  assert.equal(unrigged.engine.calls.addSurfaceMaterial.length, 2, 'materials upload as before');
  const spawned = unrigged.loader.spawn(unrigged.asset);
  assert.equal(Object.keys(spawned.parts).length, unrigged.asset.parts.length);
});
test('clips of a group without a rig are retained beside the rigged group', async () => {
  const { document, asset } = await setup((doc) => {
    delete doc.nodes.flipper.clips;
    doc.nodes.plain.clips = [clip('idle')];
  });
  assert.deepEqual(asset.clips.map((c) => c.name), ['idle']);
  assert.deepEqual(asset.rig, document.nodes.flipper.rig, 'the rig still comes from the rigged group');
  assert.equal(asset.parts.find((p) => p.name === 'Plain').skin, undefined);
  assert.equal(asset.parts.find((p) => p.name === 'Left').skin.rig, asset.rig);
});
const S16_REFUSALS = ['rig.format', 'rig.version', 'rig.schema', 'rig.not_finite', 'rig.bone_name',
  'rig.bone_duplicate', 'rig.parent_order', 'rig.root', 'rig.cap_bones', 'rig.role_unknown', 'rig.chain',
  'rig.limit', 'rig.attach'];
test('every invalid rig surfaces as sdfm.rig with its S16 code and no upload', async () => {
  for (const code of S16_REFUSALS) {
    const asset = await flipperAsset((doc) => {
      doc.nodes.flipper.fake.rigError = `${code}: refused by M5`;
    });
    const { engine, loader } = runtime();
    await assert.rejects(loader.load(asset), (e) => {
      assert.ok(e instanceof SdfmError, code);
      assert.equal(e.code, 'sdfm.rig', code);
      assert.equal(e.path, '/document', code);
      assert.ok(e.message.includes(`flipper: ${code}`), `${code}: ${e.message}`);
      return true;
    }, code);
    assert.equal(engine.calls.addMesh.length, 0, code);
    assert.equal(engine.calls.addMaterialFunction.length, 0, code);
    assert.equal(engine.calls.addSurfaceMaterial.length, 0, code);
  }
});
test('an invalid rig in a later group refuses the asset and names that group', async () => {
  const asset = await flipperAsset((doc) => {
    doc.nodes.plain.fake = { parts: [], rigError: 'rig.root: no root bone' };
  });
  const { engine, loader } = runtime();
  await assert.rejects(loader.load(asset), (e) => {
    assert.equal(e.code, 'sdfm.rig');
    assert.ok(e.message.includes('plain: rig.root'), e.message);
    return true;
  });
  assert.equal(engine.calls.addMesh.length, 0, 'no group uploads when any group is refused');
});
test('the worker reports an S16 rig refusal as sdfm.rig', async () => {
  const posted = [];
  const handle = createWorkerHandler({
    post: (message, transfer) => posted.push({ message, transfer }),
    loadMama: async () => fakeMama,
  });
  await handle({ type: 'init', mamaUrl: undefined });
  const document = flipperDocument();
  document.nodes.flipper.fake.rigError = 'rig.parent_order: bone 3 is listed before its parent';
  await handle({ type: 'mesh', id: 8, document });
  const [{ message }] = posted;
  assert.equal(message.type, 'error');
  assert.equal(message.id, 8);
  assert.equal(message.code, 'sdfm.rig');
  assert.equal(message.path, '/document');
  assert.ok(message.message.includes('flipper: rig.parent_order'), message.message);
});
