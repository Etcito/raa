import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createMamaLightOwner } from '../src/mama-light-owner.js';
import { SdfmError } from '../src/errors.js';

const light = (id, fields = {}) => ({
  id, node: 'node', key: 7, t: [1, 2, 3], r: [0, 0, 0, 1], s: [1, 1, 1],
  kind: 'point', color: [1, 1, 1], intensity: 3, range: 8, shadows: false, ...fields,
});
const refusal = Object.assign(new Error('engine refused light'), { code: 'light.invalid_desc' });
const rejectsWith = (code) => (error) => error instanceof SdfmError && error.code === code;

// Local S17 double: updates merge atomically, and explicit undefined restores defaults.
function engine() {
  const defaults = {
    kind: 'point', position: [0, 0, 0], direction: [0, 0, -1], color: [1, 1, 1],
    intensity: 2, range: 6, angleDeg: 35, innerAngleDeg: undefined,
    size: [1, 1], radius: 0, shadows: false,
  };
  const merge = (base, changes) => {
    const desc = structuredClone(base);
    for (const [key, value] of Object.entries(changes)) {
      desc[key] = structuredClone(value === undefined ? defaults[key] : value);
    }
    if (desc.intensity < 0 || desc.range <= 0 || desc.innerAngleDeg > desc.angleDeg) throw refusal;
    return desc;
  };
  let next = 1;
  return {
    lights: new Map(), calls: [], failAddAt: 0, failUpdate: () => false,
    addLight(desc) {
      this.calls.push(['add', structuredClone(desc)]);
      if (this.calls.filter(([op]) => op === 'add').length === this.failAddAt) throw refusal;
      const value = merge(defaults, desc);
      const handle = next++;
      this.lights.set(handle, value);
      return handle;
    },
    updateLight(handle, changes) {
      assert.ok(this.lights.has(handle));
      this.calls.push(['update', handle, structuredClone(changes)]);
      if (this.failUpdate(handle, changes)) throw refusal;
      this.lights.set(handle, merge(this.lights.get(handle), changes));
    },
    removeLight(handle) {
      assert.ok(this.lights.delete(handle), 'owned handle removed exactly once');
      this.calls.push(['remove', handle]);
    },
  };
}

test('complete ids retain handles across repeated, reordered and changed snapshots', () => {
  const e = engine();
  const owner = createMamaLightOwner(e);
  const a = light('node/left');
  const b = light('node/right');
  owner.prepare([a, b]).commit();
  const handles = [...e.lights.keys()];
  owner.prepare([a, b]).commit();
  owner.prepare([b, { ...a, intensity: 9 }]).commit();
  assert.deepEqual([...e.lights.keys()], handles);
  assert.equal(e.lights.get(handles[0]).intensity, 9);
  assert.equal(e.calls.filter(([op]) => op === 'add').length, 2);
  owner.prepare([b, light('replacement')]).commit();
  assert.equal(e.lights.has(handles[0]), false);
  assert.equal(e.lights.has(handles[1]), true);
  assert.equal(e.lights.size, 2);
  for (const empty of [undefined, []]) {
    owner.prepare(empty).commit();
    assert.equal(e.lights.size, 0);
    owner.prepare([a]).commit();
  }
  owner.dispose();
});

test('fixed copied placement applies once to initial and two changed immutable snapshots', () => {
  const e = engine();
  const placement = [10, -20, 30];
  const owner = createMamaLightOwner(e, placement);
  placement[0] = 999;
  const records = [light('a'), light('a', { t: [4, 5, 6] }), light('a', { t: [-2, 8, 1] })];
  const original = structuredClone(records);
  for (const record of records) {
    Object.freeze(record.t);
    Object.freeze(record);
    owner.prepare([record]).commit();
    assert.deepEqual(e.lights.get(1).position, record.t.map((v, i) => v + [10, -20, 30][i]));
    assert.deepEqual(e.lights.get(1).direction, [0, 0, -1]);
  }
  assert.deepEqual(records, original);
  assert.equal(e.lights.size, 1);
  for (const invalid of [null, [1, 2], [1, Infinity, 3], new Array(3)]) {
    assert.throws(() => createMamaLightOwner(e, invalid), rejectsWith('sdfm.scene_format'));
  }
});

test('snapshots reset omitted angles and kind-specific fields explicitly on the same handle', () => {
  const e = engine();
  const owner = createMamaLightOwner(e);
  owner.prepare([light('a', { kind: 'spot', angleDeg: 100, innerAngleDeg: 70 })]).commit();
  owner.prepare([light('a', { kind: 'spot', angleDeg: 40 })]).commit();
  assert.equal(e.lights.get(1).innerAngleDeg, undefined);
  assert.ok(e.calls.some(([op, handle, patch]) => op === 'update' && handle === 1 &&
    Object.hasOwn(patch, 'innerAngleDeg') && patch.innerAngleDeg === undefined));
  owner.prepare([light('a', { kind: 'area', size: [4, 5] })]).commit();
  owner.prepare([light('a')]).commit();
  assert.deepEqual(e.lights.get(1).size, [1, 1]);
  assert.equal(e.lights.get(1).angleDeg, 35);
  assert.deepEqual([...e.lights.keys()], [1]);
});

test('mapping, duplicate ids and concurrent preparation refuse without publishing changes', () => {
  const e = engine();
  const owner = createMamaLightOwner(e);
  for (const records of [[light('a'), light('a')], [light('a'), light('b', { s: [1, 1, 0] })],
    [light('a', { t: [null, 0, 0] })], [light(5)], null]) {
    assert.throws(() => owner.prepare(records), rejectsWith('sdfm.scene_format'));
  }
  assert.equal(e.calls.length, 0);
  const transaction = owner.prepare([light('a')]);
  assert.equal(e.lights.get(1).intensity, 0);
  assert.throws(() => owner.prepare([]), rejectsWith('sdfm.scene_format'));
  transaction.rollback();
  transaction.rollback();
  transaction.commit();
  assert.equal(e.lights.size, 0);
  owner.prepare([]).commit();
});

test('second creation failure releases staged additions and preserves the published set', () => {
  const e = engine();
  const owner = createMamaLightOwner(e);
  owner.prepare([light('old')]).commit();
  const before = structuredClone(e.lights);
  e.failAddAt = 3;
  assert.throws(() => owner.prepare([light('first'), light('second')]), (error) => error === refusal);
  assert.deepEqual(e.lights, before);
  assert.throws(() => owner.prepare([light('bad', { intensity: -1 })]), (error) => error === refusal);
  assert.deepEqual(e.lights, before);
  owner.prepare([]).commit();
});

test('failed commit restores updated descriptors and removes additions before any obsolete removal', () => {
  const e = engine();
  const owner = createMamaLightOwner(e);
  owner.prepare([light('a', { kind: 'spot', angleDeg: 100, innerAngleDeg: 70 }),
    light('b'), light('obsolete')]).commit();
  const before = structuredClone(e.lights);
  const transaction = owner.prepare([light('new'), light('a', { kind: 'spot', angleDeg: 40 }),
    light('b', { range: 10 })]);
  e.failUpdate = (handle, patch) => handle === 2 && patch.range === 10;
  assert.throws(() => transaction.commit(), (error) => error === refusal);
  assert.deepEqual(e.lights, before);
  transaction.rollback();
  transaction.commit();
  e.failUpdate = () => false;
  owner.prepare([light('a')]).commit();
  assert.deepEqual([...e.lights.keys()], [1]);
});

test('external application failure rolls back preparations; successful completion is idempotent', () => {
  const e = engine();
  const owner = createMamaLightOwner(e);
  owner.prepare([light('old')]).commit();
  const before = structuredClone(e.lights);
  const transaction = owner.prepare([light('new'), light('old', { intensity: 9 })]);
  const externalFailure = new Error('scene application failed');
  assert.throws(() => {
    try {
      throw externalFailure;
    } catch (error) {
      transaction.rollback();
      throw error;
    }
  }, (error) => error === externalFailure);
  assert.deepEqual(e.lights, before);
  const success = owner.prepare([light('old', { intensity: 9 })]);
  success.commit();
  const calls = e.calls.length;
  success.commit();
  success.rollback();
  assert.equal(e.calls.length, calls);
  assert.equal(e.lights.get(1).intensity, 9);
});

test('visibility preserves identity and authored values, rolls back failure and covers hidden snapshots', () => {
  const e = engine();
  const owner = createMamaLightOwner(e);
  owner.prepare([light('a'), light('b', { intensity: 6 })]).commit();
  const before = structuredClone(e.lights);
  e.failUpdate = (handle, patch) => handle === 2 && patch.intensity === 0;
  assert.throws(() => owner.setVisible(false), (error) => error === refusal);
  assert.deepEqual(e.lights, before);
  e.failUpdate = () => false;
  const pending = owner.prepare([light('a', { intensity: 9 }), light('c', { intensity: 7 })]);
  owner.setVisible(false);
  assert.ok([...e.lights.values()].every((desc) => desc.intensity === 0));
  pending.commit();
  owner.setVisible(false);
  assert.ok([...e.lights.values()].every((desc) => desc.intensity === 0));
  owner.setVisible(true);
  assert.deepEqual([...e.lights].map(([handle, desc]) => [handle, desc.intensity]), [[1, 9], [3, 7]]);
  owner.setVisible(true);
});

test('disposal aborts pending additions, removes handles once and refuses further owner mutations', () => {
  const e = engine();
  const owner = createMamaLightOwner(e);
  owner.prepare([light('old')]).commit();
  const pending = owner.prepare([light('new')]);
  owner.dispose();
  const calls = e.calls.length;
  owner.dispose();
  pending.rollback();
  pending.commit();
  assert.equal(e.calls.length, calls);
  assert.equal(e.lights.size, 0);
  assert.throws(() => owner.prepare(undefined), rejectsWith('sdfm.disposed'));
  assert.throws(() => owner.setVisible(false), rejectsWith('sdfm.disposed'));
  assert.throws(() => owner.setVisible(true), rejectsWith('sdfm.disposed'));
});
