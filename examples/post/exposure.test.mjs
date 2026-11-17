// Exposure scene and its room geometry: content builders, then the exposure-room scene rows and hooks.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  box,
  checker,
  facadeCourses,
  plane,
  roomWithOpening,
  stripes,
  tiledGround,
} from './lib/content.js';
import { validateInventory } from './lib/inventory.js';
import * as exposure from './scenes/exposure.js';
import { assertSubjectsInView } from '../../web/common/view-check.mjs';

test('box has 24 vertices, 36 indices and unit normals', () => {
  const m = box([1, 2, 3], [0.5, 1, 1.5]);
  assert.equal(m.positions.length, 72);
  assert.equal(m.indices.length, 36);
  for (let i = 0; i < m.normals.length; i += 3) {
    assert.ok(Math.abs(Math.hypot(m.normals[i], m.normals[i + 1], m.normals[i + 2]) - 1) < 1e-6);
  }
});

test('plane(4, 2) is four vertices at y 0 with +Y normals', () => {
  const m = plane(4, 2);
  assert.equal(m.positions.length, 12);
  for (let i = 0; i < 4; i++) {
    assert.equal(m.positions[i * 3 + 1], 0);
    assert.deepEqual([...m.normals.slice(i * 3, i * 3 + 3)], [0, 1, 0]);
  }
});

const room = {
  width: 6,
  height: 3,
  depth: 6,
  wall: 0.3,
  opening: { zMin: -4, zMax: -2, yMin: 0.3, yMax: 2.5 },
};

test('roomWithOpening: sub-meshes, indices, opening and bounds', () => {
  const m = roomWithOpening(room);
  assert.equal(m.subMeshes.length, 3);
  let next = 0;
  for (const s of m.subMeshes) {
    assert.equal(s.first, next);
    next += s.count;
  }
  assert.equal(next, m.indices.length);
  const n = m.positions.length / 3;
  assert.ok(m.indices.every((i) => i < n));
  const p = (i) => [m.positions[i * 3], m.positions[i * 3 + 1], m.positions[i * 3 + 2]];
  for (let t = 0; t < m.indices.length; t += 3) {
    const c = [0, 1, 2].map((a) => (p(m.indices[t])[a] + p(m.indices[t + 1])[a] + p(m.indices[t + 2])[a]) / 3);
    const inside = c[0] >= 3 && c[0] <= 3.3 && c[2] > -4 && c[2] < -2 && c[1] > 0.3 && c[1] < 2.5;
    assert.ok(!inside, `triangle ${t / 3} centroid in the opening`);
  }
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) {
    p(i).forEach((v, a) => {
      lo[a] = Math.min(lo[a], v);
      hi[a] = Math.max(hi[a], v);
    });
  }
  const near = (a, b) => a.every((v, i) => Math.abs(v - b[i]) < 1e-5);
  assert.ok(near(lo, [-3.3, -0.3, -6.3]));
  assert.ok(near(hi, [3.3, 3.3, 0.3]));
});

test('tiledGround splits tiles into even and odd', () => {
  const m = tiledGround({ x0: 0, z0: 0, columns: 3, rows: 2, size: 1 });
  assert.deepEqual(m.subMeshes.map((s) => s.count), [18, 18]);
});

test('facadeCourses leaves the skip rect free', () => {
  const skip = { zMin: -1, zMax: 1, yMin: 1, yMax: 2.5 };
  const m = facadeCourses({ x: 0, zMin: -4, zMax: 4, yMin: 0, yMax: 4, course: 0.5, depth: 0.2, skip });
  assert.ok(m.indices.length > 0);
  // Every box is 24 vertices; compare each box's z/y extent with the skip rect.
  for (let b = 0; b < m.positions.length / 72; b++) {
    const lo = [Infinity, Infinity];
    const hi = [-Infinity, -Infinity];
    for (let v = 0; v < 24; v++) {
      const o = b * 72 + v * 3;
      lo[0] = Math.min(lo[0], m.positions[o + 1]);
      hi[0] = Math.max(hi[0], m.positions[o + 1]);
      lo[1] = Math.min(lo[1], m.positions[o + 2]);
      hi[1] = Math.max(hi[1], m.positions[o + 2]);
    }
    const overlaps = lo[0] < skip.yMax && hi[0] > skip.yMin && lo[1] < skip.zMax && hi[1] > skip.zMin;
    assert.ok(!overlaps, `box ${b} overlaps the skip rect`);
  }
});

test('stripes and checker texels', () => {
  const s = stripes(8, 4);
  assert.deepEqual([...Array(8).keys()].map((x) => s.data[x * 4]), [255, 255, 0, 0, 255, 255, 0, 0]);
  assert.ok([...Array(64).keys()].every((i) => s.data[i * 4 + 3] === 255));
  const c = checker(4, 2);
  const at = (x, y) => c.data[(y * 4 + x) * 4];
  assert.deepEqual([at(0, 0), at(1, 0), at(0, 1), at(1, 1)], [255, 0, 0, 255]);
});

test('stripes(8, 1) throws', () => {
  assert.throws(
    () => stripes(8, 1),
    /content: stripes needs an integer size > 0 and an integer period >= 2/,
  );
});

const variantOf = (id) => exposure.default.variants.find((v) => v.id === id);

// Consumes validateInventory (examples/post/lib/inventory.js:120; its doc comment is at lines 107-119),
// given this scene's namespace and stage 'exposure'.
test('validateInventory returns the four exposure-room rows in order, with their ids and filenames', () => {
  const rows = validateInventory([exposure], 'exposure');
  assert.deepEqual(rows.map((row) => row.id), [
    'exposure-interior-on',
    'exposure-interior-off',
    'exposure-exterior-on',
    'exposure-exterior-off',
  ]);
  assert.deepEqual(rows.map((row) => row.filename), [
    'exposure-interior-on-room-readable.png',
    'exposure-interior-off-room-too-dark.png',
    'exposure-exterior-on-sunlit-facade-readable.png',
    'exposure-exterior-off-sunlit-facade-blown-out.png',
  ]);
});

test('each row names the pair of its mode; the two variants of a pair share a preset', () => {
  for (const row of exposure.inventory) {
    assert.equal(row.pair, variantOf(row.mode).measurement.pair);
  }
  for (const pair of ['exposure-interior', 'exposure-exterior']) {
    const members = exposure.default.variants.filter((v) => v.measurement.pair === pair);
    assert.deepEqual(members.map((v) => v.measurement.state).sort(), ['off', 'on']);
    assert.equal(members[0].measurement.preset, members[1].measurement.preset);
  }
});

test('settle holds exactly on the -on variants; every variant sets exposureEv 1 and matches its autoExposure state', () => {
  assert.equal(exposure.default.variants.length, 4);
  for (const v of exposure.default.variants) {
    const on = v.id.endsWith('-on');
    if (on) assert.equal(v.settle, true, v.id);
    else assert.equal(Object.hasOwn(v, 'settle'), false, v.id);
    assert.equal(v.settings.toneMapping.exposureEv, 1, v.id);
    assert.equal(v.settings.autoExposure.enabled, on, v.id);
  }
});

test('each row holds the fixed exposure in its held-constant text, the value every variant sets', () => {
  for (const row of exposure.inventory) {
    assert.ok(row.heldConstant.includes(`tone mapping exposureEv ${exposure.FIXED_EXPOSURE_EV}`), row.id);
  }
  for (const v of exposure.default.variants) {
    assert.equal(v.settings.toneMapping.exposureEv, exposure.FIXED_EXPOSURE_EV, v.id);
  }
});

// The paired intent of the -off rows, both judged at one fixed exposure: the room's own light (sky fill
// and ambient) sits far below the direct sun, so the interior stays dark while the sunlit exterior clips.
// The bound is a ratio between the scene's own constants: each fill value is at most FILL_TO_SUN_MAX of the
// sun intensity, which leaves the walls more than three decades below the sunlit facade (an estimate).
const FILL_TO_SUN_MAX = 1e-4;
test('the room fill (sky diffuse, sky specular, ambient) sits far below the direct sun', () => {
  const { SUN, ENV_LIGHTING, AMBIENT } = exposure;
  const fills = { 'sky diffuse': ENV_LIGHTING.diffuse, 'sky specular': ENV_LIGHTING.specular, ambient: AMBIENT.intensity };
  for (const [name, value] of Object.entries(fills)) {
    assert.ok(value <= FILL_TO_SUN_MAX * SUN.intensity, `${name} ${value} is not far below the sun ${SUN.intensity}`);
  }
});

// Stops above middle grey (0.18) of a sunlit diffuse surface at the fixed exposure: the luminance of its
// sRGB colour, times the sun, times cos / pi, times 2^exposureEv.
const srgbToLinear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const luminance = ([r, g, b]) => 0.2126 * srgbToLinear(r) + 0.7152 * srgbToLinear(g) + 0.0722 * srgbToLinear(b);
const sunlitStops = (sRgb, cos) => Math.log2(
  luminance(sRgb) * exposure.SUN.intensity * cos / Math.PI * 2 ** exposure.FIXED_EXPOSURE_EV / 0.18,
);
test('the sun clips the sunlit ground and facade at the fixed exposure: both sit past 5.5 stops above middle grey', () => {
  const { SUN, COLORS } = exposure;
  const length = Math.hypot(...SUN.direction);
  const cosOnGround = -SUN.direction[1] / length;
  const cosOnFacade = -SUN.direction[0] / length;
  // The darker tile of the ground and the darker course of the facade; 5.1 stops was the brighter tile at intensity 70.
  assert.ok(sunlitStops(COLORS.tileOdd, cosOnGround) > 5.5, `ground tile at ${sunlitStops(COLORS.tileOdd, cosOnGround)} stops`);
  assert.ok(sunlitStops(COLORS.courseOdd, cosOnFacade) > 5.5, `facade course at ${sunlitStops(COLORS.courseOdd, cosOnFacade)} stops`);
});

test('the two -off rows share one fixed exposureEv; each -on row differs from its -off partner only in the auto exposure switch', () => {
  const offs = exposure.default.variants.filter((v) => v.id.endsWith('-off'));
  assert.equal(offs.length, 2);
  assert.deepEqual([...new Set(offs.map((v) => v.settings.toneMapping.exposureEv))], [exposure.FIXED_EXPOSURE_EV]);
  for (const on of exposure.default.variants.filter((v) => v.id.endsWith('-on'))) {
    const off = variantOf(on.id.replace(/-on$/, '-off'));
    assert.deepEqual({ ...on.settings, autoExposure: null }, { ...off.settings, autoExposure: null }, on.id);
    assert.deepEqual(on.selection, off.selection, on.id);
    assert.equal(on.settings.autoExposure.enabled, true, on.id);
    assert.equal(off.settings.autoExposure.enabled, false, off.id);
  }
});

// Consumes assertSubjectsInView (web/common/view-check.mjs:80) on each variant's selection: both cameras.
test('assertSubjectsInView passes for the interior and the exterior camera', () => {
  for (const v of exposure.default.variants) {
    assertSubjectsInView(v.selection, `${exposure.default.id}/${v.id}`);
  }
});

const engineStub = {
  autoExposureState: () => ({ exposureEv: 2.5, averageLuminance: 0.02, settled: true, pending: false }),
  autoExposure: () => ({ enabled: true, compensationEv: 0.263, lowPercent: 0.2, highPercent: 0.7, speedUp: 2.4, speedDown: 1.1, minEv: -6, maxEv: 6 }),
};

test('observe passes all three assertions on interior-on; interior-off fails exposure-ev and exposure-enabled', async () => {
  const on = await exposure.observe(engineStub, variantOf('interior-on'), {});
  assert.deepEqual(on.assertions.map((a) => [a.name, a.passed]), [['exposure-settled', true], ['exposure-ev', true], ['exposure-enabled', true]]);
  assert.deepEqual(on.values, { 'exposure-settled': true, 'exposure-ev': 2.5 });
  const off = await exposure.observe(engineStub, variantOf('interior-off'), {});
  assert.deepEqual(off.assertions.map((a) => [a.name, a.passed]), [['exposure-settled', true], ['exposure-ev', false], ['exposure-enabled', false]]);
});

test('build sets the environment lighting, bloom and the camera, and sets neither tone mapping nor automatic exposure', async () => {
  const calls = [];
  let handle = 0;
  const engine = new Proxy({}, {
    get: (_, method) => (...args) => {
      calls.push([method, ...args]);
      return handle++;
    },
  });
  const variant = variantOf('exterior-on');
  const { instances } = await exposure.default.build(engine, {}, variant);
  const named = (method) => calls.filter(([name]) => name === method);
  assert.deepEqual(named('setEnvironmentLighting'), [['setEnvironmentLighting', exposure.ENV_LIGHTING]]);
  assert.deepEqual(named('setBloom'), [['setBloom', exposure.BLOOM]]);
  assert.deepEqual(named('setCamera'), [['setCamera', variant.selection.camera]]);
  assert.equal(named('setToneMapping').length, 0);
  assert.equal(named('setAutoExposure').length, 0);
  assert.ok(instances.length > 0);
});
