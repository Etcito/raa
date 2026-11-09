// node --test web/common/lights-job.test.mjs: CPU-only selection and visibility checks.
import assert from 'node:assert/strict';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { check, inView, main, select, variantProblems } from './lights-job.mjs';

const CAM = { eye: [0, 2, 5], target: [0, 0, 0], fovYDeg: 60, near: 0.1, far: 100 };
const FLOOR = { min: [-50, -1, -50], max: [50, 0, 50] };
const BEHIND = { min: [0, 0, 10], max: [1, 1, 11] };
const EDGE = { min: [4.5, 0, -1], max: [5, 1, 1] };
const scene = (id = 'point', subjects = [FLOOR]) => ({ id, milestone: 'm02', variants: ['on', 'off'].map((variant) =>
  ({ id: variant, description: `${id} ${variant}`, judge: true, timeMs: 1000, selection: { camera: CAM, subjects } })) });
const read = (path) => JSON.parse(readFileSync(path, 'utf8'));
const source = (def) => `export default { ...${JSON.stringify(def)}, build: async () => {} };`;

function writePage(t, defs) {
  const root = mkdtempSync(join(tmpdir(), 'lights-job-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const page = join(root, 'dist/lights');
  mkdirSync(join(page, 'scenes'), { recursive: true });
  const index = defs.map((def) => {
    const file = `scenes/${def.id}.js`;
    writeFileSync(join(page, file), source(def));
    return { id: def.id, milestone: def.milestone, file };
  });
  writeFileSync(join(page, 'scenes.json'), JSON.stringify(index));
  return { root, page, out: join(root, 'out/lights') };
}

test('inView accepts an enclosing floor and rejects hidden or off-axis boxes', () => {
  assert.equal(inView(CAM, FLOOR), true);
  assert.equal(inView(CAM, BEHIND), false);
  assert.equal(inView(CAM, { min: [0, 20, -1], max: [1, 21, 1] }), false);
  assert.equal(inView(CAM, EDGE), true);
  assert.equal(inView({ ...CAM, aspect: 1 }, EDGE), false);
  assert.equal(inView({ ...CAM, eye: [0, 0, 0], target: [0, 0, 0] }, FLOOR), false);
});

test('select reads only the requested build and writes its on/off pairs', async (t) => {
  const point = writePage(t, [scene()]);
  const spot = writePage(t, [scene('spot')]);
  writeFileSync(join(point.root, 'scene.js'), source(scene('ghost')));
  const pairs = await select(point.page, point.out);
  assert.deepEqual(pairs.map((p) => `${p.scene}--${p.variant}`), ['point--on', 'point--off']);
  const doc = read(join(point.out, 'selection.json'));
  assert.equal(doc.page, point.page);
  assert.deepEqual(doc.pairs.map((p) => p.description), ['point on', 'point off']);
  assert.deepEqual((await check(spot.page)).map((p) => p.scene), ['spot', 'spot']);
});

test('select refuses missing, empty, malformed and escaping scene indexes', async (t) => {
  const missing = writePage(t, [scene()]);
  rmSync(join(missing.page, 'scenes.json'));
  await assert.rejects(select(missing.page, missing.out), /scenes\.json is missing: build the page first/);
  const empty = writePage(t, [scene()]);
  writeFileSync(join(empty.page, 'scenes.json'), '[]');
  await assert.rejects(check(empty.page), /nonempty array/);
  const malformed = writePage(t, [scene()]);
  writeFileSync(join(malformed.page, 'scenes.json'), '[{"id":"point"}]');
  await assert.rejects(check(malformed.page), /entry 0 is not \{ id, milestone, file \}/);
  const escaping = writePage(t, [scene()]);
  writeFileSync(join(escaping.root, 'outside.js'), 'export default {};');
  writeFileSync(join(escaping.page, 'scenes.json'), JSON.stringify([{ id: 'point', milestone: 'm02', file: '../../outside.js' }]));
  await assert.rejects(check(escaping.page), /is not in the page build/);
  const absent = writePage(t, [scene()]);
  rmSync(join(absent.page, 'scenes/point.js'));
  await assert.rejects(check(absent.page), /module scenes\/point\.js is not in the page build/);
});

test('select refuses invalid scene metadata', async (t) => {
  const mismatch = writePage(t, [scene()]);
  writeFileSync(join(mismatch.page, 'scenes/point.js'), source(scene('ghost')));
  await assert.rejects(check(mismatch.page), /metadata is invalid/);
  const noBuild = writePage(t, [scene()]);
  writeFileSync(join(noBuild.page, 'scenes/point.js'), `export default ${JSON.stringify(scene())};`);
  await assert.rejects(check(noBuild.page), /metadata is invalid/);
  const duplicate = writePage(t, [scene(), scene()]);
  await assert.rejects(check(duplicate.page), /duplicate scene/);
  const wrongVariant = writePage(t, [scene()]);
  const def = scene();
  def.variants[0].id = 'both';
  writeFileSync(join(wrongVariant.page, 'scenes/point.js'), source(def));
  await assert.rejects(check(wrongVariant.page), /variant both is not on or off/);
});

test('select refuses invalid variant metadata', async (t) => {
  const cases = [
    [(s) => { s.variants[0].description = ' '; }, /no description/],
    [(s) => { s.variants[0].judge = false; }, /judge must be true/],
    [(s) => { s.variants[0].timeMs = -1; }, /timeMs is not a duration/],
    [(s) => { s.variants[0].selection.camera = { eye: [0, 0, 0], target: [0, 0, 0] }; }, /camera/],
    [(s) => { s.variants[0].selection.subjects = []; }, /at least one subject/],
    [(s) => { s.variants[0].selection.subjects = [{ min: [0, 0, 0], max: [-1, 1, 1] }]; }, /min<=max/],
  ];
  for (const [mutate, pattern] of cases) {
    const def = scene();
    mutate(def);
    const page = writePage(t, [def]);
    await assert.rejects(select(page.page, page.out), pattern);
  }
});

test('select refuses subjects outside the camera view', async (t) => {
  const behind = writePage(t, [scene('point', [FLOOR, BEHIND])]);
  await assert.rejects(select(behind.page, behind.out), /subjects\[1\] is outside the camera view/);
  const above = writePage(t, [scene('point', [{ min: [0, 30, 0], max: [1, 31, 1] }])]);
  await assert.rejects(check(above.page), /subjects\[0\] is outside the camera view/);
});

test('check validates without writing and the CLI runs both commands', async (t) => {
  const f = writePage(t, [scene()]);
  assert.equal((await check(f.page)).length, 2);
  assert.equal(existsSync(f.out), false);
  await main(['check', '--page', f.page]);
  await assert.rejects(main(['select', '--page', f.page]), /usage/);
  await main(['select', '--page', f.page, '--out', f.out]);
  assert.equal(read(join(f.out, 'selection.json')).pairs.length, 2);
});

test('every lights scene is discovered with each preset and on/off variant selectable', async () => {
  const scenes = new URL('../../examples/lights/scenes/', import.meta.url);
  const ids = [];
  for (const file of readdirSync(scenes).filter((name) => name.endsWith('.js')).sort()) {
    const scene = (await import(new URL(file, scenes))).default;
    assert.equal(scene.id, file.slice(0, -3), 'build-lights.sh discovers a scene by its file name');
    assert.match(scene.milestone, /^m\d\d$/);
    assert.deepEqual(scene.variants.map((variant) => variant.id), ['on', 'off']);
    for (const variant of scene.variants) {
      assert.deepEqual(variantProblems(variant), [], `${scene.id}--${variant.id}`);
    }
    ids.push(scene.id);
  }
  for (const id of ['point', 'spot', 'directional', 'area', 'many-lights', 'attached-light', 'motion']) {
    assert.ok(ids.includes(id), `${id} is discovered`);
  }
});
