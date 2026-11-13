import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import childProcess from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { containedPath, readJsonRecord, writeJsonRecord, captureInputs } from '../../examples/sky-weather/lib/evidence.mjs';
import {
  check, select, clean, main, parseArgs, attemptRecords, readBuildReceipt, storeRoot, STAGES,
} from './vg-job.mjs';

delete process.env.NMC_CACHE_DIR;
delete process.env.VG_RERENDER_ALL;

function scratch(t) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'vg-job-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  return directory;
}

function put(root, name, text) {
  const file = path.join(root, name);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, text);
  return file;
}

/** The staged lib/inventory.js of the fixture: cumulative rows, marked so the test sees them. */
function inventoryModule({ requiredFacts = true } = {}) {
  return `
const stages = ['bake', 'passes'];
export function validateInventory(scenes, stage) {
  const limit = stages.indexOf(stage);
  if (limit < 0) throw new Error('invalid stage: ' + stage);
  const rows = [];
  for (const namespace of scenes) {
    if (stages.indexOf(namespace.default.milestone) > limit) continue;
    for (const row of namespace.inventory) rows.push({ ...row, checked: true });
  }
  return rows;
}
${requiredFacts ? `export function validateRequiredFacts(row) {
  if (!row || row.checked !== true) throw new Error('invalid fixture row');
}
` : ''}`;
}

const BAKE = {
  id: 'bake-scene',
  milestone: 'bake',
  file: 'scenes/bake.js',
  rows: [
    { id: 'bake-finest-cut', filename: 'bake-finest-cut-cluster-colours.png', heldConstant: ['ground plane'] },
    { id: 'bake-three-cuts', filename: 'bake-three-cuts-coarser-left-to-right.png', heldConstant: ['ground plane'] },
  ],
};
const PASSES = {
  id: 'passes-scene',
  milestone: 'passes',
  file: 'scenes/passes.js',
  rows: [
    { id: 'dense-field-vg-on', filename: 'dense-field-vg-on-many-high-poly-instances.png', heldConstant: ['ground plane', 'sky'] },
    { id: 'dense-field-vg-off', filename: 'dense-field-vg-off-forward-path.png', heldConstant: ['ground plane', 'sky'] },
  ],
};
const CUMULATIVE = [...BAKE.rows, ...PASSES.rows];

function sceneModule(scene) {
  return `
export const inventory = ${JSON.stringify(scene.rows)};
export default {
  id: ${JSON.stringify(scene.id)},
  milestone: ${JSON.stringify(scene.milestone)},
  title: ${JSON.stringify(scene.id)},
};
`;
}

/** A temporary repository root with a built page under dist/virtual-geometry. */
function fixture(t, options = {}) {
  const parent = scratch(t);
  const root = path.join(parent, 'raa');
  const page = path.join(root, 'dist/virtual-geometry');
  const revision = '1'.repeat(40);
  const manifest = {
    'Cargo.toml': '[workspace]\n', 'Cargo.lock': 'version = 4\n',
    'rust-toolchain.toml': '[toolchain]\nchannel = "1.96.0"\n',
    '.nmc/commands.json': '[]\n',
    'examples/sky-weather/lib/inventory.js': 'export const criteria = [];\n',
  };
  for (const [name, text] of Object.entries(manifest)) put(root, name, text);
  put(root, '.git', 'gitdir: fixture\n');
  const scenes = options.scenes ?? [BAKE, PASSES];
  const state = { root, parent, page, revision, names: Object.keys(manifest), scenes: [], build: null };
  for (const scene of scenes) {
    const source = sceneModule(scene);
    const sourceName = `examples/virtual-geometry/${scene.file}`;
    put(root, sourceName, source);
    state.names.push(sourceName);
    state.scenes.push({ ...scene, source });
  }
  t.mock.method(childProcess, 'execFileSync', (program, args, settings = {}) => {
    if (program === 'cargo') {
      assert.equal(settings.cwd, root);
      return JSON.stringify({ workspace_root: root, packages: [{ dependencies: [] }] });
    }
    if (program === 'wasm-bindgen') return 'wasm-bindgen 0.2.129';
    if (program === (process.env.RUSTC || 'rustc')) return 'rustc 1.96.0';
    assert.equal(program, 'git');
    assert.equal(settings.cwd, root);
    if (args[0] === 'rev-parse' && args[1] === '--show-toplevel') return root;
    if (args[0] === 'rev-parse' && args[1] === '--verify' && args[2] === 'HEAD') return revision;
    if (args[0] === 'status') return '';
    if (args[0] === 'ls-files') return state.names.join('\0');
    throw new Error(`unexpected git ${args.join(' ')}`);
  });
  fs.mkdirSync(path.join(page, 'scenes'), { recursive: true });
  for (const scene of state.scenes) put(page, scene.file, scene.source);
  put(page, 'lib/inventory.js', inventoryModule(options.inventory));
  put(page, 'scenes.json', `${JSON.stringify(state.scenes.map(({ id, milestone, file }) => ({ id, milestone, file }))
    .sort((a, b) => a.id.localeCompare(b.id)), null, 2)}\n`);
  const preparationId = randomUUID();
  const build = {
    buildId: randomUUID(), preparationId, inputs: captureInputs(root, preparationId),
    builtAt: new Date().toISOString(), page, files: ['scenes.json', ...state.scenes.map((scene) => scene.file)],
    status: 'ready', error: null,
  };
  writeJsonRecord(containedPath(page, 'build.json'), build);
  state.build = build;
  return state;
}

function rewrite(record, mutate) {
  const value = JSON.parse(JSON.stringify(record));
  mutate(value);
  return value;
}

function capture(t) {
  const written = [];
  t.mock.method(process.stdout, 'write', (chunk) => { written.push(String(chunk)); return true; });
  return written;
}

test('STAGES and parseArgs follow the virtual-geometry stages and usage', () => {
  assert.deepEqual(STAGES, ['bake', 'passes']);
  assert.equal(Object.isFrozen(STAGES), true);
  assert.throws(() => parseArgs(['check', '--page', 'dist/virtual-geometry', '--stage', 'weather']),
    (error) => error.message === 'invalid stage: weather');
  assert.throws(() => parseArgs(['deploy', '--page', 'dist/virtual-geometry', '--stage', 'bake']),
    (error) => error.message.startsWith('usage: node web/common/vg-job.mjs')
      && error.message.includes('dist/virtual-geometry') && error.message.includes('<bake|passes>'));
  assert.deepEqual(parseArgs(['check', '--page', 'dist/virtual-geometry', '--stage', 'bake']),
    { action: 'check', page: 'dist/virtual-geometry', out: undefined, stage: 'bake' });
});

test('storeRoot follows NMC_CACHE_DIR and the resolved out otherwise', (t) => {
  const cache = scratch(t);
  const out = path.join(scratch(t), 'out/virtual-geometry/bake');
  t.after(() => { delete process.env.NMC_CACHE_DIR; });
  delete process.env.NMC_CACHE_DIR;
  assert.equal(storeRoot(out, 'bake'), path.resolve(out));
  process.env.NMC_CACHE_DIR = '   ';
  assert.equal(storeRoot(out, 'bake'), path.resolve(out));
  process.env.NMC_CACHE_DIR = cache;
  assert.equal(storeRoot(out, 'bake'), path.join(path.resolve(cache), 'virtual-geometry', 'bake'));
  assert.equal(storeRoot(out, 'passes'), path.join(path.resolve(cache), 'virtual-geometry', 'passes'));
});

test('check returns the rows of the built page inventory and refuses a partial validator module', async (t) => {
  const state = fixture(t);
  const bake = await check(state.page, 'bake');
  assert.deepEqual(bake.map((row) => row.id), BAKE.rows.map((row) => row.id));
  assert.deepEqual(bake.map((row) => row.filename), BAKE.rows.map((row) => row.filename));
  assert.equal(bake.every((row) => row.checked === true), true);
  assert.equal(bake.every((row) => Array.isArray(row.heldConstant)), true);
  const passes = await check(state.page, 'passes');
  assert.deepEqual(passes.map((row) => row.id), CUMULATIVE.map((row) => row.id));

  const partial = fixture(t);
  put(partial.page, 'lib/inventory.js', 'export function validateInventory() { return []; }\n');
  const file = path.join(partial.page, 'lib/inventory.js');
  await assert.rejects(check(partial.page, 'bake'),
    (error) => error.message === `${file}: validateInventory and validateRequiredFacts must be exported`);
});

test('readBuildReceipt accepts the ready fixture and refuses malformed receipts', (t) => {
  const state = fixture(t);
  assert.deepEqual(readBuildReceipt(state.page), state.build);

  const cases = [
    [(build) => { delete build.files; }, /fields must be exactly/],
    [(build) => { build.extra = true; }, /fields must be exactly/],
    [(build) => { build.status = 'failed'; build.error = 'compile failed'; }, /is failed: compile failed/],
  ];
  for (const [mutate, expected] of cases) {
    const broken = fixture(t);
    writeJsonRecord(containedPath(broken.page, 'build.json'), rewrite(broken.build, mutate));
    assert.throws(() => readBuildReceipt(broken.page), expected);
  }
});

test('main prints one row line per inventory row and the stage summary', async (t) => {
  const state = fixture(t);
  const written = capture(t);
  await main(['check', '--page', state.page, '--stage', 'bake']);
  assert.deepEqual(written, [
    ...BAKE.rows.map((row) => `${row.id} ${row.filename}\n`),
    `vg-job check: ok (bake, ${BAKE.rows.length} rows)\n`,
  ]);
  written.length = 0;
  await main(['check', '--page', state.page, '--stage', 'passes']);
  assert.deepEqual(written, [
    ...CUMULATIVE.map((row) => `${row.id} ${row.filename}\n`),
    `vg-job check: ok (passes, ${CUMULATIVE.length} rows)\n`,
  ]);
});

test('main select then clean print their summaries and write the stage records', async (t) => {
  const state = fixture(t);
  const out = path.join(state.root, 'out/virtual-geometry/bake');
  const written = capture(t);
  await main(['select', '--page', state.page, '--out', out, '--stage', 'bake']);
  assert.deepEqual(written, [`vg-job select: ok (bake, ${BAKE.rows.length} rows)\n`]);
  const selection = readJsonRecord(containedPath(out, 'selection.json'));
  assert.equal(selection.stage, 'bake');
  assert.deepEqual(selection.inventory.map((row) => row.id), BAKE.rows.map((row) => row.id));
  assert.equal(fs.existsSync(containedPath(out, 'current.json')), false);

  written.length = 0;
  await main(['clean', '--page', state.page, '--out', out, '--stage', 'bake']);
  assert.deepEqual(written, [`vg-job clean: ok (bake, ${BAKE.rows.length} rows)\n`]);
  const pointer = readJsonRecord(containedPath(out, 'current.json'));
  assert.equal(pointer.selectionId, selection.selectionId);
  assert.equal(pointer.directory, `attempts/${pointer.attemptId}`);
  const records = attemptRecords(pointer.directory);
  assert.deepEqual(readJsonRecord(containedPath(out, records.selection)), selection);
  assert.deepEqual(readJsonRecord(containedPath(out, records.build)), state.build);
  assert.equal(readJsonRecord(containedPath(out, records.attempt)).status, 'allocated');
  assert.equal(fs.existsSync(containedPath(out, pointer.directory)), true);
});

test('select and clean keep each stage in its own directory', async (t) => {
  const state = fixture(t);
  const bake = path.join(state.root, 'out/virtual-geometry/bake');
  const passes = path.join(state.root, 'out/virtual-geometry/passes');
  const written = capture(t);
  await select(state.page, bake, 'bake');
  await clean(state.page, bake, 'bake');
  written.length = 0;
  await select(state.page, passes, 'passes');
  await clean(state.page, passes, 'passes');
  assert.deepEqual(readJsonRecord(containedPath(passes, 'selection.json')).inventory.map((row) => row.id),
    CUMULATIVE.map((row) => row.id));
  assert.notEqual(readJsonRecord(containedPath(bake, 'current.json')).attemptId,
    readJsonRecord(containedPath(passes, 'current.json')).attemptId);
});
