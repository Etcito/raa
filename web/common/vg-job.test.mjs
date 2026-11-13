import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import childProcess from 'node:child_process';
import { randomUUID } from 'node:crypto';
import zlib from 'node:zlib';
import { containedPath, readJsonRecord, writeJsonRecord, captureInputs } from '../../examples/sky-weather/lib/evidence.mjs';
import {
  check, select, clean, render, fresh, judge, main, parseArgs, attemptRecords, readBuildReceipt, storeRoot, STAGES,
} from './vg-job.mjs';

delete process.env.NMC_CACHE_DIR;
delete process.env.VG_RERENDER_ALL;

/** Base64 of an 8-bit RGB PNG whose pixel (x, y) is pixel(x, y). */
function png(width, height, pixel) {
  const raw = Buffer.alloc((width * 3 + 1) * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) raw.set(pixel(x, y), y * (width * 3 + 1) + 1 + x * 3);
  const chunk = (type, body) => {
    const head = Buffer.alloc(8);
    head.writeUInt32BE(body.length);
    head.write(type, 4, 'latin1');
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(zlib.crc32(Buffer.concat([head.subarray(4), body])));
    return Buffer.concat([head, body, crc]);
  };
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0);
  header.writeUInt32BE(height, 4);
  header[8] = 8; header[9] = 2;
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', header),
    chunk('IDAT', zlib.deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]).toString('base64');
}

/** One deterministic, non-blank capture per row of a group. */
function capturePng(row) {
  const seed = [...row.id].reduce((sum, character) => sum + character.charCodeAt(0), 0);
  return png(8, 8, (x, y) => [20 + (seed % 80) + y * 10, 40 + x * 7, 200 - y * 9]);
}

/** Point the job at a fresh project store for the rest of the test. */
function projectStore(t) {
  process.env.NMC_CACHE_DIR = scratch(t);
  t.after(() => { delete process.env.NMC_CACHE_DIR; delete process.env.VG_RERENDER_ALL; });
  return process.env.NMC_CACHE_DIR;
}

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
    { id: 'bake-finest-cut', scenario: 'bake-scene', mode: 'finest-cut',
      filename: 'bake-finest-cut-cluster-colours.png',
      expectedVisibleBehavior: 'The finest cut draws one colour patch per cluster with no hole.',
      heldConstant: ['ground plane'] },
    { id: 'bake-three-cuts', scenario: 'bake-scene', mode: 'three-cuts',
      filename: 'bake-three-cuts-coarser-left-to-right.png',
      expectedVisibleBehavior: 'Three cuts show fewer, larger patches toward the right with the same silhouette.',
      heldConstant: ['ground plane'] },
  ],
};
const PASSES = {
  id: 'passes-scene',
  milestone: 'passes',
  file: 'scenes/passes.js',
  rows: [
    { id: 'dense-field-vg-on', scenario: 'passes-scene', mode: 'dense-field-vg-on',
      filename: 'dense-field-vg-on-many-high-poly-instances.png',
      expectedVisibleBehavior: 'Dense high-poly instances render through the virtualized path.',
      heldConstant: ['ground plane', 'sky'] },
    { id: 'dense-field-vg-off', scenario: 'passes-scene', mode: 'dense-field-vg-off',
      filename: 'dense-field-vg-off-forward-path.png',
      expectedVisibleBehavior: 'Dense high-poly instances render on the forward path.',
      heldConstant: ['ground plane', 'sky'] },
  ],
};
const CUMULATIVE = [...BAKE.rows, ...PASSES.rows];

function sceneModule(scene) {
  const variants = scene.rows.map((row) => ({ id: row.mode }));
  return `
export const factNames = ${JSON.stringify(scene.factNames ?? [`${scene.id}-ready`])};
export const inventory = ${JSON.stringify(scene.rows)};
export default {
  id: ${JSON.stringify(scene.id)},
  milestone: ${JSON.stringify(scene.milestone)},
  title: ${JSON.stringify(scene.id)},
  variants: ${JSON.stringify(variants)},
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

/**
 * A rendered attempt of the virtual-geometry page: its own project store, the selected page and
 * the managed transport the stages use instead of a Windows GPU client.
 */
async function renderFixture(t, scenes = [BAKE], options = {}) {
  const state = fixture(t, { scenes });
  const cache = projectStore(t);
  state.stage = options.stage ?? 'bake';
  state.out = path.join(state.root, 'out/virtual-geometry', state.stage);
  state.store = path.join(cache, 'virtual-geometry', state.stage);
  state.selection = await select(state.page, state.out, state.stage);
  state.attempt = await clean(state.page, state.out, state.stage);
  state.records = attemptRecords(state.attempt.directory);
  state.read = (name) => readJsonRecord(containedPath(state.out, name));
  state.calls = [];
  state.inputs = [];
  state.transport = {
    async publishGpuPage(page, settings) {
      state.calls.push('publish');
      assert.equal(page, state.page);
      assert.deepEqual(settings, { timeoutMs: 300000 });
      return { pageRef: 'host-page' };
    },
    async runGpuPageOp(pageRef, op, input, settings) {
      state.calls.push(`${input.scenario}/${input.mode}`);
      state.inputs.push({ ...input });
      assert.equal(pageRef, 'host-page');
      assert.equal(op, 'run');
      assert.equal(settings.timeoutMs, 300000);
      assert.equal(input.remainingMs, 270000);
      assert.equal(input.width, 1280);
      assert.equal(input.height, 720);
      const { width, height, remainingMs, ...provenance } = input;
      const rows = state.selection.inventory.filter((row) => row.scenario === input.scenario && row.mode === input.mode);
      const result = {
        provenance,
        browser: {
          userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
          platform: 'Win32', devicePixelRatio: 1,
        },
        adapter: {
          vendor: 'nvidia', architecture: 'ampere', device: 'NVIDIA GeForce RTX 3090',
          description: 'NVIDIA GeForce RTX 3090 (0x00002204)', isFallbackAdapter: false,
        },
        hardware: true,
        images: rows.map((row) => ({ id: row.id, filename: row.filename, mime: 'image/png',
          data: capturePng(row), observation: `Actual capture ${row.id}` })),
        facts: { readiness: { status: 'ready', stage: state.stage, error: null } },
        measurements: [],
        errors: { console: [], page: [], gpu: [] },
      };
      return { jobId: `job-${state.calls.length}`, clientId: 'windows-client', result };
    },
  };
  state.render = () => render(state.page, state.out, state.stage, state.transport);
  return state;
}

/** Allocate a later attempt of the same stage and clear the transport log. */
async function cleanAgain(s) {
  s.attempt = await clean(s.page, s.out, s.stage);
  s.records = attemptRecords(s.attempt.directory);
  s.calls.length = 0;
  return s;
}

/** A new build receipt of the current sources: new buildId and preparationId, then select again. */
async function rebuild(s, mutate) {
  mutate?.();
  const preparationId = randomUUID();
  s.build = { ...s.build, buildId: randomUUID(), preparationId,
    inputs: captureInputs(s.root, preparationId), builtAt: new Date().toISOString() };
  writeJsonRecord(containedPath(s.page, 'build.json'), s.build);
  s.selection = await select(s.page, s.out, s.stage);
  return s.build;
}

test('render publishes once, sends the virtual-geometry canvas and stores the attempt under NMC_CACHE_DIR', async (t) => {
  const s = await renderFixture(t, [BAKE]);
  const run = await s.render();
  assert.equal(run.status, 'passed');
  assert.equal(s.read('current.json').attemptId, s.attempt.attemptId);
  assert.equal(s.read(s.records.attempt).status, 'passed');
  assert.equal(s.calls.filter((call) => call === 'publish').length, 1);
  assert.equal(run.jobs.length, BAKE.rows.length);
  assert.equal(s.inputs.length, BAKE.rows.length);
  for (const input of s.inputs) {
    assert.equal(input.width, 1280);
    assert.equal(input.height, 720);
    assert.equal(input.remainingMs, 270000);
  }
  for (const row of BAKE.rows) {
    assert.ok(fs.existsSync(containedPath(s.out, s.records.image(row.filename))));
    assert.ok(fs.existsSync(containedPath(s.store, s.records.image(row.filename))));
  }
  assert.ok(fs.existsSync(containedPath(s.store, s.records.run)));
  assert.ok(fs.existsSync(containedPath(s.store, s.records.input(0))));
  assert.ok(fs.existsSync(containedPath(s.store, s.records.result(0))));
});

test('unchanged rows and build keep every image and the CLI reports the kept count', async (t) => {
  const s = await renderFixture(t, [BAKE]);
  const first = s.attempt;
  await s.render();
  await cleanAgain(s);
  const kept = await s.render();
  assert.deepEqual(s.calls, []);
  assert.equal(kept.kept.length, BAKE.rows.length);
  for (const entry of kept.kept) {
    assert.equal(entry.attemptId, first.attemptId);
    assert.equal(entry.selectionId, first.selectionId);
    assert.equal(entry.buildId, first.buildId);
  }
  await cleanAgain(s);
  const written = capture(t);
  await main(['render', '--page', s.page, '--out', s.out, '--stage', s.stage]);
  assert.deepEqual(written, [
    `vg-job render: kept ${BAKE.rows.length}, rendered 0\n`,
    `vg-job render: ok (bake, ${BAKE.rows.length} rows)\n`,
  ]);
});

test('a changed inventory row renders only its group again', async (t) => {
  const s = await renderFixture(t, [BAKE]);
  await s.render();
  const changed = BAKE.rows[0];
  const scene = s.scenes.find((entry) => entry.id === changed.scenario);
  const rows = scene.rows.map((row) => (row.id === changed.id
    ? { ...row, filename: 'bake-finest-cut-renamed.png' } : row));
  const changedFile = 'scenes/bake-changed.js';
  put(s.page, changedFile, sceneModule({ ...scene, rows }));
  const index = readJsonRecord(containedPath(s.page, 'scenes.json'));
  for (const entry of index) if (entry.id === scene.id) entry.file = changedFile;
  writeJsonRecord(containedPath(s.page, 'scenes.json'), index);
  s.build = { ...s.build, files: [...s.build.files, changedFile] };
  writeJsonRecord(containedPath(s.page, 'build.json'), s.build);
  s.selection = await select(s.page, s.out, s.stage);
  await cleanAgain(s);
  const run = await s.render();
  assert.deepEqual(s.calls, ['publish', `${changed.scenario}/${changed.mode}`]);
  assert.deepEqual(run.kept.map((entry) => entry.id).sort(),
    BAKE.rows.filter((row) => row.id !== changed.id).map((row) => row.id).sort());
  assert.ok(run.images.some((image) => image.filename === 'bake-finest-cut-renamed.png'));
});

test('a changed build renders every group again while identical inputs keep them', async (t) => {
  const s = await renderFixture(t, [BAKE]);
  await s.render();
  await rebuild(s);
  await cleanAgain(s);
  const kept = await s.render();
  assert.deepEqual(s.calls, []);
  assert.equal(kept.kept.length, BAKE.rows.length);

  await rebuild(s, () => put(s.root, 'examples/sky-weather/lib/inventory.js', 'export const criteria = [7];\n'));
  await cleanAgain(s);
  const rerendered = await s.render();
  assert.deepEqual(rerendered.kept, []);
  assert.deepEqual(s.calls, ['publish', ...BAKE.rows.map((row) => `${row.scenario}/${row.mode}`)]);
  assert.equal(rerendered.images.length, BAKE.rows.length);
});

test('VG_RERENDER_ALL=1 renders every group again', async (t) => {
  const s = await renderFixture(t, [BAKE]);
  await s.render();
  await cleanAgain(s);
  process.env.VG_RERENDER_ALL = '1';
  const run = await s.render();
  assert.deepEqual(run.kept, []);
  assert.deepEqual(s.calls, ['publish', ...BAKE.rows.map((row) => `${row.scenario}/${row.mode}`)]);
});

test('fresh passes the kept attempt and writes fresh.json', async (t) => {
  const s = await renderFixture(t, [BAKE]);
  await s.render();
  await cleanAgain(s);
  await s.render();
  const record = await fresh(s.page, s.out, s.stage);
  assert.equal(record.status, 'passed');
  assert.deepEqual(record.errors, []);
  assert.equal(record.attemptId, s.attempt.attemptId);
  assert.equal(record.buildId, s.build.buildId);
  assert.deepEqual(s.read(s.records.fresh), record);
});

test('fresh refuses a fallback adapter naming the physical GPU adapter requirement', async (t) => {
  const s = await renderFixture(t, [BAKE]);
  await s.render();
  const run = s.read(s.records.run);
  run.jobs[0].result.adapter = { ...run.jobs[0].result.adapter, isFallbackAdapter: true };
  run.jobs[0].result.hardware = false;
  writeJsonRecord(containedPath(s.out, s.records.run), run);
  await assert.rejects(fresh(s.page, s.out, s.stage), /physical GPU adapter is required/);
  assert.equal(s.read(s.records.fresh).status, 'failed');
});

test('judge passes the held constants and writes one verdict per row to the attempt and the store', async (t) => {
  const s = await renderFixture(t, [BAKE]);
  await s.render();
  const calls = [];
  const record = await judge(s.page, s.out, s.stage, {
    judge: async (capture, description, options) => {
      calls.push({ filename: path.basename(capture), description, heldConstant: options.heldConstant });
      return { exit: 0, score: 8, output: 'ok' };
    },
    log: () => {},
  });
  assert.equal(record.status, 'passed');
  assert.deepEqual(record.errors, []);
  assert.equal(record.images.length, BAKE.rows.length);
  for (const row of BAKE.rows) {
    const verdict = record.images.find((entry) => entry.id === row.id);
    assert.equal(verdict.filename, row.filename);
    assert.equal(verdict.description, row.expectedVisibleBehavior);
    assert.equal(verdict.minScore, 6);
    assert.equal(verdict.exit, 0);
    assert.equal(verdict.score, 8);
    const call = calls.find((entry) => entry.filename === row.filename);
    assert.equal(call.description, row.expectedVisibleBehavior);
    assert.deepEqual(call.heldConstant, row.heldConstant);
  }
  assert.deepEqual(s.read(s.records.judge), record);
  assert.deepEqual(readJsonRecord(containedPath(s.store, s.records.judge)), record);
});

test('judge names the images its zero budget skipped and reuses an earlier verdict at zero budget', async (t) => {
  const s = await renderFixture(t, [BAKE]);
  await s.render();
  const calls = [];
  const stub = async () => {
    calls.push('judge');
    return { exit: 0, score: 8, output: 'JUDGE PASS: 8/10' };
  };
  await assert.rejects(judge(s.page, s.out, s.stage, { judge: stub, budgetMs: 0, log: () => {} }),
    /not judged within the 0 s judge budget/);
  assert.deepEqual(calls, []);
  const failed = s.read(s.records.judge);
  assert.equal(failed.status, 'failed');
  for (const row of BAKE.rows) assert.ok(failed.errors.some((error) => error.includes(row.filename)));

  const first = await judge(s.page, s.out, s.stage, { judge: stub, log: () => {} });
  assert.equal(first.status, 'passed');
  const judgedAt = new Map(first.images.map((entry) => [entry.id, entry.judgedAt]));
  calls.length = 0;
  await cleanAgain(s);
  const run = await s.render();
  assert.equal(run.kept.length, BAKE.rows.length);
  const reused = await judge(s.page, s.out, s.stage, { judge: stub, budgetMs: 0, log: () => {} });
  assert.equal(reused.status, 'passed');
  assert.deepEqual(calls, []);
  for (const entry of reused.images) assert.equal(entry.judgedAt, judgedAt.get(entry.id));
});
