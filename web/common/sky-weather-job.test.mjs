import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import childProcess from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { containedPath, readJsonRecord, writeJsonRecord, captureInputs } from '../../examples/sky-weather/lib/evidence.mjs';
import { createSkyPageRunner } from '../../examples/sky-weather/lib/runtime.js';
import { check, select, clean, render, fresh, judge, parseArgs, main, attemptRecords, RECORD_FIELDS } from './sky-weather-job.mjs';

function scratch(t) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'sky-job-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  return directory;
}

function put(root, name, text) {
  const file = path.join(root, name);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, text);
  return file;
}

const foundation = { id: 'foundation-scene', milestone: 'foundation', file: 'scenes/foundation.mjs', filename: 'foundation-sky-and-sphere.png' };
const atmosphere = { id: 'atmosphere-scene', milestone: 'atmosphere', file: 'scenes/atmosphere.mjs', filename: 'atmosphere-clear-day.png' };
const paired = {
  id: 'paired-scene', milestone: 'foundation', file: 'scenes/paired.mjs',
  variants: [
    { id: 'on', measurement: { pair: 'cloud-shadows', state: 'on', preset: 'fair', features: { shadows: true } } },
    { id: 'off', measurement: { pair: 'cloud-shadows', state: 'off', preset: 'fair', features: { shadows: false } } },
  ],
  rows: [
    { id: 'paired-on-image', mode: 'on', filename: 'paired-cloud-shadows-on.png', pair: 'cloud-shadows' },
    { id: 'paired-off-image', mode: 'off', filename: 'paired-cloud-shadows-off.png', pair: 'cloud-shadows' },
  ],
};

const DEFAULT_SELECTION = {
  camera: { eye: [0, 1, 5], target: [0, 1, 0], fovYDeg: 60, near: 0.1, far: 100 },
  subjects: [{ name: 'sphere', min: [-1, 0, -1], max: [1, 2, 1] }],
};
const DEFAULT_FACTS = ['snapshot:sky', 'runtime:passes', 'runtime:viewSizes', 'runtime:ready', 'assertion:sky-visible'];

function sceneModule(scene) {
  const factNames = scene.factNames ?? ['sky-visible'];
  const variants = scene.variants ?? [{ id: 'default' }];
  const rows = scene.rows ?? [{ id: `${scene.id}-image`, mode: 'default', filename: scene.filename, pair: null }];
  const inventory = rows.map((row) => ({
    id: row.id,
    criterionName: row.criterionName ?? 'Sky weather scene',
    scenario: scene.id,
    mode: row.mode,
    filename: row.filename,
    expectedVisibleBehavior: row.expectedVisibleBehavior ?? 'The accepted sky state renders above the ground receiver.',
    requiredFacts: row.requiredFacts ?? scene.requiredFacts ?? DEFAULT_FACTS,
    pair: row.pair ?? null,
  }));
  const definitions = variants.map((variant) => ({
    id: variant.id,
    description: variant.description ?? `${variant.id} state`,
    selection: variant.selection ?? DEFAULT_SELECTION,
    ...(variant.measurement ? { measurement: variant.measurement } : {}),
  }));
  return `
export const factNames = ${JSON.stringify(factNames)};
export const inventory = ${JSON.stringify(inventory)};
export async function observe() { return { sky: null, time: null, weather: null, diagnostics: null, assertions: [], edits: [] }; }
export default {
  id: '${scene.id}',
  milestone: '${scene.milestone}',
  title: '${scene.id}',
  build() {},
  variants: ${JSON.stringify(definitions)},
};
`;
}

function factsRecord(overrides = {}) {
  return {
    sky: { model: 'physical' },
    time: null,
    weather: null,
    diagnostics: null,
    assertions: [{ name: 'sky-visible', passed: true, detail: 'The sky renders above the ground receiver.' }],
    edits: [],
    passes: [{ name: 'forward', role: 'pass', enabled: true, order: 60, required: true }],
    viewSizes: [{ view: 0, width: 1280, height: 720 }],
    readiness: { status: 'ready', stage: 'foundation', error: null },
    ...overrides,
  };
}

function measurementRecord(spec) {
  return {
    pair: spec.pair, state: spec.state, preset: spec.preset, features: { ...spec.features },
    width: 1280, height: 720, warmup: 30, frames: 300,
    frameTimesMs: Array.from({ length: 300 }, () => 10),
    fps: 100, p50Ms: 10, p95Ms: 10, p99Ms: 10,
  };
}

function variantOf(state, scenario, mode) {
  const scene = state.scenes.find((entry) => entry.id === scenario);
  const variants = scene?.variants ?? [{ id: 'default' }];
  return variants.find((entry) => entry.id === mode) ?? null;
}

function fixture(t, options = {}) {
  const parent = scratch(t);
  const root = path.join(parent, 'raa');
  const page = path.join(root, 'dist/sky-weather');
  const revision = '1'.repeat(40);
  const manifest = {
    'Cargo.toml': '[workspace]\n', 'Cargo.lock': 'version = 4\n',
    'rust-toolchain.toml': '[toolchain]\nchannel = "1.96.0"\n',
    '.nmc/commands.json': '[]\n',
    'examples/sky-weather/lib/inventory.js': 'export const criteria = [];\n',
  };
  for (const [name, text] of Object.entries(manifest)) put(root, name, text);
  put(root, '.git', 'gitdir: fixture\n');
  const scenes = options.scenes ?? [foundation];
  const state = { root, parent, page, revision, names: Object.keys(manifest), scenes: [], build: null };
  for (const scene of scenes) {
    const source = scene.source ?? sceneModule(scene);
    const sourceName = `examples/sky-weather/scenes/${path.basename(scene.file)}`;
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
  put(page, 'scenes.json', `${JSON.stringify(state.scenes.map(({ id, milestone, file }) => ({ id, milestone, file }))
    .sort((a, b) => a.id.localeCompare(b.id)), null, 2)}\n`);
  const preparationId = randomUUID();
  state.preparationId = preparationId;
  const build = {
    buildId: randomUUID(), preparationId, inputs: captureInputs(root, preparationId),
    builtAt: new Date().toISOString(), page, files: ['scenes.json', ...state.scenes.map((scene) => scene.file)],
    status: 'ready', error: null,
  };
  if (options.build) options.build(build, state);
  writeJsonRecord(containedPath(page, 'build.json'), build);
  state.build = build;
  return state;
}

function rewrite(record, mutate) {
  const value = JSON.parse(JSON.stringify(record));
  mutate(value);
  return value;
}

test('check validates the ready build and resolves cumulative inventory without GPU work', async (t) => {
  const state = fixture(t, { scenes: [foundation, atmosphere] });
  const first = await check(state.page, 'foundation');
  assert.deepEqual(first.map((row) => row.id), ['foundation-scene-image']);
  const second = await check(state.page, 'atmosphere');
  assert.deepEqual(second.map((row) => row.id).sort(), ['atmosphere-scene-image', 'foundation-scene-image']);
  await assert.rejects(check(state.page, 'weather'), /unavailable/);
  await assert.rejects(check(state.page, 'planet'), /invalid stage/);
});

test('stale or invalid build receipts reject before scene resolution', async (t) => {
  const failed = fixture(t, { build: (build) => { build.status = 'failed'; build.error = 'actual compiler failure'; } });
  await assert.rejects(check(failed.page, 'foundation'), /actual compiler failure/);

  const inherited = fixture(t, { build: (build) => { build.error = 'stale ready error'; } });
  await assert.rejects(check(inherited.page, 'foundation'), /records an error/);

  const changed = fixture(t);
  put(changed.root, 'examples/sky-weather/lib/inventory.js', 'export const criteria = [1];\n');
  await assert.rejects(check(changed.page, 'foundation'), /changed input/);

  const missing = fixture(t);
  fs.rmSync(path.join(missing.page, 'scenes/foundation.mjs'));
  await assert.rejects(check(missing.page, 'foundation'), /missing or empty|is missing/);

  const unstaged = fixture(t);
  writeJsonRecord(containedPath(unstaged.page, 'build.json'),
    rewrite(unstaged.build, (build) => { build.files = ['scenes/foundation.mjs']; }));
  await assert.rejects(check(unstaged.page, 'foundation'), /scenes\.json/);

  const wrongPage = fixture(t);
  writeJsonRecord(containedPath(wrongPage.page, 'build.json'),
    rewrite(wrongPage.build, (build) => { build.page = path.join(wrongPage.root, 'dist/other'); }));
  await assert.rejects(check(wrongPage.page, 'foundation'), /page does not name/);

  const wrongPreparation = fixture(t);
  writeJsonRecord(containedPath(wrongPreparation.page, 'build.json'),
    rewrite(wrongPreparation.build, (build) => { build.preparationId = randomUUID(); }));
  await assert.rejects(check(wrongPreparation.page, 'foundation'), /preparation identity differs/);

  const staleBuiltAt = fixture(t);
  writeJsonRecord(containedPath(staleBuiltAt.page, 'build.json'),
    rewrite(staleBuiltAt.build, (build) => { build.builtAt = 'not-a-time'; }));
  await assert.rejects(check(staleBuiltAt.page, 'foundation'), /builtAt/);
});

test('unknown fact selectors and duplicate registry names reject at check', async (t) => {
  const unknown = fixture(t, { scenes: [{ ...foundation, requiredFacts: ['snapshot:sky', 'assertion:missing-fact'] }] });
  await assert.rejects(check(unknown.page, 'foundation'), /unknown selector|assertion:missing-fact/);

  const unknownSnapshot = fixture(t, { scenes: [{ ...foundation, requiredFacts: ['snapshot:clouds'] }] });
  await assert.rejects(check(unknownSnapshot.page, 'foundation'), /unknown selector/);

  const duplicate = fixture(t, { scenes: [{ ...foundation, factNames: ['sky-visible', 'sky-visible'] }] });
  await assert.rejects(check(duplicate.page, 'foundation'), /invalid or duplicate fact name/);

  const duplicateSelector = fixture(t, { scenes: [{ ...foundation, requiredFacts: ['snapshot:sky', 'snapshot:sky'] }] });
  await assert.rejects(check(duplicateSelector.page, 'foundation'), /duplicate selector/);
});

test('containment rejects escaping staged files, scene modules and output roots', async (t) => {
  const buildEscape = fixture(t);
  writeJsonRecord(containedPath(buildEscape.page, 'build.json'),
    rewrite(buildEscape.build, (build) => { build.files = ['scenes.json', '../outside.mjs']; }));
  await assert.rejects(check(buildEscape.page, 'foundation'), /unsafe/);

  const sceneEscape = fixture(t);
  put(sceneEscape.page, 'scenes.json', JSON.stringify([{ id: 'foundation-scene', milestone: 'foundation', file: '../outside.mjs' }]));
  await assert.rejects(check(sceneEscape.page, 'foundation'), /not a staged build output|unsafe/);

  const linked = fixture(t);
  const outside = path.join(linked.parent, 'outside');
  fs.mkdirSync(outside);
  fs.symlinkSync(outside, path.join(linked.root, 'out'));
  await assert.rejects(select(linked.page, path.join(linked.root, 'out/foundation'), 'foundation'), /symlink/);
  assert.deepEqual(fs.readdirSync(outside), []);

  await assert.rejects(select(linked.page, `${linked.root}/out/../escape`, 'foundation'), /unsafe path/);
});

test('select freezes the build, inputs and cumulative inventory at the stage root', async (t) => {
  const state = fixture(t, { scenes: [foundation, atmosphere] });
  const out = path.join(state.root, 'out/sky-weather/foundation');
  const selection = await select(state.page, out, 'atmosphere');
  const stored = readJsonRecord(containedPath(out, 'selection.json'));
  assert.deepEqual(stored, selection);
  assert.deepEqual(Object.keys(stored).sort(), [...RECORD_FIELDS.selection].sort());
  assert.equal(stored.stage, 'atmosphere');
  assert.equal(stored.buildId, state.build.buildId);
  assert.equal(stored.preparationId, state.build.preparationId);
  assert.deepEqual(stored.inputs, state.build.inputs);
  assert.deepEqual(stored.inventory.map((row) => row.filename).sort(),
    ['atmosphere-clear-day.png', 'foundation-sky-and-sphere.png']);
  assert.equal(fs.existsSync(path.join(out, 'current.json')), false);
  assert.equal(fs.existsSync(path.join(out, 'attempts')), false);
  const again = await select(state.page, out, 'atmosphere');
  assert.notEqual(again.selectionId, selection.selectionId);
  assert.deepEqual(readJsonRecord(containedPath(out, 'selection.json')), again);
});

test('clean allocates immutable attempts non-destructively and writes the exact pointer', async (t) => {
  const state = fixture(t);
  const out = path.join(state.root, 'out/sky-weather/foundation');
  const selection = await select(state.page, out, 'foundation');
  const first = await clean(state.page, out, 'foundation');
  assert.match(first.attemptId, /^[0-9a-f-]{36}$/);
  assert.equal(first.directory, `attempts/${first.attemptId}`);
  assert.equal(first.selectionId, selection.selectionId);
  assert.equal(first.buildId, state.build.buildId);
  assert.equal(first.status, 'allocated');
  assert.equal(first.error, null);
  assert.deepEqual(Object.keys(first).sort(), [...RECORD_FIELDS.attempt].sort());
  assert.deepEqual(readJsonRecord(containedPath(out, 'current.json')), {
    selectionId: selection.selectionId, attemptId: first.attemptId, directory: first.directory,
  });
  const records = attemptRecords(first.directory);
  const frozenSelection = readJsonRecord(containedPath(out, records.selection));
  const frozenBuild = readJsonRecord(containedPath(out, records.build));
  const frozenAttempt = readJsonRecord(containedPath(out, records.attempt));
  assert.deepEqual(frozenSelection, selection);
  assert.deepEqual(frozenBuild, state.build);
  assert.deepEqual(frozenAttempt, first);
  assert.equal(fs.existsSync(path.join(out, records.run)), false);
  const second = await clean(state.page, out, 'foundation');
  assert.notEqual(second.attemptId, first.attemptId);
  assert.equal(second.selectionId, selection.selectionId);
  assert.deepEqual(readJsonRecord(containedPath(out, records.selection)), frozenSelection);
  assert.deepEqual(readJsonRecord(containedPath(out, records.build)), frozenBuild);
  assert.deepEqual(readJsonRecord(containedPath(out, records.attempt)), frozenAttempt);
  assert.deepEqual(readJsonRecord(containedPath(out, 'current.json')), {
    selectionId: selection.selectionId, attemptId: second.attemptId, directory: second.directory,
  });
  assert.deepEqual(fs.readdirSync(path.join(out, 'attempts')).sort(), [first.attemptId, second.attemptId].sort());
});

test('clean refuses stale selections without allocating an attempt', async (t) => {
  const state = fixture(t);
  const out = path.join(state.root, 'out/sky-weather/foundation');
  await select(state.page, out, 'foundation');
  writeJsonRecord(containedPath(state.page, 'build.json'),
    rewrite(state.build, (build) => { build.buildId = randomUUID(); }));
  await assert.rejects(clean(state.page, out, 'foundation'), /belongs to build|select again/);
  assert.equal(fs.existsSync(path.join(out, 'attempts')), false);
  assert.equal(fs.existsSync(path.join(out, 'current.json')), false);

  writeJsonRecord(containedPath(state.page, 'build.json'), state.build);
  const selectionFile = containedPath(out, 'selection.json');
  writeJsonRecord(selectionFile, rewrite(readJsonRecord(selectionFile), (selection) => { selection.inventory = []; }));
  await assert.rejects(clean(state.page, out, 'foundation'), /inventory changed|select again/);
  assert.equal(fs.existsSync(path.join(out, 'attempts')), false);

  const absent = path.join(state.root, 'out/sky-weather/weather');
  await assert.rejects(select(state.page, absent, 'weather'), /unavailable/);
  await assert.rejects(clean(state.page, path.join(state.root, 'out/sky-weather/atmosphere'), 'atmosphere'), /unavailable/);
});

test('CLI parses the protocol and preserves selection', async (t) => {
  const state = fixture(t);
  const out = path.join(state.root, 'out/sky-weather/foundation');
  assert.throws(() => parseArgs(['select', '--page', state.page, '--out', out]), /invalid stage/);
  assert.equal(parseArgs(['render', '--page', state.page, '--out', out, '--stage', 'foundation']).action, 'render');
  assert.equal(parseArgs(['fresh', '--page', state.page, '--out', out, '--stage', 'foundation']).action, 'fresh');
  assert.equal(parseArgs(['judge', '--page', state.page, '--out', out, '--stage', 'foundation']).action, 'judge');
  assert.throws(() => parseArgs(['verify', '--page', state.page, '--out', out, '--stage', 'foundation']), /usage/);
  assert.throws(() => parseArgs(['select', '--page', state.page, '--stage', 'foundation']), /usage/);
  assert.deepEqual(parseArgs(['check', '--page', state.page, '--stage', 'foundation']),
    { action: 'check', page: state.page, out: undefined, stage: 'foundation' });
  t.mock.method(process.stdout, 'write', () => true);
  await main(['select', '--page', state.page, '--out', out, '--stage', 'foundation']);
  assert.equal(readJsonRecord(containedPath(out, 'selection.json')).stage, 'foundation');
});

async function renderFixture(t, scenes = [foundation], options = {}) {
  const state = fixture(t, { scenes });
  state.out = path.join(state.root, 'out/sky-weather', scenes.length > 1 ? 'atmosphere' : 'foundation');
  state.stage = scenes.length > 1 ? 'atmosphere' : 'foundation';
  state.selection = await select(state.page, state.out, state.stage);
  state.attempt = await clean(state.page, state.out, state.stage);
  state.records = attemptRecords(state.attempt.directory);
  state.read = (name) => readJsonRecord(containedPath(state.out, name));
  state.calls = [];
  state.transport = {
    async publishGpuPage(page, settings) {
      state.calls.push('publish');
      assert.equal(page, state.page);
      assert.deepEqual(settings, { timeoutMs: 300000 });
      return { pageRef: 'host-page' };
    },
    async runGpuPageOp(pageRef, op, input, settings) {
      state.calls.push(input.scenario);
      assert.equal(pageRef, 'host-page');
      assert.equal(op, 'run');
      assert.equal(settings.timeoutMs, 300000);
      assert.equal(input.remainingMs, 270000);
      assert.equal(input.width, 1280);
      assert.equal(input.height, 720);
      const { width, height, remainingMs, ...provenance } = input;
      const rows = state.selection.inventory.filter((row) => row.scenario === input.scenario && row.mode === input.mode);
      const variant = variantOf(state, input.scenario, input.mode);
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
        json: { legacy: 'must not unwrap' }, nested: [1, { values: [false, null] }],
        images: rows.map((row) => ({ id: row.id, filename: row.filename, mime: 'image/png',
          data: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jf1sAAAAASUVORK5CYII=',
          observation: `Actual capture ${row.id}` })),
        facts: state.factsFor ? state.factsFor(input, rows) : factsRecord(),
        measurements: variant?.measurement ? [measurementRecord(variant.measurement)] : [],
        errors: { console: [], page: [], gpu: [] },
      };
      return { jobId: `job-${state.calls.length}`, clientId: 'windows-client',
        result: state.resultFor ? state.resultFor(result, input, rows) : result };
    },
  };
  if (options.factsFor) state.factsFor = options.factsFor;
  if (options.resultFor) state.resultFor = options.resultFor;
  state.render = () => render(state.page, state.out, state.stage, state.transport);
  return state;
}

test('render publishes once, saves direct results and named captures, and refuses reuse', async (t) => {
  const s = await renderFixture(t, [foundation, atmosphere]);
  const run = await s.render();
  assert.deepEqual(Object.keys(run), RECORD_FIELDS.run);
  assert.equal(run.status, 'passed');
  assert.equal(s.calls.filter((call) => call === 'publish').length, 1);
  assert.equal(run.jobs.length, 2);
  for (const [index, job] of run.jobs.entries()) {
    assert.deepEqual(s.read(s.records.result(index)), job.result);
    assert.deepEqual(job.result.nested, [1, { values: [false, null] }]);
    assert.equal(s.read(s.records.input(index)).remainingMs, 270000);
  }
  for (const image of run.images) {
    assert.equal(image.path, `images/${image.filename}`);
    assert.ok(fs.statSync(containedPath(s.out, `${s.attempt.directory}/${image.path}`)).size > 8);
    assert.equal(image.clientId, 'windows-client');
  }
  assert.deepEqual(s.read(s.records.run), run);
  assert.equal(s.read(s.records.attempt).status, 'passed');
  await assert.rejects(s.render(), /already started/);
});

test('render preserves failed operation captures and originating download attribution', async (t) => {
  const s = await renderFixture(t);
  const runOp = s.transport.runGpuPageOp;
  s.transport.runGpuPageOp = async (...args) => {
    const reply = await runOp(...args);
    writeJsonRecord(path.join(args[3].out, 'result.json'), {
      job_id: reply.jobId, client_id: reply.clientId, result: reply.result,
    });
    throw new Error('managed output download timeout');
  };
  await assert.rejects(s.render(), /managed output download timeout/);
  const run = s.read(s.records.run);
  assert.equal(run.status, 'failed');
  assert.equal(run.jobs[0].error, 'managed output download timeout');
  assert.equal(run.images.length, 1);
  assert.deepEqual(s.read(s.records.result(0)), run.jobs[0].result);
  await assert.rejects(s.render(), /already started/);
});

for (const failure of ['GPU waiting timeout after 60s', 'operation budget expired', 'GPU job running timeout after 300s']) {
  test(`render records ${failure} without resetting or retrying the allowance`, async (t) => {
    const s = await renderFixture(t);
    s.transport.runGpuPageOp = async () => { throw new Error(failure); };
    await assert.rejects(s.render(), { message: failure });
    const run = s.read(s.records.run);
    assert.equal(run.jobs[0].result, null);
    assert.equal(run.jobs[0].error, failure);
    assert.equal(s.read(s.records.result(0)), null);
    assert.equal(s.read(s.records.attempt).status, 'failed');
  });
}

test('superseded pointers fail while retaining recoverable captures', async (t) => {
  const s = await renderFixture(t);
  const runOp = s.transport.runGpuPageOp;
  s.transport.runGpuPageOp = async (...args) => {
    const result = await runOp(...args);
    await clean(s.page, s.out, s.stage);
    return result;
  };
  await assert.rejects(s.render(), /superseded/);
  assert.equal(s.read(s.records.run).images.length, 1);
  assert.equal(s.read(s.records.attempt).status, 'failed');
  assert.notEqual(s.read('current.json').attemptId, s.attempt.attemptId);
});

test('render rejects altered frozen records, stale builds and symlink outputs before publication', async (t) => {
  for (const kind of ['build', 'selection', 'attempt', 'stale', 'symlink']) {
    const s = await renderFixture(t);
    if (kind === 'stale') {
      put(s.root, 'examples/sky-weather/lib/inventory.js', 'changed source');
    } else if (kind === 'symlink') {
      fs.symlinkSync(s.parent, path.join(s.out, s.attempt.directory, 'jobs'));
    } else {
      const name = s.records[kind];
      const record = s.read(name);
      record.buildId = randomUUID();
      writeJsonRecord(containedPath(s.out, name), record);
    }
    await assert.rejects(s.render(), /differs|differ|belongs to build|changed input|symlink/);
    assert.deepEqual(s.calls, []);
  }
});

test('publication failure leaves a complete failed run and consumes the attempt', async (t) => {
  const s = await renderFixture(t);
  s.transport.publishGpuPage = async () => { throw new Error('publication refused'); };
  await assert.rejects(s.render(), /publication refused/);
  const run = s.read(s.records.run);
  assert.deepEqual(Object.keys(run), RECORD_FIELDS.run);
  assert.deepEqual(run.jobs, []);
  assert.deepEqual(run.errors, ['publication refused']);
  await assert.rejects(s.render(), /already started/);
});

test('null and array direct results are retained but cannot pass as sky output', async (t) => {
  for (const result of [null, [1, { nested: [true] }]]) {
    const s = await renderFixture(t);
    s.transport.runGpuPageOp = async () => ({ jobId: 'host-job', clientId: 'host-client', result });
    await assert.rejects(s.render(), /missing or malformed direct operation result/);
    assert.deepEqual(s.read(s.records.result(0)), result);
    assert.deepEqual(s.read(s.records.run).jobs[0].result, result);
  }
});

test('capture containment failures retain the direct result and other valid captures', async (t) => {
  const s = await renderFixture(t);
  const runOp = s.transport.runGpuPageOp;
  s.transport.runGpuPageOp = async (...args) => {
    const reply = await runOp(...args);
    reply.result.images.unshift({ ...reply.result.images[0], filename: '../../escape.png' });
    return reply;
  };
  await assert.rejects(s.render(), /unexpected capture identity/);
  const run = s.read(s.records.run);
  assert.equal(run.images.length, 1);
  assert.equal(run.jobs[0].result.images.length, 2);
  assert.equal(fs.existsSync(path.join(s.out, 'escape.png')), false);
});

async function freshFixture(t, scenes = [foundation], options = {}) {
  const s = await renderFixture(t, scenes, options);
  await s.render();
  s.fresh = () => fresh(s.page, s.out, s.stage);
  s.judge = (judgeOptions = {}) => judge(s.page, s.out, s.stage, judgeOptions);
  return s;
}

function retamper(s, mutate) {
  const run = s.read(s.records.run);
  mutate(run);
  writeJsonRecord(containedPath(s.out, s.records.run), run);
}

test('fresh verifies the frozen attempt and actual GPU results and writes fresh.json', async (t) => {
  const s = await freshFixture(t);
  const record = await s.fresh();
  assert.deepEqual(Object.keys(record).sort(), [...RECORD_FIELDS.fresh].sort());
  assert.equal(record.status, 'passed');
  assert.deepEqual(record.errors, []);
  assert.equal(record.selectionId, s.selection.selectionId);
  assert.equal(record.attemptId, s.attempt.attemptId);
  assert.equal(record.buildId, s.build.buildId);
  assert.deepEqual(s.read(s.records.fresh), record);
  assert.equal(s.read(s.records.attempt).status, 'passed');
  assert.equal(s.read(s.records.run).status, 'passed');
  await s.fresh();
});

test('creating adapter identity propagates from the page through render records to fresh', async (t) => {
  const hardware = { vendor: 'nvidia', architecture: 'ampere', device: '', description: '', isFallbackAdapter: false };
  for (const [info, expectedError] of [
    [hardware, null],
    [{ ...hardware, vendor: '', architecture: '' }, /adapter identity is blank/],
    [{ ...hardware, isFallbackAdapter: true }, /physical GPU/],
    [{ ...hardware, description: 'SwiftShader' }, /physical GPU|software/],
  ]) await t.test(expectedError?.source ?? 'physical GPU', async (t) => {
    const s = await renderFixture(t);
    const originalOp = s.transport.runGpuPageOp;
    let requests = 0;
    const gpu = {
      async requestAdapter() {
        requests++;
        return {
          // WebIDL identity fields are prototype accessors, not own properties.
          info: Object.create(Object.defineProperties({}, Object.fromEntries(
            Object.entries(info).map(([key, value]) => [key, { get: () => value }]),
          ))),
          async requestDevice() { return { addEventListener() {} }; },
        };
      },
    };
    const globals = {
      navigator: { gpu, userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/156.0.0.0', platform: 'Win32' },
      devicePixelRatio: 1,
      requestAnimationFrame() {}, cancelAnimationFrame() {},
    };
    for (const [key, value] of Object.entries(globals)) {
      const descriptor = Object.getOwnPropertyDescriptor(globalThis, key);
      Object.defineProperty(globalThis, key, { configurable: true, value });
      t.after(() => {
        if (descriptor) Object.defineProperty(globalThis, key, descriptor);
        else delete globalThis[key];
      });
    }
    s.transport.runGpuPageOp = async (...args) => {
      const reply = await originalOp(...args);
      const namespace = await import(pathToFileURL(path.join(s.page, foundation.file)).href);
      const context = { drawImage() {}, fillRect() {}, fillText() {} };
      const canvas = { ownerDocument: { createElement: () => ({
        getContext: () => context,
        toDataURL: () => `data:image/png;base64,${reply.result.images[0].data}`,
      }) } };
      const runner = createSkyPageRunner({
        canvas, status: { textContent: '' }, lib: {},
        scenes: new Map([[foundation.id, { ...namespace, observe: async () => factsRecord() }]]),
        load: async () => ({ createEngine: async () => {
          await (await gpu.requestAdapter()).requestDevice();
          return {
            resize() {}, render() {}, dispose() {}, passes: () => factsRecord().passes,
            stats: () => ({ adapter: { vendor: '', architecture: '', device: '', description: '', isFallbackAdapter: false }, hardware: true }),
          };
        } }),
      });
      try { reply.result = await runner.run(args[2]); }
      finally { runner.dispose(); }
      return reply;
    };
    const run = await s.render();
    assert.equal(requests, 1, 'evidence must use the adapter that created the engine device');
    assert.deepEqual(run.jobs[0].result.adapter, info);
    assert.equal(run.jobs[0].result.hardware, !info.isFallbackAdapter && info.description !== 'SwiftShader');
    assert.deepEqual(s.read(s.records.result(0)).adapter, info);
    assert.deepEqual(s.read(s.records.run).jobs[0].result.adapter, info);
    if (expectedError) {
      await assert.rejects(fresh(s.page, s.out, s.stage), expectedError);
      assert.equal(s.read(s.records.fresh).status, 'failed');
    } else {
      assert.equal((await fresh(s.page, s.out, s.stage)).status, 'passed');
    }
  });
});

test('fresh rejects superseded pointers, stale builds and altered frozen records', async (t) => {
  const superseded = await freshFixture(t);
  await clean(superseded.page, superseded.out, superseded.stage);
  await assert.rejects(superseded.fresh(), /allocated|superseded|completed GPU render/);
  const current = readJsonRecord(containedPath(superseded.out, 'current.json'));
  assert.equal(superseded.read(attemptRecords(current.directory).fresh).status, 'failed');

  const rebuilt = await freshFixture(t);
  writeJsonRecord(containedPath(rebuilt.page, 'build.json'),
    rewrite(rebuilt.build, (build) => { build.buildId = randomUUID(); }));
  await assert.rejects(rebuilt.fresh(), /belongs to build|select again/);

  const changed = await freshFixture(t);
  put(changed.root, 'examples/sky-weather/lib/inventory.js', 'export const criteria = [2];\n');
  await assert.rejects(changed.fresh(), /changed input/);

  const frozenBuild = await freshFixture(t);
  writeJsonRecord(containedPath(frozenBuild.out, frozenBuild.records.build),
    rewrite(frozenBuild.build, (build) => { build.buildId = randomUUID(); }));
  await assert.rejects(frozenBuild.fresh(), /frozen build differs/);

  const failedAttempt = await freshFixture(t);
  writeJsonRecord(containedPath(failedAttempt.out, failedAttempt.records.attempt),
    rewrite(failedAttempt.attempt, (attempt) => { attempt.status = 'failed'; attempt.error = 'render failed'; }));
  await assert.rejects(failedAttempt.fresh(), /failed: render failed|completed GPU render/);

  const failedRun = await freshFixture(t);
  retamper(failedRun, (run) => { run.status = 'failed'; run.errors = ['actual GPU failure']; });
  await assert.rejects(failedRun.fresh(), /run is failed|actual GPU failure/);
});

test('fresh rejects missing, duplicate, unexpected and stale images', async (t) => {
  const missing = await freshFixture(t);
  retamper(missing, (run) => { run.images = []; });
  await assert.rejects(missing.fresh(), /records 0 image|selected image/);

  const duplicate = await freshFixture(t);
  retamper(duplicate, (run) => { run.images.push({ ...run.images[0] }); });
  await assert.rejects(duplicate.fresh(), /records 2 image|duplicate run image/);

  const unexpected = await freshFixture(t);
  retamper(unexpected, (run) => { run.images[0].id = 'unexpected-image'; });
  await assert.rejects(unexpected.fresh(), /not a selected capture|selected image/);

  const changedBytes = await freshFixture(t);
  const changedImage = changedBytes.read(changedBytes.records.run).images[0];
  const changedFile = containedPath(changedBytes.out, changedBytes.records.image(changedImage.filename));
  const bytes = fs.readFileSync(changedFile);
  bytes[bytes.length - 1] ^= 0xff;
  fs.writeFileSync(changedFile, bytes);
  await assert.rejects(changedBytes.fresh(), /differs from the direct operation payload/);

  const absent = await freshFixture(t);
  const absentImage = absent.read(absent.records.run).images[0];
  fs.rmSync(containedPath(absent.out, absent.records.image(absentImage.filename)));
  await assert.rejects(absent.fresh(), /is missing at/);

  const stray = await freshFixture(t);
  fs.writeFileSync(path.join(stray.out, stray.attempt.directory, 'images', 'stray-capture.png'), 'not selected');
  await assert.rejects(stray.fresh(), /unexpected capture file images\/stray-capture\.png/);
});

test('fresh rejects incomplete jobs, unexpected captures and failed operation results', async (t) => {
  const jobError = await freshFixture(t);
  retamper(jobError, (run) => { run.jobs[0].error = 'operation failed'; });
  await assert.rejects(jobError.fresh(), /operation failed/);

  const nullResult = await freshFixture(t);
  retamper(nullResult, (run) => { run.jobs[0].result = null; });
  await assert.rejects(nullResult.fresh(), /direct GPU operation result/);

  const shortResult = await freshFixture(t);
  retamper(shortResult, (run) => { run.jobs[0].result.images = []; });
  await assert.rejects(shortResult.fresh(), /returned 0 image|missing selected capture/);

  const duplicateJob = await freshFixture(t, [foundation, atmosphere]);
  retamper(duplicateJob, (run) => { run.jobs[1] = { ...run.jobs[0], jobId: 'job-copy' }; });
  await assert.rejects(duplicateJob.fresh(), /duplicate scenario\/mode/);

  const unexpectedJob = await freshFixture(t);
  retamper(unexpectedJob, (run) => { run.jobs.push({ ...run.jobs[0], jobId: 'job-extra', scenario: 'other' }); });
  await assert.rejects(unexpectedJob.fresh(), /job\(s\) for|unexpected scenario/);
});

test('fresh rejects software adapters, non-Windows browsers and console/page/GPU errors', async (t) => {
  const cases = [
    [(result) => { result.adapter.isFallbackAdapter = true; }, /physical GPU|fallback/],
    [(result) => { result.hardware = false; }, /physical GPU|hardware/],
    [(result) => { result.adapter.device = 'SwiftShader Device'; }, /software or fallback adapter/],
    [(result) => { result.browser.userAgent = result.browser.userAgent.replace('Windows NT 10.0', 'X11; Linux x86_64'); }, /Chrome on Windows/],
    [(result) => { result.browser.platform = 'Linux x86_64'; }, /Chrome on Windows/],
    [(result) => { result.browser.devicePixelRatio = 2; }, /device pixel ratio/],
    [(result) => { result.errors.console = ['uncaught error']; }, /console errors/],
    [(result) => { result.errors.page = ['page error']; }, /page errors/],
    [(result) => { result.errors.gpu = ['validation error']; }, /gpu errors/],
  ];
  for (const [mutate, expected] of cases) {
    const s = await freshFixture(t);
    retamper(s, (run) => mutate(run.jobs[0].result));
    await assert.rejects(s.fresh(), expected);
  }
});

test('fresh evaluates every requiredFacts selector against the scene registry', async (t) => {
  const falseAssertion = await freshFixture(t);
  retamper(falseAssertion, (run) => { run.jobs[0].result.facts.assertions[0].passed = false; });
  await assert.rejects(falseAssertion.fresh(), /assertion:sky-visible/);

  const missingAssertion = await freshFixture(t);
  retamper(missingAssertion, (run) => { run.jobs[0].result.facts.assertions = []; });
  await assert.rejects(missingAssertion.fresh(), /assertion:sky-visible/);

  const nullSky = await freshFixture(t);
  retamper(nullSky, (run) => { run.jobs[0].result.facts.sky = null; });
  await assert.rejects(nullSky.fresh(), /snapshot:sky/);

  const noPasses = await freshFixture(t);
  retamper(noPasses, (run) => { run.jobs[0].result.facts.passes = []; });
  await assert.rejects(noPasses.fresh(), /runtime:passes/);

  const notReady = await freshFixture(t);
  retamper(notReady, (run) => {
    run.jobs[0].result.facts.readiness = { status: 'pending', stage: 'foundation', error: null };
  });
  await assert.rejects(notReady.fresh(), /not ready|pending/);
});

test('fresh validates declared pair coverage and exact measurement records', async (t) => {
  const valid = await freshFixture(t, [paired]);
  assert.equal(valid.read(valid.records.run).jobs.length, 2);
  const record = await valid.fresh();
  assert.equal(record.status, 'passed');

  const missing = await freshFixture(t, [paired]);
  retamper(missing, (run) => { run.jobs[0].result.measurements = []; });
  await assert.rejects(missing.fresh(), /measurement record/);

  const unexpected = await freshFixture(t);
  retamper(unexpected, (run) => {
    run.jobs[0].result.measurements = [measurementRecord({ pair: 'cloud-shadows', state: 'on', preset: 'fair', features: {} })];
  });
  await assert.rejects(unexpected.fresh(), /expected 0 measurement record/);

  const corrupted = await freshFixture(t, [paired]);
  retamper(corrupted, (run) => { run.jobs[0].result.measurements[0].p95Ms = 99; });
  await assert.rejects(corrupted.fresh(), /measurement\.p95Ms|p95Ms/);

  const uncovered = await freshFixture(t, [{ ...paired, rows: [paired.rows[0]] }]);
  await assert.rejects(uncovered.fresh(), /not collected in both on and off states/);
});

function judgeStub(t, source) {
  const directory = scratch(t);
  const script = path.join(directory, 'judge.py');
  const calls = path.join(directory, 'calls.txt');
  fs.writeFileSync(script, source);
  const previous = { client: process.env.NMC_JUDGE_CLIENT, calls: process.env.JUDGE_CALLS };
  process.env.NMC_JUDGE_CLIENT = script;
  process.env.JUDGE_CALLS = calls;
  t.after(() => {
    if (previous.client === undefined) delete process.env.NMC_JUDGE_CLIENT;
    else process.env.NMC_JUDGE_CLIENT = previous.client;
    if (previous.calls === undefined) delete process.env.JUDGE_CALLS;
    else process.env.JUDGE_CALLS = previous.calls;
  });
  return calls;
}

const PASSING_JUDGE = [
  'import os, sys',
  'open(os.environ["JUDGE_CALLS"], "a").write(" ".join(sys.argv[1:]) + "\\n")',
  'print("JUDGE PASS: 8/10")',
].join('\n');

test('judge scores every selected capture without a reference image', async (t) => {
  const s = await freshFixture(t);
  const calls = judgeStub(t, PASSING_JUDGE);
  const record = await s.judge();
  assert.deepEqual(Object.keys(record).sort(), [...RECORD_FIELDS.judge].sort());
  assert.equal(record.status, 'passed');
  assert.deepEqual(record.errors, []);
  assert.equal(record.images.length, 1);
  const [image] = record.images;
  assert.equal(image.id, 'foundation-scene-image');
  assert.equal(image.filename, 'foundation-sky-and-sphere.png');
  assert.equal(image.description, 'The accepted sky state renders above the ground receiver.');
  assert.equal(image.minScore, 6);
  assert.equal(image.exit, 0);
  assert.equal(image.score, 8);
  assert.match(image.output, /JUDGE PASS: 8\/10/);
  assert.equal(new Date(image.judgedAt).toISOString(), image.judgedAt);
  assert.deepEqual(s.read(s.records.judge), record);
  assert.equal(s.read(s.records.fresh).status, 'passed');
  const lines = fs.readFileSync(calls, 'utf8').trim().split('\n');
  assert.equal(lines.length, 1);
  assert.match(lines[0], /--description The accepted sky state renders above the ground receiver\./);
  assert.match(lines[0], /--min-score 6/);
  assert.doesNotMatch(lines[0], /--reference/);
});

test('judge retains one failed image verdict among passes before failing', async (t) => {
  const s = await freshFixture(t, [foundation, atmosphere]);
  const calls = judgeStub(t, [
    'import os, sys',
    'open(os.environ["JUDGE_CALLS"], "a").write(" ".join(sys.argv[1:]) + "\\n")',
    'if os.path.basename(sys.argv[1]) == "atmosphere-clear-day.png":',
    '    print("JUDGE FAIL: 4/10")',
    '    sys.exit(1)',
    'print("JUDGE PASS: 9/10")',
  ].join('\n'));
  await assert.rejects(s.judge(), /image judgment failures/);
  const record = s.read(s.records.judge);
  assert.deepEqual(Object.keys(record).sort(), [...RECORD_FIELDS.judge].sort());
  assert.equal(record.status, 'failed');
  assert.equal(record.images.length, 2);
  const failed = record.images.find((image) => image.filename === 'atmosphere-clear-day.png');
  const passed = record.images.find((image) => image.filename === 'foundation-sky-and-sphere.png');
  assert.equal(failed.exit, 1);
  assert.equal(failed.score, 4);
  assert.match(failed.output, /JUDGE FAIL: 4\/10/);
  assert.equal(failed.minScore, 6);
  assert.equal(passed.exit, 0);
  assert.equal(passed.score, 9);
  assert.match(passed.output, /JUDGE PASS: 9\/10/);
  assert.equal(record.errors.length, 1);
  assert.match(record.errors[0], /atmosphere-clear-day\.png/);
  assert.equal(s.read(s.records.fresh).status, 'passed');
  assert.equal(fs.readFileSync(calls, 'utf8').trim().split('\n').length, 2);
});

test('judge refuses stale evidence and records the freshness failure', async (t) => {
  const s = await freshFixture(t);
  retamper(s, (run) => { run.jobs[0].result.facts.assertions[0].passed = false; });
  const calls = judgeStub(t, PASSING_JUDGE);
  await assert.rejects(s.judge(), /assertion:sky-visible/);
  const record = s.read(s.records.judge);
  assert.equal(record.status, 'failed');
  assert.deepEqual(record.images, []);
  assert.match(record.errors[0], /assertion:sky-visible/);
  assert.equal(fs.existsSync(calls), false);
});

// P2-RESUMABLE-RENDER. scenes.json sorts atmosphere-scene first, so group A renders before B.
const GROUP_A = atmosphere.id, GROUP_B = foundation.id;

async function cleanAgain(s) {
  s.attempt = await clean(s.page, s.out, s.stage);
  s.records = attemptRecords(s.attempt.directory);
  s.calls.length = 0;
  return s;
}

/** A new build of the current sources: new buildId and preparationId, then select and clean. */
async function rebuild(s) {
  const preparationId = randomUUID();
  const build = { ...s.build, buildId: randomUUID(), preparationId, inputs: captureInputs(s.root, preparationId),
    builtAt: new Date().toISOString() };
  writeJsonRecord(containedPath(s.page, 'build.json'), build);
  s.build = build;
  s.selection = await select(s.page, s.out, s.stage);
  return cleanAgain(s);
}

function imageBytes(s, directory, filename) {
  return fs.readFileSync(containedPath(s.out, `${directory}/images/${filename}`));
}

test('render after a failed run renders only the missing or failed groups and keeps the passed images', async (t) => {
  const s = await renderFixture(t, [foundation, atmosphere]);
  const first = s.attempt;
  const runOp = s.transport.runGpuPageOp;
  s.transport.runGpuPageOp = async (...args) => {
    if (args[2].scenario === GROUP_B) { s.calls.push(GROUP_B); throw new Error('operation failed'); }
    return runOp(...args);
  };
  await assert.rejects(s.render(), /operation failed/);
  s.transport.runGpuPageOp = runOp;
  await cleanAgain(s);
  const run = await s.render();
  assert.equal(run.status, 'passed');
  assert.deepEqual(s.calls, ['publish', GROUP_B]);
  assert.deepEqual(run.kept, [{ id: `${GROUP_A}-image`, filename: atmosphere.filename,
    attemptId: first.attemptId, selectionId: first.selectionId, buildId: first.buildId }]);
  assert.deepEqual(Object.keys(run), RECORD_FIELDS.run);
  assert.equal(run.jobs.length, 2);
  assert.deepEqual(s.read(s.records.result(0)), run.jobs[0].result);
  assert.deepEqual(imageBytes(s, s.attempt.directory, atmosphere.filename), imageBytes(s, first.directory, atmosphere.filename));
});

test('a rebuild with identical inputs keeps the images although buildId changed', async (t) => {
  const s = await renderFixture(t, [foundation, atmosphere]);
  const first = s.attempt;
  await s.render();
  await rebuild(s);
  assert.notEqual(s.build.buildId, first.buildId);
  const run = await s.render();
  assert.deepEqual(s.calls, []);
  assert.equal(run.buildId, s.build.buildId);
  assert.ok(run.kept.length === 2 && run.kept.every((entry) => entry.buildId === first.buildId));
  assert.ok(fs.readdirSync(containedPath(s.out, `${first.directory}/images`)).length === 2);
});

test('a changed source keeps nothing and clean removes only the older build images', async (t) => {
  const s = await renderFixture(t, [foundation, atmosphere]);
  const first = s.attempt;
  await s.render();
  put(s.root, 'examples/sky-weather/lib/inventory.js', 'export const criteria = ["changed"];\n');
  await rebuild(s);
  assert.deepEqual(fs.readdirSync(containedPath(s.out, `${first.directory}/images`)), []);
  for (const name of ['run', 'attempt', 'build', 'selection']) assert.ok(s.read(attemptRecords(first.directory)[name]));
  const run = await s.render();
  assert.deepEqual(run.kept, []);
  assert.equal(s.calls.filter((call) => call !== 'publish').length, 2);
  const second = s.attempt;
  await cleanAgain(s);
  assert.equal(fs.readdirSync(containedPath(s.out, `${second.directory}/images`)).length, 2);
});

test('render publishes nothing when every group is kept and reuses the pageRef of the attempt that supplied the first group', async (t) => {
  const s = await renderFixture(t, [foundation, atmosphere]);
  const firstRun = await s.render();
  const firstRecord = s.read(s.records.run);
  firstRecord.pageRef = 'first-attempt-page';
  writeJsonRecord(containedPath(s.out, s.records.run), firstRecord);
  await cleanAgain(s);
  const run = await s.render();
  assert.deepEqual(s.calls, []);
  assert.equal(run.status, 'passed');
  assert.equal(run.pageRef, 'first-attempt-page');
  assert.deepEqual(run.jobs, firstRun.jobs);
  assert.deepEqual(run.images, firstRun.images);
});

test('a kept candidate with page errors, a readiness that is not ready, a false required fact, a software adapter, an invalid measurement, a changed inventory row or a damaged PNG is rendered again', async (t) => {
  const defects = {
    page: { resultFor: (result) => ({ ...result, errors: { console: [], page: ['boom'], gpu: [] } }) },
    readiness: { factsFor: () => factsRecord({ readiness: { status: 'pending', stage: 'foundation', error: null } }) },
    fact: { factsFor: () => factsRecord({ assertions: [{ name: 'sky-visible', passed: false, detail: 'hidden' }] }) },
    software: { resultFor: (result) => ({ ...result, adapter: { ...result.adapter, description: 'SwiftShader' } }) },
    measurement: { resultFor: (result) => ({ ...result, measurements: [{}] }) },
    row: { after: (s) => {
      const selection = s.read(s.records.selection);
      selection.inventory[0].expectedVisibleBehavior = 'changed';
      writeJsonRecord(containedPath(s.out, s.records.selection), selection);
    } },
    png: { after: (s) => fs.writeFileSync(containedPath(s.out, s.records.image(foundation.filename)), 'damaged') },
  };
  for (const [name, defect] of Object.entries(defects)) {
    const s = await renderFixture(t, [foundation], defect);
    await s.render().catch(() => {});
    defect.after?.(s);
    s.factsFor = null; s.resultFor = null;
    await cleanAgain(s);
    const run = await s.render();
    assert.deepEqual(s.calls, ['publish', GROUP_B], name);
    assert.deepEqual(run.kept, [], name);
  }
});

test('a group kept twice keeps the identities of the attempt that rendered it', async (t) => {
  const s = await renderFixture(t, [foundation, atmosphere]);
  const first = s.attempt;
  const runOp = s.transport.runGpuPageOp;
  const failB = async (...args) => {
    if (args[2].scenario === GROUP_B) throw new Error('operation failed');
    return runOp(...args);
  };
  s.transport.runGpuPageOp = failB;
  await assert.rejects(s.render(), /operation failed/);
  await cleanAgain(s);
  await assert.rejects(s.render(), /operation failed/);
  s.transport.runGpuPageOp = runOp;
  await cleanAgain(s);
  const run = await s.render();
  assert.deepEqual(s.calls, ['publish', GROUP_B]);
  assert.equal(run.kept.length, 1);
  assert.equal(run.kept[0].attemptId, first.attemptId);
  assert.equal(run.jobs[0].result.provenance.attemptId, first.attemptId);
});

test('an attempt left running by an interrupted render still supplies its completed groups', async (t) => {
  const s = await renderFixture(t, [foundation, atmosphere]);
  const first = s.records;
  const runOp = s.transport.runGpuPageOp;
  let snapshot = null;
  s.transport.runGpuPageOp = async (...args) => {
    if (args[2].scenario !== GROUP_B) return runOp(...args);
    snapshot = { run: s.read(first.run), attempt: s.read(first.attempt) };
    throw new Error('interrupted');
  };
  await assert.rejects(s.render(), /interrupted/);
  writeJsonRecord(containedPath(s.out, first.run), snapshot.run);
  writeJsonRecord(containedPath(s.out, first.attempt), snapshot.attempt);
  assert.equal(s.read(first.attempt).status, 'running');
  s.transport.runGpuPageOp = runOp;
  await cleanAgain(s);
  const run = await s.render();
  assert.deepEqual(s.calls, ['publish', GROUP_B]);
  assert.equal(run.kept.length, 1);
});
