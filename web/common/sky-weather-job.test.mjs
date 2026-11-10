import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import childProcess from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { containedPath, readJsonRecord, writeJsonRecord, captureInputs } from '../../examples/sky-weather/lib/evidence.mjs';
import { check, select, clean, render, parseArgs, main, attemptRecords, RECORD_FIELDS } from './sky-weather-job.mjs';

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

function sceneModule(scene) {
  const factNames = scene.factNames ?? ['sky-visible'];
  const requiredFacts = scene.requiredFacts ?? ['snapshot:sky', 'runtime:passes', 'runtime:viewSizes', 'runtime:ready', 'assertion:sky-visible'];
  return `
export const factNames = ${JSON.stringify(factNames)};
export const inventory = [{
  id: '${scene.id}-image',
  criterionName: 'Sky weather scene',
  scenario: '${scene.id}',
  mode: 'default',
  filename: '${scene.filename}',
  expectedVisibleBehavior: 'The accepted sky state renders above the ground receiver.',
  requiredFacts: ${JSON.stringify(requiredFacts)},
  pair: null,
}];
export async function observe() { return { sky: null, time: null, weather: null, diagnostics: null, assertions: [], edits: [] }; }
export default {
  id: '${scene.id}',
  milestone: '${scene.milestone}',
  title: '${scene.id}',
  build() {},
  variants: [{
    id: 'default',
    description: 'default state',
    selection: {
      camera: { eye: [0, 1, 5], target: [0, 1, 0], fovYDeg: 60, near: 0.1, far: 100 },
      subjects: [{ name: 'sphere', min: [-1, 0, -1], max: [1, 2, 1] }],
    },
  }],
};
`;
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
  assert.throws(() => parseArgs(['fresh', '--page', state.page, '--out', out, '--stage', 'foundation']), /usage/);
  assert.throws(() => parseArgs(['select', '--page', state.page, '--stage', 'foundation']), /usage/);
  assert.deepEqual(parseArgs(['check', '--page', state.page, '--stage', 'foundation']),
    { action: 'check', page: state.page, out: undefined, stage: 'foundation' });
  t.mock.method(process.stdout, 'write', () => true);
  await main(['select', '--page', state.page, '--out', out, '--stage', 'foundation']);
  assert.equal(readJsonRecord(containedPath(out, 'selection.json')).stage, 'foundation');
});

async function renderFixture(t, scenes = [foundation]) {
  const state = fixture(t, { scenes });
  state.out = path.join(state.root, 'out/sky-weather', scenes.length > 1 ? 'atmosphere' : 'foundation');
  state.stage = scenes.length > 1 ? 'atmosphere' : 'foundation';
  state.selection = await select(state.page, state.out, state.stage);
  state.attempt = await clean(state.page, state.out, state.stage);
  state.records = attemptRecords(state.attempt.directory);
  state.read = (name) => readJsonRecord(containedPath(state.out, name));
  state.calls = [];
  state.transport = {
    async publishGpuPage(page, options) {
      state.calls.push('publish');
      assert.equal(page, state.page);
      assert.deepEqual(options, { timeoutMs: 300000 });
      return { pageRef: 'host-page' };
    },
    async runGpuPageOp(pageRef, op, input, options) {
      state.calls.push(input.scenario);
      assert.equal(pageRef, 'host-page');
      assert.equal(op, 'run');
      assert.equal(options.timeoutMs, 300000);
      assert.equal(input.remainingMs, 270000);
      assert.equal(input.width, 1280);
      assert.equal(input.height, 720);
      const { width, height, remainingMs, ...provenance } = input;
      return { jobId: `job-${state.calls.length}`, clientId: 'windows-client', result: {
        provenance, json: { legacy: 'must not unwrap' }, nested: [1, { values: [false, null] }],
        images: state.selection.inventory.filter((row) => row.scenario === input.scenario && row.mode === input.mode)
          .map((row) => ({ id: row.id, filename: row.filename, mime: 'image/png',
            data: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jf1sAAAAASUVORK5CYII=',
            observation: 'Actual capture' })),
        errors: { console: [], page: [], gpu: [] },
      } };
    },
  };
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
