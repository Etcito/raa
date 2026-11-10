import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import childProcess from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { containedPath, readJsonRecord, writeJsonRecord, captureInputs } from '../../examples/sky-weather/lib/evidence.mjs';
import { check, select, clean, parseArgs, main, attemptRecords, RECORD_FIELDS } from './sky-weather-job.mjs';

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

test('CLI parses the protocol and drives select without inventing actions yet', async (t) => {
  const state = fixture(t);
  const out = path.join(state.root, 'out/sky-weather/foundation');
  assert.throws(() => parseArgs(['select', '--page', state.page, '--out', out]), /invalid stage/);
  assert.throws(() => parseArgs(['render', '--page', state.page, '--out', out, '--stage', 'foundation']), /usage/);
  assert.throws(() => parseArgs(['select', '--page', state.page, '--stage', 'foundation']), /usage/);
  assert.deepEqual(parseArgs(['check', '--page', state.page, '--stage', 'foundation']),
    { action: 'check', page: state.page, out: undefined, stage: 'foundation' });
  t.mock.method(process.stdout, 'write', () => true);
  await main(['select', '--page', state.page, '--out', out, '--stage', 'foundation']);
  assert.equal(readJsonRecord(containedPath(out, 'selection.json')).stage, 'foundation');
});
