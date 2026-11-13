import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import childProcess from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { containedPath, readJsonRecord, writeJsonRecord, captureInputs, assertCurrentInputs } from './evidence.mjs';

function scratch(t) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'sky-evidence-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  return directory;
}

function put(root, name, text) {
  const file = path.join(root, name);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, text);
  return file;
}

test('contained paths support page, stage, attempt, job and image roots without creating them', t => {
  const root = scratch(t);
  for (const name of [
    'dist/sky-weather/build.json', 'out/sky-weather/foundation/selection.json',
    'out/sky-weather/foundation/current.json', `out/sky-weather/foundation/attempts/${randomUUID()}/attempt.json`,
    `attempts/${randomUUID()}/jobs/0/input.json`, 'images/clear-day-warm-horizon-and-reflective-sphere.png',
  ]) assert.equal(containedPath(root, name), path.join(root, name));
  assert.deepEqual(fs.readdirSync(root), []);
  assert.equal(containedPath('.', 'examples/sky-weather'), path.join(process.cwd(), 'examples/sky-weather'));
  const relative = path.relative(process.cwd(), root);
  assert.throws(() => containedPath(relative, 'record.json'), /unsafe path/);
});

test('root, relative names and direct IO refuse traversal, absolute names and backslashes', t => {
  const root = scratch(t);
  for (const name of ['', '.', '..', '../escape', 'safe/../escape', '/tmp/escape',
    'C:/escape', 'C:escape', '\\\\server\\share', 'safe\\escape', 'a//b', './a', 'a/', 'a\0b']) {
    assert.throws(() => containedPath(root, name), /unsafe/, name);
  }
  for (const badRoot of ['', `${root}/..`, `${root}\\child`, 'C:/tmp', 'bad\0root']) {
    assert.throws(() => containedPath(badRoot, 'record.json'), /unsafe/);
  }
  for (const badFile of ['', `${root}/../escape.json`, `${root}/a\\b.json`]) {
    assert.throws(() => readJsonRecord(badFile), /unsafe/);
    assert.throws(() => writeJsonRecord(badFile, {}), /unsafe/);
  }
  const file = put(root, 'file', '{}');
  assert.throws(() => containedPath(file, 'child'), /not a directory/);
  assert.throws(() => containedPath(root, 'file/child'), /not a directory/);
});

test('symlink roots, ancestors and targets reject even inside the root or when dangling', t => {
  const root = scratch(t);
  const directory = path.join(root, 'real');
  fs.mkdirSync(directory);
  put(directory, 'record.json', '{"error":"preserved"}');
  fs.symlinkSync(directory, path.join(root, 'alias'));
  fs.symlinkSync(scratch(t), path.join(root, 'outside'));
  fs.symlinkSync(path.join(directory, 'record.json'), path.join(directory, 'alias.json'));
  fs.symlinkSync(path.join(root, 'missing'), path.join(root, 'dangling'));
  for (const [base, name] of [[root, 'alias/record.json'], [root, 'outside/record.json'], [root, 'dangling/child.json'],
    [path.join(root, 'alias'), 'missing/record.json'], [directory, 'alias.json'],
    [path.join(root, 'alias/nested'), 'record.json']]) {
    assert.throws(() => containedPath(base, name), /symlink/);
    assert.throws(() => readJsonRecord(path.join(base, name)), /symlink/);
    assert.throws(() => writeJsonRecord(path.join(base, name), {}), /symlink/);
  }
  assert.deepEqual(readJsonRecord(path.join(directory, 'record.json')), { error: 'preserved' });
});

test('stage updates leave build and earlier attempt copies intact', t => {
  const { root } = repository(t);
  const page = path.join(root, 'dist/sky-weather');
  const stage = path.join(root, 'out/sky-weather/foundation');
  const preparationId = randomUUID();
  const inputs = captureInputs(root, preparationId);
  const build = { buildId: randomUUID(), preparationId, inputs, builtAt: new Date().toISOString(),
    page, files: [], status: 'failed', error: 'actual compiler failure' };
  writeJsonRecord(containedPath(page, 'build.json'), build);
  const attempt = randomUUID();
  const selection = { selectionId: randomUUID(), stage: 'foundation', buildId: build.buildId,
    preparationId, inputs, inventory: [] };
  writeJsonRecord(containedPath(stage, `attempts/${attempt}/build.json`), build);
  writeJsonRecord(containedPath(stage, `attempts/${attempt}/selection.json`), selection);
  writeJsonRecord(containedPath(stage, 'selection.json'), selection);
  writeJsonRecord(containedPath(stage, 'current.json'), {
    selectionId: selection.selectionId, attemptId: attempt, directory: `attempts/${attempt}`,
  });
  writeJsonRecord(containedPath(stage, 'selection.json'), { ...selection, selectionId: randomUUID() });
  assert.deepEqual(readJsonRecord(containedPath(page, 'build.json')), build);
  assert.deepEqual(readJsonRecord(containedPath(stage, `attempts/${attempt}/build.json`)), build);
  assert.deepEqual(readJsonRecord(containedPath(stage, `attempts/${attempt}/selection.json`)), selection);
});

test('interrupted writes, sync and rename preserve prior diagnostics and remove temporary files', t => {
  const root = scratch(t);
  const file = containedPath(root, 'current.json');
  const previous = { status: 'failed', errors: ['originating GPU failure'] };
  writeJsonRecord(file, previous);
  for (const operation of ['writeFileSync', 'fsyncSync', 'renameSync']) {
    const original = fs[operation];
    const hook = t.mock.method(fs, operation, (...args) => {
      if (operation === 'writeFileSync') original(args[0], '{"partial":');
      if (operation === 'renameSync') {
        assert.equal(path.dirname(args[0]), root);
        assert.deepEqual(readJsonRecord(file), previous);
        assert.deepEqual(readJsonRecord(args[0]), { status: 'passed' });
      }
      throw new Error(`interrupted ${operation}`);
    });
    assert.throws(() => writeJsonRecord(file, { status: 'passed' }), new RegExp(`interrupted ${operation}`));
    hook.mock.restore();
    assert.deepEqual(readJsonRecord(file), previous);
    assert.deepEqual(fs.readdirSync(root), ['current.json']);
  }
  const circular = {}; circular.self = circular;
  assert.throws(() => writeJsonRecord(file, circular), /circular/i);
  assert.throws(() => writeJsonRecord(file, undefined), /JSON document/);
  assert.deepEqual(readJsonRecord(file), previous);
  put(root, 'broken.json', '{');
  assert.throws(() => readJsonRecord(path.join(root, 'broken.json')), error =>
    error.message.includes('broken.json') && error.cause instanceof SyntaxError);
  assert.throws(() => readJsonRecord(path.join(root, 'absent.json')), error => error.cause.code === 'ENOENT');
});

function repository(t, files = {
  'Cargo.toml': '[workspace]\n', 'Cargo.lock': 'version = 4\n',
  'rust-toolchain.toml': '[toolchain]\nchannel = "1.96.0"\n',
  '.nmc/commands.json': '[]\n', 'examples/sky-weather/lib/inventory.js': 'export const criteria = [];\n',
  'examples/sky-weather/scenes/day.js': 'export const inventory = [];\n',
  'crates/raa-sky/src/lib.rs': '// actual source\n', 'packages/raa/package.json': '{"name":"@etcito/raa"}\n',
}) {
  const parent = scratch(t);
  const root = path.join(parent, 'raa');
  for (const [name, text] of Object.entries(files)) put(root, name, text);
  const state = { root, parent, names: Object.keys(files), revision: '1'.repeat(40),
    dependencyRevision: '2'.repeat(40), targets: [], dirty: '', untracked: '', rustc: 'rustc 1.96.0', bindgen: 'wasm-bindgen 0.2.129' };
  t.mock.method(childProcess, 'execFileSync', (program, args, options) => {
    if (program === 'cargo') return JSON.stringify({ workspace_root: root,
      packages: [{ dependencies: state.targets.map(target => ({ path: target })) }] });
    if (program === 'wasm-bindgen') return state.bindgen;
    if (program === (process.env.RUSTC || 'rustc')) return state.rustc;
    assert.equal(program, 'git');
    const own = options.cwd === root;
    if (args.includes('--show-toplevel')) return own ? root : path.join(parent, 'linked');
    if (args.includes('HEAD')) return own ? state.revision : state.dependencyRevision;
    if (args[0] === 'status') return state.dirty;
    if (args[0] === 'ls-files') return own ? state.names.join('\0') : state.untracked;
    throw new Error(`unexpected command: ${args}`);
  });
  return state;
}

test('capture requires a preparation and retains the complete exact input receipt', t => {
  const fixture = repository(t);
  for (const id of [undefined, '', 'not-a-preparation']) {
    assert.throws(() => captureInputs(fixture.root, id), /preparationId/);
  }
  const preparationId = randomUUID();
  const receipt = captureInputs(fixture.root, preparationId);
  assert.deepEqual(Object.keys(receipt).sort(), ['repository', 'revision', 'preparationId',
    'dependencies', 'sources', 'toolchain', 'commandsText', 'criteriaText'].sort());
  assert.equal(receipt.repository, 'raa');
  assert.equal(receipt.revision, fixture.revision);
  assert.equal(receipt.preparationId, preparationId);
  assert.deepEqual(receipt.dependencies, []);
  assert.deepEqual(receipt.toolchain, { rustc: fixture.rustc, wasmBindgen: fixture.bindgen, node: process.version });
  assert.equal(receipt.commandsText, '[]\n');
  assert.deepEqual(receipt.sources.map(source => source.path), [...fixture.names].sort());
  for (const source of receipt.sources) {
    assert.equal(path.isAbsolute(source.path), false);
    assert.equal(source.text, fs.readFileSync(path.join(fixture.root, source.path), 'utf8'));
  }
  assert.deepEqual(JSON.parse(receipt.criteriaText).map(source => source.path), [
    'examples/sky-weather/lib/inventory.js', 'examples/sky-weather/scenes/day.js',
  ]);
  assertCurrentInputs(fixture.root, JSON.parse(JSON.stringify(receipt)));
  assert.throws(() => assertCurrentInputs(fixture.root, { ...receipt, dependencies: undefined }), /dependencies/);
  assert.throws(() => assertCurrentInputs(fixture.root, { ...receipt, sources: [...receipt.sources, receipt.sources[0]] }), /duplicate/);
});

const VIRTUAL_GEOMETRY_FILES = {
  'Cargo.toml': '[workspace]\n', 'Cargo.lock': 'version = 4\n',
  'rust-toolchain.toml': '[toolchain]\nchannel = "1.96.0"\n',
  '.nmc/commands.json': '[]\n',
  'examples/virtual-geometry/lib/inventory.js': 'export const rows = [];\n',
  'examples/virtual-geometry/scenes/bake.js': 'export default { id: "bake" };\n',
  'examples/virtual-geometry/nmc.gpu.json': '{"page":"index.html","ops":["run"]}\n',
};

test('a virtual-geometry fixture captures its own inventory criteria and sources', t => {
  const fixture = repository(t, VIRTUAL_GEOMETRY_FILES);
  const receipt = captureInputs(fixture.root, randomUUID(), 'virtual-geometry');
  assert.deepEqual(JSON.parse(receipt.criteriaText).map(source => source.path), [
    'examples/virtual-geometry/lib/inventory.js', 'examples/virtual-geometry/nmc.gpu.json',
    'examples/virtual-geometry/scenes/bake.js',
  ]);
  for (const name of Object.keys(VIRTUAL_GEOMETRY_FILES)) {
    assert.ok(receipt.sources.some(source => source.path === name), name);
  }
  assert.equal(receipt.sources.some(source => source.path.startsWith('examples/sky-weather/')), false);
  assertCurrentInputs(fixture.root, JSON.parse(JSON.stringify(receipt)));
});

test('a changed virtual-geometry scene invalidates the page receipt', t => {
  const fixture = repository(t, VIRTUAL_GEOMETRY_FILES);
  const receipt = captureInputs(fixture.root, randomUUID(), 'virtual-geometry');
  put(fixture.root, 'examples/virtual-geometry/scenes/bake.js', 'export default { id: "bake", rows: [] };\n');
  assert.throws(() => assertCurrentInputs(fixture.root, receipt),
    /changed input.*examples\/virtual-geometry\/scenes/);
});

test('the default page refuses a fixture without a sky-weather inventory', t => {
  const fixture = repository(t, VIRTUAL_GEOMETRY_FILES);
  assert.throws(() => captureInputs(fixture.root, randomUUID()),
    /missing sky-weather inventory criteria/);
});

test('an unsafe page name is refused before any input is read', t => {
  const fixture = repository(t);
  for (const page of ['', 'Virtual', '../escape', 'sky/weather', 'sky_weather']) {
    assert.throws(() => captureInputs(fixture.root, randomUUID(), page), /invalid page name/);
  }
});

test('source, manifests, locks, criteria, commands, toolchain and revision changes invalidate inputs', t => {
  const fixture = repository(t);
  const receipt = captureInputs(fixture.root, randomUUID());
  for (const source of receipt.sources) {
    const file = path.join(fixture.root, source.path);
    const stat = fs.statSync(file);
    // Valid JSON where needed, equal byte length and original timestamp still fail.
    const changed = source.path.endsWith('package.json') ? source.text.replace('raa', 'rbb')
      : source.path.endsWith('commands.json') ? '{}\n' : source.text.replace(/\S/, 'X');
    fs.writeFileSync(file, changed);
    fs.utimesSync(file, stat.atime, stat.mtime);
    assert.throws(() => assertCurrentInputs(fixture.root, receipt), /changed input/);
    fs.writeFileSync(file, source.text);
  }
  for (const key of ['rustc', 'bindgen', 'revision']) {
    const previous = fixture[key];
    fixture[key] = key === 'revision' ? '3'.repeat(40) : `${previous} changed`;
    assert.throws(() => assertCurrentInputs(fixture.root, receipt), /changed inputs/);
    fixture[key] = previous;
  }
  const addition = 'examples/sky-weather/scenes/new.js';
  put(fixture.root, addition, '// new scene\n');
  fixture.names.push(addition);
  assert.throws(() => assertCurrentInputs(fixture.root, receipt), /changed input.*new\.js/);
  fixture.names.pop();
  fs.unlinkSync(path.join(fixture.root, receipt.sources[0].path));
  assert.throws(() => assertCurrentInputs(fixture.root, receipt), /ENOENT/);
});

test('configured linked inputs require current, clean, isolated host preparations', t => {
  const fixture = repository(t);
  const linked = path.join(fixture.parent, 'linked');
  fs.mkdirSync(linked);
  fixture.targets = [linked, linked];
  const preparationId = randomUUID();
  const marker = `.nmc-built-${fixture.dependencyRevision}`;
  assert.throws(() => captureInputs(fixture.root, preparationId), /ENOENT.*\.nmc-built/);
  put(linked, marker, 'stale\n');
  assert.throws(() => captureInputs(fixture.root, preparationId), /stale preparation/);
  put(linked, marker, `${fixture.dependencyRevision}\n`);
  const receipt = captureInputs(fixture.root, preparationId);
  assert.deepEqual(receipt.dependencies, [{ repository: 'linked', revision: fixture.dependencyRevision, preparationId: marker }]);
  assertCurrentInputs(fixture.root, receipt);
  fixture.dirty = ' M src/lib.rs';
  assert.throws(() => assertCurrentInputs(fixture.root, receipt), /changed dependency/);
  fixture.dirty = '';
  fixture.untracked = 'src/new.rs';
  assert.throws(() => assertCurrentInputs(fixture.root, receipt), /unprepared dependency inputs/);
  fixture.untracked = '';
  fixture.dependencyRevision = '3'.repeat(40);
  put(linked, `.nmc-built-${fixture.dependencyRevision}`, fixture.dependencyRevision);
  assert.throws(() => assertCurrentInputs(fixture.root, receipt), /changed inputs: dependencies/);
  fixture.targets = [];
  assert.throws(() => assertCurrentInputs(fixture.root, receipt), /changed inputs: dependencies/);
});

test('input and linked preparation symlinks cannot be captured', t => {
  const fixture = repository(t);
  const source = 'crates/raa-sky/src/lib.rs';
  const original = fs.readFileSync(path.join(fixture.root, source), 'utf8');
  fs.unlinkSync(path.join(fixture.root, source));
  fs.symlinkSync(path.join(fixture.root, 'Cargo.toml'), path.join(fixture.root, source));
  assert.throws(() => captureInputs(fixture.root, randomUUID()), /symlink/);
  fs.unlinkSync(path.join(fixture.root, source));
  put(fixture.root, source, original);
  const linked = path.join(fixture.parent, 'linked');
  fs.mkdirSync(linked);
  fixture.targets = [linked];
  fs.symlinkSync(path.join(fixture.root, 'Cargo.lock'), path.join(linked, `.nmc-built-${fixture.dependencyRevision}`));
  assert.throws(() => captureInputs(fixture.root, randomUUID()), /symlink/);
});

test('cleanup failure retains the interrupted replacement as its cause', t => {
  const root = scratch(t);
  const file = put(root, 'failed.json', '{"error":"previous failure"}\n');
  const original = new Error('rename interrupted');
  const rename = t.mock.method(fs, 'renameSync', () => { throw original; });
  const unlink = t.mock.method(fs, 'unlinkSync', () => { throw new Error('cleanup unavailable'); });
  assert.throws(() => writeJsonRecord(file, { status: 'passed' }), error =>
    error instanceof AggregateError && error.cause === original
    && error.errors[1].message === 'cleanup unavailable');
  rename.mock.restore();
  unlink.mock.restore();
  assert.deepEqual(readJsonRecord(file), { error: 'previous failure' });
});
