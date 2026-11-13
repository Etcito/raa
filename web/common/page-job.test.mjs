import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import childProcess from 'node:child_process';
import zlib from 'node:zlib';
import { randomUUID } from 'node:crypto';
import { containedPath, readJsonRecord, writeJsonRecord, captureInputs } from '../../examples/sky-weather/lib/evidence.mjs';
import { createPageJob, buildKey, attemptRecords } from './page-job.mjs';

delete process.env.NMC_CACHE_DIR;
delete process.env.TOY_RERENDER_ALL;
delete process.env.SKY_WEATHER_RERENDER_ALL;

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
const TOY_PNG = png(8, 8, (x, y) => [40 + y * 20, 90 + y * 15, 200 - x * 5]);

function scratch(t) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'page-job-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  return directory;
}

function put(root, name, text) {
  const file = path.join(root, name);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, text);
  return file;
}

/** Point the job at a fresh project store for the rest of the test. */
function projectStore(t) {
  const directory = scratch(t);
  process.env.NMC_CACHE_DIR = directory;
  t.after(() => {
    delete process.env.NMC_CACHE_DIR;
    delete process.env.TOY_RERENDER_ALL;
    delete process.env.SKY_WEATHER_RERENDER_ALL;
  });
  return directory;
}

const ONE = { id: 'toy-one-scene', milestone: 'one', file: 'scenes/one.mjs', filename: 'toy-one-scene.png' };
const TWO = { id: 'toy-two-scene', milestone: 'two', file: 'scenes/two.mjs', filename: 'toy-two-scene.png' };
const PAIR = {
  id: 'toy-pair-scene', milestone: 'one', file: 'scenes/pair.mjs',
  variants: [
    { id: 'on', measurement: { pair: 'toy-pair', state: 'on', preset: 'toy', features: { on: true } } },
    { id: 'off', measurement: { pair: 'toy-pair', state: 'off', preset: 'toy', features: { on: false } } },
  ],
  rows: [
    { id: 'toy-pair-on', mode: 'on', filename: 'toy-pair-on.png', pair: 'toy-pair' },
    { id: 'toy-pair-off', mode: 'off', filename: 'toy-pair-off.png', pair: 'toy-pair' },
  ],
};

/** The inventory the toy scene module exports; the job must return exactly these rows. */
function toyInventory(scene) {
  const rows = scene.rows ?? [{ id: `${scene.id}-image`, mode: 'default', filename: scene.filename }];
  return rows.map((row) => ({
    id: row.id,
    criterionName: row.criterionName ?? 'Toy page scene',
    scenario: scene.id,
    mode: row.mode,
    filename: row.filename,
    expectedVisibleBehavior: row.expectedVisibleBehavior ?? 'The toy page renders its fixture.',
    requiredFacts: row.requiredFacts ?? ['assertion:toy-visible'],
    pair: row.pair ?? null,
  }));
}

function sceneModule(scene) {
  const variants = scene.variants ?? [{ id: 'default' }];
  return `
export const factNames = ['toy-visible'];
export const inventory = ${JSON.stringify(toyInventory(scene))};
export default {
  id: ${JSON.stringify(scene.id)},
  milestone: ${JSON.stringify(scene.milestone)},
  title: ${JSON.stringify(scene.id)},
  variants: ${JSON.stringify(variants)},
};
`;
}

function fixture(t, options = {}) {
  const parent = scratch(t);
  const root = path.join(parent, 'raa');
  const page = path.join(root, 'dist/toy');
  const revision = '1'.repeat(40);
  const manifest = {
    'Cargo.toml': '[workspace]\n', 'Cargo.lock': 'version = 4\n',
    'rust-toolchain.toml': '[toolchain]\nchannel = "1.96.0"\n',
    '.nmc/commands.json': '[]\n',
    'examples/sky-weather/lib/inventory.js': 'export const criteria = [];\n',
  };
  for (const [name, text] of Object.entries(manifest)) put(root, name, text);
  put(root, '.git', 'gitdir: fixture\n');
  const scenes = options.scenes ?? [ONE];
  const state = { root, parent, page, revision, names: Object.keys(manifest), scenes: [], build: null, stage: 'one' };
  for (const scene of scenes) {
    const source = sceneModule(scene);
    state.scenes.push({ ...scene, source, inventory: toyInventory(scene) });
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
  const build = {
    buildId: randomUUID(), preparationId, inputs: captureInputs(root, preparationId),
    builtAt: new Date().toISOString(), page, files: ['scenes.json', ...state.scenes.map((scene) => scene.file)],
    status: 'ready', error: null,
  };
  if (options.build) options.build(build);
  writeJsonRecord(containedPath(page, 'build.json'), build);
  state.build = build;
  return state;
}

/** The toy configuration: every entry differs from the sky-weather configuration. */
function createToyJob(overrides = {}) {
  const stages = overrides.stages ?? ['one', 'two'];
  const validatorsCalls = [];
  const factCalls = [];
  const measurementCalls = [];
  const validators = {
    validateInventory(namespaces, stage) {
      const limit = stages.indexOf(stage);
      return namespaces
        .filter((namespace) => stages.indexOf(namespace.default.milestone) <= limit)
        .flatMap((namespace) => namespace.inventory);
    },
    validateRequiredFacts(row, facts, factNames) {
      factCalls.push({ row: row.id, facts, factNames });
      if (overrides.factError) throw new Error(overrides.factError);
    },
  };
  const job = createPageJob({
    name: overrides.name ?? 'toy-job',
    topic: overrides.topic ?? 'toy',
    stages,
    envPrefix: overrides.envPrefix ?? 'TOY',
    validators: async (page) => {
      validatorsCalls.push(page);
      return validators;
    },
    validateMeasurement: (measurement) => {
      measurementCalls.push(measurement);
      if (overrides.measureError) throw new Error(overrides.measureError);
    },
  });
  return { job, validators, validatorsCalls, factCalls, measurementCalls };
}

/** A fake managed transport returning the direct-operation result shape of the sky page. */
function fakeTransport(state) {
  const calls = { publish: 0, ops: [] };
  const transport = {
    async publishGpuPage(page, settings) {
      calls.publish += 1;
      assert.equal(page, state.page);
      assert.deepEqual(settings, { timeoutMs: 300000 });
      return { pageRef: 'toy-page-ref' };
    },
    async runGpuPageOp(pageRef, op, input, settings) {
      calls.ops.push(input.scenario);
      assert.equal(pageRef, 'toy-page-ref');
      assert.equal(op, 'run');
      assert.equal(settings.timeoutMs, 300000);
      assert.equal(input.remainingMs, 270000);
      assert.equal(input.width, 1280);
      assert.equal(input.height, 720);
      const { width, height, remainingMs, ...provenance } = input;
      const rows = state.selection.inventory.filter((row) => row.scenario === input.scenario && row.mode === input.mode);
      const scene = state.scenes.find((entry) => entry.id === input.scenario);
      const variant = (scene?.variants ?? [{ id: 'default' }]).find((entry) => entry.id === input.mode);
      assert.ok(variant, `unexpected variant ${input.mode}`);
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
        images: rows.map((row) => ({
          id: row.id, filename: row.filename, mime: 'image/png', data: TOY_PNG,
          observation: `Toy capture ${row.id}`,
        })),
        facts: { readiness: { status: 'ready', stage: state.stage, error: null } },
        measurements: variant.measurement ? [{
          ...variant.measurement, width: 1280, height: 720, warmup: 30, frames: 300,
          frameTimesMs: Array.from({ length: 300 }, () => 10), fps: 100, p50Ms: 10, p95Ms: 10, p99Ms: 10,
        }] : [],
        errors: { console: [], page: [], gpu: [] },
      };
      return { jobId: `toy-job-${calls.ops.length}`, clientId: 'windows-toy-client', result };
    },
  };
  return { transport, calls };
}

/** Select, clean and render one toy attempt; returns everything a stage test needs. */
async function renderToy(t, options = {}) {
  const toy = createToyJob(options.config ?? {});
  const state = fixture(t, { scenes: options.scenes ?? [ONE] });
  state.stage = options.stage ?? 'one';
  const out = path.join(state.root, 'out/toy', state.stage);
  const selection = await toy.job.select(state.page, out, state.stage);
  state.selection = selection;
  const attempt = await toy.job.clean(state.page, out, state.stage);
  const { transport, calls } = fakeTransport(state);
  const run = await toy.job.render(state.page, out, state.stage, transport);
  return { ...toy, state, out, stage: state.stage, selection, attempt, run, transport, calls };
}

test('configuration refusals name the invalid field', () => {
  const base = {
    name: 'toy-job', topic: 'toy', stages: ['one', 'two'], envPrefix: 'TOY',
    validators: () => ({ validateInventory() { return []; }, validateRequiredFacts() {} }),
    validateMeasurement() {},
  };
  const cases = [
    [{ name: '   ' }, 'name'],
    [{ topic: '' }, 'topic'],
    [{ envPrefix: '  ' }, 'envPrefix'],
    [{ stages: [] }, 'stages'],
    [{ stages: ['one', 'one'] }, 'stages'],
    [{ validators: 'not a function' }, 'validators'],
    [{ validateMeasurement: null }, 'validateMeasurement'],
  ];
  for (const [patch, field] of cases) {
    assert.throws(() => createPageJob({ ...base, ...patch }),
      (error) => error.message.startsWith(`page job config: ${field}: `));
  }
});

test('STAGES and USAGE are the configured ones and frozen', () => {
  const { job } = createToyJob();
  assert.deepEqual(job.STAGES, ['one', 'two']);
  assert.equal(Object.isFrozen(job.STAGES), true);
  assert.equal(job.USAGE, 'usage: node web/common/toy-job.mjs <check|select|clean|render|fresh|judge> '
    + '--page dist/toy --out out/toy/<stage> --stage <one|two>');
  assert.equal(Object.isFrozen(job.ACTIONS), true);
});

test('parseArgs accepts the full form and refuses an unknown action or stage', () => {
  const { job } = createToyJob();
  for (const action of ['check', 'select', 'clean', 'render', 'fresh', 'judge']) {
    assert.deepEqual(job.parseArgs([action, '--page', 'dist/toy', '--out', 'out/toy/one', '--stage', 'one']),
      { action, page: 'dist/toy', out: 'out/toy/one', stage: 'one' });
  }
  assert.throws(() => job.parseArgs(['deploy', '--page', 'dist/toy', '--stage', 'one']),
    (error) => error.message === job.USAGE);
  assert.throws(() => job.parseArgs(['check', '--stage', 'one']), (error) => error.message === job.USAGE);
  assert.throws(() => job.parseArgs(['check', '--page', 'dist/toy', '--stage', 'foundation']),
    /invalid stage: foundation/);
  assert.deepEqual(job.parseArgs(['check', '--page', 'dist/toy', '--stage', 'one']),
    { action: 'check', page: 'dist/toy', out: undefined, stage: 'one' });
  assert.throws(() => job.parseArgs(['render', '--page', 'dist/toy', '--out', 'out/toy/one',
    '--stage', 'one', '--budget-s', '30']), /--budget-s applies to judge only/);
  assert.deepEqual(job.parseArgs(['judge', '--page', 'dist/toy', '--out', 'out/toy/one',
    '--stage', 'one', '--budget-s', '30']),
  { action: 'judge', page: 'dist/toy', out: 'out/toy/one', stage: 'one', budgetMs: 30000 });
});

test('storeRoot follows NMC_CACHE_DIR and the resolved out otherwise', (t) => {
  const { job } = createToyJob();
  const cache = scratch(t);
  const out = path.join(scratch(t), 'out/toy/one');
  t.after(() => { delete process.env.NMC_CACHE_DIR; });
  delete process.env.NMC_CACHE_DIR;
  assert.equal(job.storeRoot(out, 'one'), path.resolve(out));
  process.env.NMC_CACHE_DIR = '   ';
  assert.equal(job.storeRoot(out, 'one'), path.resolve(out));
  process.env.NMC_CACHE_DIR = cache;
  assert.equal(job.storeRoot(out, 'one'), path.join(path.resolve(cache), 'toy', 'one'));
  assert.equal(job.storeRoot(out, 'two'), path.join(path.resolve(cache), 'toy', 'two'));
});

test('check awaits validators with the resolved page and returns the stage rows', async (t) => {
  const state = fixture(t, { scenes: [ONE, TWO] });
  const toy = createToyJob();
  const relative = path.relative(process.cwd(), state.page);
  const one = await toy.job.check(relative, 'one');
  assert.deepEqual(one, state.scenes.find((scene) => scene.id === ONE.id).inventory);
  assert.deepEqual(one.map((row) => row.id), ['toy-one-scene-image']);
  const two = await toy.job.check(state.page, 'two');
  assert.deepEqual(two.map((row) => row.id), ['toy-one-scene-image', 'toy-two-scene-image']);
  assert.equal(toy.validatorsCalls.length, 2);
  assert.equal(toy.validatorsCalls[0], state.page);
  assert.equal(toy.validatorsCalls[1], state.page);

  const unknown = fixture(t, { scenes: [{ ...ONE, milestone: 'foundation' }] });
  await assert.rejects(toy.job.check(unknown.page, 'one'), /unknown milestone foundation/);
});

test('select, clean and render write the stage records and save the passed job in the store', async (t) => {
  const cache = projectStore(t);
  const rendered = await renderToy(t);
  const { state, out, stage, selection, attempt, run } = rendered;
  const records = attemptRecords(attempt.directory);
  assert.equal(selection.stage, stage);
  assert.equal(selection.buildId, state.build.buildId);
  assert.deepEqual(readJsonRecord(containedPath(out, 'selection.json')), selection);
  assert.deepEqual(readJsonRecord(containedPath(out, 'current.json')),
    { selectionId: selection.selectionId, attemptId: attempt.attemptId, directory: attempt.directory });
  assert.deepEqual(readJsonRecord(containedPath(out, records.selection)), selection);
  assert.deepEqual(readJsonRecord(containedPath(out, records.build)), state.build);
  assert.equal(readJsonRecord(containedPath(out, records.attempt)).status, 'passed');
  assert.equal(run.status, 'passed');
  assert.equal(run.jobs.length, 1);
  assert.deepEqual(run.kept, []);
  assert.equal(run.images.length, 1);
  assert.equal(run.images[0].filename, ONE.filename);
  assert.equal(fs.existsSync(containedPath(out, records.image(ONE.filename))), true);
  assert.equal(fs.existsSync(containedPath(out, records.input(0))), true);
  assert.equal(fs.existsSync(containedPath(out, records.result(0))), true);
  const stored = path.join(cache, 'toy', stage, 'attempts', attempt.attemptId);
  assert.equal(fs.existsSync(path.join(stored, 'selection.json')), true);
  assert.equal(fs.existsSync(path.join(stored, 'build.json')), true);
  assert.equal(fs.existsSync(path.join(stored, 'images', ONE.filename)), true);
  assert.equal(readJsonRecord(path.join(stored, 'run.json')).jobs.length, 1);
});

test('an unchanged build keeps the image and the configured prefix gates re-rendering', async (t) => {
  projectStore(t);
  const first = await renderToy(t);
  assert.equal(first.calls.ops.length, 1);

  await first.job.clean(first.state.page, first.out, first.stage);
  const kept = await first.job.render(first.state.page, first.out, first.stage, {
    publishGpuPage: async () => { throw new Error('a kept render must not publish'); },
    runGpuPageOp: async () => { throw new Error('a kept render must not run a job'); },
  });
  assert.equal(kept.kept.length, 1);
  assert.equal(kept.kept[0].id, `${ONE.id}-image`);
  assert.equal(kept.kept[0].attemptId, first.attempt.attemptId);
  assert.equal(first.calls.ops.length, 1);

  process.env.TOY_RERENDER_ALL = '1';
  t.after(() => { delete process.env.TOY_RERENDER_ALL; });
  await first.job.clean(first.state.page, first.out, first.stage);
  const rerendered = await first.job.render(first.state.page, first.out, first.stage, first.transport);
  assert.deepEqual(rerendered.kept, []);
  assert.equal(first.calls.ops.length, 2);

  delete process.env.TOY_RERENDER_ALL;
  process.env.SKY_WEATHER_RERENDER_ALL = '1';
  t.after(() => { delete process.env.SKY_WEATHER_RERENDER_ALL; });
  await first.job.clean(first.state.page, first.out, first.stage);
  const prefixed = await first.job.render(first.state.page, first.out, first.stage, first.transport);
  assert.equal(prefixed.kept.length, 1);
  assert.equal(prefixed.kept[0].id, `${ONE.id}-image`);
  assert.equal(first.calls.ops.length, 2);
});

test('fresh runs the configured fact and measurement validators per row and record', async (t) => {
  const rendered = await renderToy(t, { scenes: [PAIR] });
  const { state, out, stage } = rendered;
  rendered.factCalls.length = 0;
  rendered.measurementCalls.length = 0;
  const record = await rendered.job.fresh(state.page, out, stage);
  assert.equal(record.status, 'passed');
  assert.deepEqual(rendered.factCalls.map((call) => call.row).sort(), ['toy-pair-off', 'toy-pair-on']);
  assert.equal(rendered.factCalls.every((call) => call.factNames.includes('toy-visible')), true);
  assert.deepEqual(rendered.measurementCalls.map((measurement) => `${measurement.pair}/${measurement.state}`).sort(),
    ['toy-pair/off', 'toy-pair/on']);
  const freshRecord = readJsonRecord(containedPath(out, attemptRecords(rendered.attempt.directory).fresh));
  assert.equal(freshRecord.status, 'passed');

  const factsFail = createToyJob({ factError: 'toy required facts refused' });
  await assert.rejects(factsFail.job.fresh(state.page, out, stage), /toy required facts refused/);

  const measureFail = createToyJob({ measureError: 'toy measurement refused' });
  await assert.rejects(measureFail.job.fresh(state.page, out, stage), /toy measurement refused/);
});

test('judge logs through the injected logger with the configured prefix', async (t) => {
  const rendered = await renderToy(t);
  const lines = [];
  const verdicts = [];
  const record = await rendered.job.judge(rendered.state.page, rendered.out, rendered.stage, {
    judge: async (capture, description) => {
      verdicts.push({ capture, description });
      return { exit: 0, score: 8, output: 'toy judge output' };
    },
    log: (line) => lines.push(line),
  });
  assert.equal(record.status, 'passed');
  assert.equal(verdicts.length, 1);
  assert.equal(verdicts[0].description, 'The toy page renders its fixture.');
  assert.equal(lines.length, 1);
  assert.match(lines[0], /^toy-job: judge one:/);
  assert.match(lines[0], /pass 8/);
  assert.equal(readJsonRecord(containedPath(rendered.out,
    attemptRecords(rendered.attempt.directory).judge)).images[0].score, 8);
});

test('two jobs keep their own stages, usage and store roots', (t) => {
  const first = createToyJob();
  const second = createToyJob({ name: 'other-job', topic: 'other', stages: ['alpha', 'beta'], envPrefix: 'OTHER' });
  assert.deepEqual(first.job.STAGES, ['one', 'two']);
  assert.deepEqual(second.job.STAGES, ['alpha', 'beta']);
  assert.equal(Object.isFrozen(second.job.STAGES), true);
  assert.equal(second.job.USAGE, 'usage: node web/common/other-job.mjs <check|select|clean|render|fresh|judge> '
    + '--page dist/other --out out/other/<stage> --stage <alpha|beta>');
  assert.equal(first.job.USAGE, 'usage: node web/common/toy-job.mjs <check|select|clean|render|fresh|judge> '
    + '--page dist/toy --out out/toy/<stage> --stage <one|two>');
  assert.throws(() => second.job.parseArgs(['check', '--page', 'dist/other', '--stage', 'one']),
    /invalid stage: one/);
  const cache = scratch(t);
  process.env.NMC_CACHE_DIR = cache;
  t.after(() => { delete process.env.NMC_CACHE_DIR; });
  const out = path.join(scratch(t), 'out');
  assert.equal(first.job.storeRoot(out, 'one'), path.join(cache, 'toy', 'one'));
  assert.equal(second.job.storeRoot(out, 'alpha'), path.join(cache, 'other', 'alpha'));
});

test('buildKey drops the preparation id and follows the source text', () => {
  const inputs = {
    repository: 'raa', revision: 'a'.repeat(40), preparationId: randomUUID(), dependencies: [],
    sources: [{ path: 'examples/toy.js', text: 'export const one = 1;\n' }],
    toolchain: { rustc: 'rustc 1.96.0', wasmBindgen: 'wasm-bindgen 0.2.129', node: 'v24.0.0' },
    commandsText: '[]', criteriaText: '[]',
  };
  const key = buildKey({ inputs });
  assert.equal(Object.hasOwn(key, 'preparationId'), false);
  const { preparationId: _preparationId, ...rest } = inputs;
  assert.deepEqual(key, rest);
  assert.notEqual(inputs.preparationId, '');
  const changed = buildKey({
    inputs: { ...inputs, sources: [{ path: 'examples/toy.js', text: 'export const one = 2;\n' }] },
  });
  assert.notDeepEqual(key, changed);
  assert.deepEqual(key, buildKey({ inputs: { ...inputs, preparationId: randomUUID() } }));
});
