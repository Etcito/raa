import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import childProcess from 'node:child_process';
import zlib from 'node:zlib';
import { randomUUID } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { containedPath, readJsonRecord, writeJsonRecord, captureInputs } from '../../examples/sky-weather/lib/evidence.mjs';
import { createPageJob, buildKey, attemptRecords, runCli, judgeCapture } from './page-job.mjs';

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
    ...(row.heldConstant === undefined ? {} : { heldConstant: row.heldConstant }),
    ...(row.judgeReference === undefined ? {} : { judgeReference: row.judgeReference }),
    ...(row.judgeAspect === undefined ? {} : { judgeAspect: row.judgeAspect }),
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
    report: overrides.report,
    keepSameBuildOnly: overrides.keepSameBuildOnly,
    judgeHeldConstant: overrides.judgeHeldConstant,
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

/** Select, clean and render again on the same stage; updates the selection the fake transport serves. */
async function rerenderToy(job, state, out, stage, transport) {
  const selection = await job.select(state.page, out, stage);
  state.selection = selection;
  await job.clean(state.page, out, stage);
  return job.render(state.page, out, stage, transport);
}

/** A new build of the page: a fresh buildId and preparationId, optionally after a source text edit. */
function rebuildPage(state, sourceText) {
  if (sourceText !== undefined) put(state.root, 'examples/sky-weather/lib/inventory.js', sourceText);
  const preparationId = randomUUID();
  const build = {
    buildId: randomUUID(), preparationId, inputs: captureInputs(state.root, preparationId),
    builtAt: new Date().toISOString(), page: state.page, files: ['scenes.json', ...state.scenes.map((scene) => scene.file)],
    status: 'ready', error: null,
  };
  writeJsonRecord(containedPath(state.page, 'build.json'), build);
  state.build = build;
  return build;
}

/** Rewrite one delivered scene module (and scenes.json) without touching the build inputs. */
function rewriteScene(state, sceneId, changes) {
  const scene = state.scenes.find((entry) => entry.id === sceneId);
  Object.assign(scene, changes);
  scene.inventory = toyInventory(scene);
  scene.source = sceneModule(scene);
  put(state.page, scene.file, scene.source);
  if (!state.build.files.includes(scene.file)) state.build.files.push(scene.file);
  writeJsonRecord(containedPath(state.page, 'build.json'), state.build);
  put(state.page, 'scenes.json', `${JSON.stringify(state.scenes.map(({ id, milestone, file }) => ({ id, milestone, file }))
    .sort((a, b) => a.id.localeCompare(b.id)), null, 2)}\n`);
}

/** A transport that refuses every call: used to prove a render kept everything. */
function refusingTransport() {
  return {
    async publishGpuPage() { throw new Error('a kept render must not publish'); },
    async runGpuPageOp() { throw new Error('a kept render must not run a job'); },
  };
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

test('the job exposes its read-only name', () => {
  const { job } = createToyJob();
  assert.equal(job.name, 'toy-job');
  assert.throws(() => { job.name = 'other'; }, TypeError);
  assert.equal(job.name, 'toy-job');
});

test('main prints the default report when no report is configured', async (t) => {
  const state = fixture(t, { scenes: [ONE, TWO] });
  const { job } = createToyJob();
  const written = [];
  t.mock.method(process.stdout, 'write', (chunk) => { written.push(String(chunk)); return true; });
  await job.main(['check', '--page', state.page, '--stage', 'one']);
  assert.deepEqual(written, ['toy-job: check one: 1 image(s)\n']);
  written.length = 0;
  await job.main(['check', '--page', state.page, '--stage', 'two']);
  assert.deepEqual(written, ['toy-job: check two: 2 image(s)\n']);
});

test('a configured report prints exactly its lines for check and select', async (t) => {
  const state = fixture(t, { scenes: [ONE] });
  const out = path.join(state.root, 'out/toy/one');
  const calls = [];
  const report = (action, stage, result, inventory) => {
    calls.push({ action, stage, result, inventory });
    return action === 'check'
      ? ['checked the toy page', 'and its fixture']
      : [`selected ${result.selectionId}`, `stage ${stage}`, `rows ${inventory.length}`];
  };
  const { job } = createToyJob({ report });
  const written = [];
  t.mock.method(process.stdout, 'write', (chunk) => { written.push(String(chunk)); return true; });

  await job.main(['check', '--page', state.page, '--stage', 'one']);
  assert.deepEqual(written, ['checked the toy page\n', 'and its fixture\n']);
  const checked = calls.at(-1);
  assert.equal(checked.action, 'check');
  assert.deepEqual(checked.result, checked.inventory);
  assert.deepEqual(checked.inventory.map((row) => row.id), ['toy-one-scene-image']);

  written.length = 0;
  await job.main(['select', '--page', state.page, '--out', out, '--stage', 'one']);
  const selection = readJsonRecord(containedPath(out, 'selection.json'));
  assert.deepEqual(written, [`selected ${selection.selectionId}\n`, 'stage one\n', 'rows 1\n']);
  const selected = calls.at(-1);
  assert.equal(selected.action, 'select');
  assert.deepEqual(selected.result, selection);
  assert.deepEqual(selected.inventory, selection.inventory);
  assert.deepEqual(selected.inventory.map((row) => row.id), ['toy-one-scene-image']);
});

test('a present non-function report is refused at creation', () => {
  const base = {
    name: 'toy-job', topic: 'toy', stages: ['one', 'two'], envPrefix: 'TOY',
    validators: () => ({ validateInventory() { return []; }, validateRequiredFacts() {} }),
    validateMeasurement() {},
  };
  for (const report of ['not a function', null, ['lines']]) {
    assert.throws(() => createPageJob({ ...base, report }),
      (error) => error.message.startsWith('page job config: report: '));
  }
});

test('a report that does not return an array of lines rejects main', async (t) => {
  const state = fixture(t, { scenes: [ONE] });
  t.mock.method(process.stdout, 'write', () => true);
  for (const lines of ['one line', ['a line', 7], {}]) {
    const { job } = createToyJob({ report: () => lines });
    await assert.rejects(job.main(['check', '--page', state.page, '--stage', 'one']),
      (error) => error.message === 'toy-job: report must return an array of lines');
  }
});

test('runCli acts only when the metaUrl is the entry point', async (t) => {
  const entry = path.join(scratch(t), 'toy-job.mjs');
  fs.writeFileSync(entry, '#!/usr/bin/env node\n');
  const original = process.argv[1];
  t.after(() => { process.argv[1] = original; });

  let called = 0;
  const job = { name: 'toy-job', main: async () => { called += 1; } };
  process.argv[1] = entry;
  const matching = pathToFileURL(path.resolve(entry)).href;
  runCli(job, pathToFileURL(path.join(path.dirname(entry), 'other-job.mjs')).href);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(called, 0);

  runCli(job, matching);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(called, 1);
});

const HELD = {
  id: 'toy-held-scene', milestone: 'one', file: 'scenes/held.mjs',
  rows: [{
    id: 'toy-held-image', mode: 'default', filename: 'toy-held-scene.png',
    heldConstant: ['fixture-a', 'fixture-b'],
  }],
};

test('keepSameBuildOnly and judgeHeldConstant are refused unless boolean', () => {
  const base = {
    name: 'toy-job', topic: 'toy', stages: ['one', 'two'], envPrefix: 'TOY',
    validators: () => ({ validateInventory() { return []; }, validateRequiredFacts() {} }),
    validateMeasurement() {},
  };
  for (const field of ['keepSameBuildOnly', 'judgeHeldConstant']) {
    for (const value of ['true', 1, 0, null, {}, []]) {
      assert.throws(() => createPageJob({ ...base, [field]: value }),
        (error) => error.message === `page job config: ${field}: must be a boolean`);
    }
  }
  assert.doesNotThrow(() => createPageJob({ ...base, keepSameBuildOnly: true, judgeHeldConstant: false }));
});

test('keepSameBuildOnly keeps an image only for an unchanged build key', async (t) => {
  projectStore(t);
  const toy = createToyJob({ keepSameBuildOnly: true });
  const state = fixture(t, { scenes: [ONE] });
  const stage = 'one';
  const out = path.join(state.root, 'out/toy', stage);
  const { transport, calls } = fakeTransport(state);
  const first = await rerenderToy(toy.job, state, out, stage, transport);
  assert.equal(calls.ops.length, 1);

  // A new build of the same sources has the same build key and keeps the image.
  rebuildPage(state);
  const kept = await rerenderToy(toy.job, state, out, stage, refusingTransport());
  assert.equal(kept.kept.length, 1);
  assert.equal(kept.kept[0].attemptId, first.attemptId);
  assert.equal(calls.ops.length, 1);

  // A new build whose source text differs is rendered again.
  rebuildPage(state, 'export const criteria = ["changed"];\n');
  const changed = await rerenderToy(toy.job, state, out, stage, transport);
  assert.deepEqual(changed.kept, []);
  assert.equal(calls.ops.length, 2);

  // A changed inventory row filename is rendered again even under the same build.
  rewriteScene(state, ONE.id, { filename: 'toy-one-scene-renamed.png', file: 'scenes/one-renamed.mjs' });
  const renamed = await rerenderToy(toy.job, state, out, stage, transport);
  assert.deepEqual(renamed.kept, []);
  assert.equal(calls.ops.length, 3);
});

test('keepSameBuildOnly false keeps an image across a changed build', async (t) => {
  projectStore(t);
  const toy = createToyJob({ keepSameBuildOnly: false });
  const state = fixture(t, { scenes: [ONE] });
  const stage = 'one';
  const out = path.join(state.root, 'out/toy', stage);
  const { transport, calls } = fakeTransport(state);
  const first = await rerenderToy(toy.job, state, out, stage, transport);
  rebuildPage(state, 'export const criteria = ["changed"];\n');
  const kept = await rerenderToy(toy.job, state, out, stage, refusingTransport());
  assert.equal(kept.kept.length, 1);
  assert.equal(kept.kept[0].attemptId, first.attemptId);
  assert.equal(calls.ops.length, 1);
});

test('fresh refuses a kept origin whose frozen build has another build key', async (t) => {
  projectStore(t);
  const rendered = await renderToy(t, { config: { keepSameBuildOnly: false } });
  const { state, out, stage } = rendered;
  rebuildPage(state, 'export const criteria = ["changed"];\n');
  const kept = await rerenderToy(rendered.job, state, out, stage, rendered.transport);
  assert.equal(kept.kept.length, 1);
  const originId = kept.kept[0].attemptId;
  const strict = createToyJob({ keepSameBuildOnly: true }).job;
  const store = strict.storeRoot(out, stage);
  for (const root of [out, store]) {
    const file = path.join(root, 'attempts', originId, 'build.json');
    if (!fs.existsSync(file)) continue;
    const build = readJsonRecord(file);
    build.inputs.revision = '2'.repeat(40);
    writeJsonRecord(file, build);
  }
  await assert.rejects(strict.fresh(state.page, out, stage),
    new RegExp(`kept origin attempt ${originId} was built from other sources`));
});

test('judgeHeldConstant passes the row constants to the judge', async (t) => {
  const withOption = await renderToy(t, { config: { judgeHeldConstant: true }, scenes: [HELD] });
  const seen = [];
  const record = await withOption.job.judge(withOption.state.page, withOption.out, withOption.stage, {
    judge: async (capture, description, options) => { seen.push({ capture, description, options }); return { exit: 0, score: 7, output: '' }; },
    log: () => {},
  });
  assert.equal(record.status, 'passed');
  assert.deepEqual(Object.keys(seen[0].options).sort(), ['heldConstant', 'killAt']);
  assert.deepEqual(seen[0].options.heldConstant, ['fixture-a', 'fixture-b']);
  assert.equal(seen[0].description, 'The toy page renders its fixture.');

  const withoutOption = await renderToy(t, { scenes: [HELD] });
  const plain = [];
  await withoutOption.job.judge(withoutOption.state.page, withoutOption.out, withoutOption.stage, {
    judge: async (capture, description, options) => { plain.push(options); return { exit: 0, score: 7, output: '' }; },
    log: () => {},
  });
  assert.deepEqual(Object.keys(plain[0]), ['killAt']);
});

test('judgeCapture places the held constants between the description and the minimum score', async (t) => {
  const directory = scratch(t);
  const client = path.join(directory, 'toy-judge.py');
  fs.writeFileSync(client, [
    'import json, sys',
    'print(json.dumps(sys.argv[1:]))',
    'print("JUDGE PASS: 8/10")',
    '',
  ].join('\n'));
  const capture = path.join(directory, 'toy-capture.png');
  fs.writeFileSync(capture, 'toy capture');
  const original = process.env.NMC_JUDGE_CLIENT;
  t.after(() => {
    if (original === undefined) delete process.env.NMC_JUDGE_CLIENT;
    else process.env.NMC_JUDGE_CLIENT = original;
  });
  process.env.NMC_JUDGE_CLIENT = client;

  const held = await judgeCapture(capture, 'the toy shows its fixture', { heldConstant: ['a', 'b'] });
  assert.equal(held.exit, 0);
  assert.equal(held.score, 8);
  const heldArgs = JSON.parse(held.output.split('\n')[0]);
  assert.equal(heldArgs[0], capture);
  const heldDescription = heldArgs.indexOf('--description');
  const heldMinimum = heldArgs.indexOf('--min-score');
  assert.ok(heldDescription >= 0 && heldMinimum > heldDescription);
  assert.deepEqual(heldArgs.slice(heldDescription + 2, heldMinimum),
    ['--held-constant', 'a', '--held-constant', 'b']);

  const plain = await judgeCapture(capture, 'the toy shows its fixture');
  const plainArgs = JSON.parse(plain.output.split('\n')[0]);
  const plainDescription = plainArgs.indexOf('--description');
  const plainMinimum = plainArgs.indexOf('--min-score');
  assert.deepEqual(plainArgs.slice(plainDescription + 2, plainMinimum), []);
  assert.equal(plainArgs.includes('--held-constant'), false);

  assert.throws(() => judgeCapture(capture, 'the toy shows its fixture', { heldConstant: 'a' }),
    (error) => error.message === 'judge: heldConstant must be an array of fixture names');
  assert.throws(() => judgeCapture(capture, 'the toy shows its fixture', { heldConstant: ['a', '  '] }),
    (error) => error.message === 'judge: heldConstant must be an array of fixture names');
});

/** A judge client that records its argv, and touches `started.txt` beside itself when it runs. */
function recordingJudgeClient(directory) {
  const client = path.join(directory, 'toy-judge.py');
  fs.writeFileSync(client, [
    'import json, os, sys',
    "open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'started.txt'), 'w').close()",
    'print(json.dumps(sys.argv[1:]))',
    'print("JUDGE PASS: 8/10")',
    '',
  ].join('\n'));
  return client;
}

function withJudgeClient(t, client) {
  const original = process.env.NMC_JUDGE_CLIENT;
  t.after(() => {
    if (original === undefined) delete process.env.NMC_JUDGE_CLIENT;
    else process.env.NMC_JUDGE_CLIENT = original;
  });
  process.env.NMC_JUDGE_CLIENT = client;
}

test('judgeCapture adds --reference and --aspect before the minimum score only for a row with a reference', async (t) => {
  const directory = scratch(t);
  const client = recordingJudgeClient(directory);
  withJudgeClient(t, client);
  const capture = path.join(directory, 'toy-capture.png');
  fs.writeFileSync(capture, 'toy capture');
  const reference = path.join(directory, 'toy-reference.png');
  fs.writeFileSync(reference, 'toy reference');

  const judged = await judgeCapture(capture, 'the toy shows its fixture', { reference, aspect: 'look' });
  assert.equal(judged.exit, 0);
  const args = JSON.parse(judged.output.split('\n')[0]);
  const description = args.indexOf('--description');
  const minimum = args.indexOf('--min-score');
  assert.deepEqual(args.slice(description + 2, minimum), ['--reference', reference, '--aspect', 'look']);

  const plain = await judgeCapture(capture, 'the toy shows its fixture');
  const plainArgs = JSON.parse(plain.output.split('\n')[0]);
  assert.equal(plainArgs.includes('--reference'), false);
  assert.equal(plainArgs.includes('--aspect'), false);
  assert.deepEqual(plainArgs.slice(plainArgs.indexOf('--description') + 2, plainArgs.indexOf('--min-score')), []);

  const missing = path.join(directory, 'absent.png');
  fs.rmSync(path.join(directory, 'started.txt'), { force: true });
  assert.throws(() => judgeCapture(capture, 'the toy shows its fixture', { reference: missing, aspect: 'look' }),
    (error) => error.message === `judge: reference ${missing} is missing`);
  assert.throws(() => judgeCapture(capture, 'the toy shows its fixture', { reference, aspect: 'mood' }),
    (error) => error.message === 'judge: aspect must be look, functional or design');
  assert.throws(() => judgeCapture(capture, 'the toy shows its fixture', { reference: 'relative.png' }),
    (error) => error.message === 'judge: reference must be an absolute .png path');
  assert.equal(fs.existsSync(path.join(directory, 'started.txt')), false);
});

test('judge passes a row reference and aspect to the judge and records both in its verdict', async (t) => {
  const directory = scratch(t);
  const reference = path.join(directory, 'toy-reference.png');
  fs.writeFileSync(reference, 'toy reference');
  const rendered = await renderToy(t, {
    scenes: [{ ...ONE, rows: [{ id: `${ONE.id}-image`, mode: 'default', filename: ONE.filename, judgeReference: reference, judgeAspect: 'look' }] }],
  });
  const seen = [];
  const record = await rendered.job.judge(rendered.state.page, rendered.out, rendered.stage, {
    judge: async (capture, description, options) => { seen.push(options); return { exit: 0, score: 7, output: '' }; },
    log: () => {},
  });
  assert.equal(record.status, 'passed');
  assert.deepEqual(Object.keys(seen[0]).sort(), ['aspect', 'killAt', 'reference']);
  assert.equal(seen[0].reference, reference);
  assert.equal(seen[0].aspect, 'look');
  const saved = readJsonRecord(containedPath(rendered.out, attemptRecords(rendered.attempt.directory).judge)).images[0];
  assert.equal(saved.reference, reference);
  assert.equal(saved.aspect, 'look');
});

test('judge fails a row whose reference is missing and starts no judge process', async (t) => {
  const directory = scratch(t);
  const client = recordingJudgeClient(directory);
  withJudgeClient(t, client);
  const reference = path.join(directory, 'absent.png');
  const rendered = await renderToy(t, {
    scenes: [{ ...ONE, rows: [{ id: `${ONE.id}-image`, mode: 'default', filename: ONE.filename, judgeReference: reference, judgeAspect: 'look' }] }],
  });
  await assert.rejects(rendered.job.judge(rendered.state.page, rendered.out, rendered.stage, { log: () => {} }));
  const saved = readJsonRecord(containedPath(rendered.out, attemptRecords(rendered.attempt.directory).judge));
  assert.equal(saved.status, 'failed');
  assert.ok(saved.errors.some((error) => error === `${ONE.filename}: judge: reference ${reference} is missing`), saved.errors.join('; '));
  assert.equal(fs.existsSync(path.join(directory, 'started.txt')), false);
});

test('judge reuses an earlier verdict only when the reference and aspect match', async (t) => {
  const directory = scratch(t);
  const reference = path.join(directory, 'toy-reference.png');
  fs.writeFileSync(reference, 'toy reference');
  const rendered = await renderToy(t);
  const calls = [];
  const runner = async (capture, description, options) => { calls.push(options); return { exit: 0, score: 8, output: '' }; };
  const judgeNow = () => rendered.job.judge(rendered.state.page, rendered.out, rendered.stage, { judge: runner, log: () => {} });

  assert.equal((await judgeNow()).status, 'passed');
  assert.equal(calls.length, 1);

  // The same image judged against a reference: the earlier verdict without one is refused.
  // A new file name: the page loader caches a module by its path (as the renamed-scene test does).
  rewriteScene(rendered.state, ONE.id, {
    file: 'scenes/one-reference.mjs',
    rows: [{ id: `${ONE.id}-image`, mode: 'default', filename: ONE.filename, judgeReference: reference, judgeAspect: 'look' }],
  });
  await rerenderToy(rendered.job, rendered.state, rendered.out, rendered.stage, rendered.transport);
  assert.equal((await judgeNow()).status, 'passed');
  assert.equal(calls.length, 2);
  assert.equal(calls[1].reference, reference);

  // The same reference and aspect reuse that verdict.
  await rerenderToy(rendered.job, rendered.state, rendered.out, rendered.stage, rendered.transport);
  assert.equal((await judgeNow()).status, 'passed');
  assert.equal(calls.length, 2);
});
