// Non-GPU coverage: the client is a fixture, never a render or acceptance record.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { runInNewContext } from 'node:vm';
import test from 'node:test';
import { DET_CASES, parseArgs, USAGE } from './anim-capture.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const script = join(root, 'scripts/anim-capture.mjs');

test('CLI lists both cases, accepts clip-sample forms, and preserves det arguments', () => {
  assert.equal(parseArgs(['--case', 'det-selftest']), 'det-selftest');
  assert.equal(parseArgs(['clip-sample']), 'clip-sample');
  assert.equal(parseArgs(['--case', 'clip-sample']), 'clip-sample');
  for (const args of [[], ['det-selftest'], ['--case'], ['other'], ['clip-sample', '--bad']]) {
    assert.equal(parseArgs(args), null);
  }
  assert.match(USAGE, /--case det-selftest/);
  assert.match(USAGE, /clip-sample/);
  const child = spawnSync(process.execPath, [script, '--bad'], { encoding: 'utf8', timeout: 5000 });
  assert.equal(child.status, 2);
  assert.equal(child.stdout, '');
  assert.equal(child.stderr, USAGE + '\n');
});

function fixture(t) {
  const dir = mkdtempSync(join(tmpdir(), 'raa-anim-capture-test-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const put = (name, text) => {
    mkdirSync(dirname(join(dir, name)), { recursive: true });
    writeFileSync(join(dir, name), text);
  };
  put('scripts/anim-capture.mjs', readFileSync(script));
  for (const source of ['examples/anim-brick', 'web/common']) {
    mkdirSync(dirname(join(dir, source)), { recursive: true });
    cpSync(join(root, source), join(dir, source), { recursive: true });
  }
  put('dist/bricks/raa-anim/index.js', '// fixture only');
  put('dist/bricks/raa-anim/pkg/raa_anim_js_bg.wasm', 'fixture only');
  put('stub-client.mjs', `
import assert from 'node:assert/strict';
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
const args = process.argv.slice(2);
appendFileSync('calls.jsonl', JSON.stringify(args) + '\\n');
if (args[0] === 'publish') {
  const files = args[1].endsWith('probe') ? ['index.html', 'nmc.gpu.json']
    : ['index.html', 'main.js', 'common/nmc-gpu.js', 'raa-anim/index.js',
      'raa-anim/pkg/raa_anim_js_bg.wasm', 'nmc.gpu.json'];
  for (const file of files) assert.ok(existsSync(join(args[1], file)));
  console.log(JSON.stringify({page_ref: 'fixture-page'}));
} else {
  assert.equal(args[0], 'render');
  assert.deepEqual(args.slice(2, 4), ['--page-ref', 'fixture-page']);
  const input = JSON.parse(readFileSync(args[1], 'utf8'));
  mkdirSync(args[5], {recursive: true});
  if (!input.args) {
    writeFileSync(join(args[5], 'result.json'), JSON.stringify({result:{json:{
      ok: process.env.STUB_FAILURE === 'probe' ? 0 : 1
    }}}));
    process.exit(0);
  }
  const request = input.args;
  assert.equal(request.case, 'clip-sample');
  const adapter = {vendor:'fixture', architecture:'test', device:'test', description:'test',
    isFallbackAdapter: process.env.STUB_FAILURE === 'software'};
  const result = {case: request.case, seed: request.seeds[0], ticks: request.ticks, hash:'0123abcd'};
  if (process.env.STUB_FAILURE === 'record') result.extra = true;
  if (process.env.STUB_FAILURE === 'exit') process.exit(7);
  mkdirSync(args[5], {recursive: true});
  writeFileSync(join(args[5], 'result.json'), JSON.stringify({result: {json: {
    ...request, adapter, hardware: !adapter.isFallbackAdapter,
    brick: {name:'raa-anim', version:'0.5.0'}, results:[result]
  }}}));
}
`);
  return {
    dir,
    run(failure = '') {
      return spawnSync(process.execPath, ['scripts/anim-capture.mjs', 'clip-sample'], {
        cwd: dir, encoding: 'utf8', timeout: 10_000,
        env: { ...process.env, NMC_GPU_CLIENT: join(dir, 'stub-client.mjs'), STUB_FAILURE: failure },
      });
    },
  };
}

test('clip-sample publishes the staged page and writes canonical records to its output path', t => {
  const f = fixture(t);
  const child = f.run();
  assert.equal(child.status, 0, child.stderr);
  const output = join(f.dir, 'proofs/results/anim/clip-sample.json');
  const record = JSON.parse(readFileSync(output, 'utf8'));
  assert.deepEqual(record, JSON.parse(child.stdout));
  assert.equal(record.hardware, true);
  assert.deepEqual(record.records.map(({ case: name, seed, ticks }) => [name, seed, ticks]),
    DET_CASES.map(([seed, ticks]) => ['clip-sample', seed, ticks]));
  for (const result of record.records) {
    assert.deepEqual(Object.keys(result), ['case', 'seed', 'ticks', 'hash']);
    assert.match(result.hash, /^[0-9a-f]{8}$/); // Format only; no hash comparison.
  }
  const calls = readFileSync(join(f.dir, 'calls.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  assert.deepEqual(calls.map(call => call[0]), ['publish', 'render', 'publish', 'render', 'render', 'render']);
  assert.equal(existsSync(join(f.dir, 'proofs/results/anim/det-selftest.json')), false);
});

for (const failure of ['probe', 'software', 'record', 'exit']) {
  test(`clip-sample rejects ${failure} and writes no evidence`, t => {
    const f = fixture(t);
    const child = f.run(failure);
    assert.equal(child.status, 1, child.stderr);
    assert.equal(child.stdout, '');
    assert.equal(existsSync(join(f.dir, 'proofs/results/anim/clip-sample.json')), false);
    assert.equal(readFileSync(join(f.dir, 'calls.jsonl'), 'utf8').trim().split('\n').length,
      failure === 'probe' ? 2 : 4);
  });
}

test('missing NMC_GPU_CLIENT fails without launching Chrome or writing a record', t => {
  const f = fixture(t);
  const child = spawnSync(process.execPath, ['scripts/anim-capture.mjs', '--case', 'clip-sample'], {
    cwd: f.dir, encoding: 'utf8', timeout: 5000,
    env: { ...process.env, NMC_GPU_CLIENT: '' },
  });
  assert.equal(child.status, 1);
  assert.match(child.stderr, /NMC_GPU_CLIENT is required/);
  assert.equal(existsSync(join(f.dir, 'calls.jsonl')), false);
  assert.equal(existsSync(join(f.dir, 'proofs/results/anim/clip-sample.json')), false);
});

test('page validation reaches real WASM dispatch for both cases before any GPU setup', async () => {
  // Execute the page's actual validation/run functions with setup supplied by this
  // CPU test. This is dispatch coverage, not browser/GPU capture evidence.
  const { loadRaaAnim } = await import('../dist/bricks/raa-anim/index.js');
  const api = await loadRaaAnim();
  const source = readFileSync(join(root, 'examples/anim-brick/main.js'), 'utf8');
  const calls = [];
  let setups = 0;
  const wrapped = { selftest(...args) { calls.push(args); return api.selftest(...args); } };
  const run = runInNewContext(source.slice(source.indexOf('function isU32('),
    source.indexOf('registerNmcGpu({ run });')) + '\nrun', {
    errorText: {}, resultsText: {}, status: {}, gpuFailure: undefined,
    showError() {},
    setup: async () => {
      setups++;
      return { api: wrapped, brick: api.brickInfo(), adapter: {}, hardware: false };
    },
  });
  const invalidNumbers = [NaN, Infinity, -1, 0.5, 4294967296, '1', null, undefined];
  for (const name of ['clip-sample', 'det-selftest']) {
    for (const [seed, ticks] of DET_CASES) {
      const output = await run({ case: name, seeds: [seed], ticks });
      assert.deepEqual(calls.at(-1), [name, seed, ticks]);
      assert.equal(output.results[0].case, name);
      assert.equal(output.results[0].seed, seed);
      assert.equal(output.results[0].ticks, ticks);
    }
    const before = setups;
    for (const value of invalidNumbers) {
      await assert.rejects(run({ case: name, seeds: [value], ticks: 0 }), /Expected/);
      await assert.rejects(run({ case: name, seeds: [0], ticks: value }), /Expected/);
    }
    for (const seeds of [[], new Array(1), null, '1']) {
      await assert.rejects(run({ case: name, seeds, ticks: 0 }), /Expected/);
    }
    assert.equal(setups, before); // Invalid input keeps the pre-dispatch error path.
  }
  for (const args of [null, [], {}, { case: 'unknown', seeds: [0], ticks: 0 }]) {
    await assert.rejects(run(args), /det-selftest.*clip-sample/);
  }
  assert.equal(calls.length, 6);
});
