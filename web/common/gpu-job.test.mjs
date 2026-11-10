// node --test web/common/gpu-job.test.mjs: browser-free managed transport checks.
// The controlled client follows the installed publication/result-file conventions;
// it is a protocol fixture, never a browser, GPU or timing substitute.
import assert from 'node:assert/strict';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { publishGpuPage, runGpuPageOp } from './gpu-job.mjs';

const ENV = ['NMC_GPU_CLIENT', 'NMC_GPU_WAIT_S', 'NMC_GPU_TOKEN', 'GPU_JOB_TEST_CASE', 'GPU_JOB_TEST_TRACE'];
const REF = 'host-page-123';

const CLIENT = String.raw`
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
const [op, input, ...args] = process.argv.slice(2);
const mode = process.env.GPU_JOB_TEST_CASE;
appendFileSync(process.env.GPU_JOB_TEST_TRACE, JSON.stringify({op, input, args, wait: process.env.NMC_GPU_WAIT_S}) + '\n');
const sleep = ms => new Promise(r => setTimeout(r, ms));
if (op === 'publish') {
  if (mode === 'publish-failure') { console.error('controlled publication failure'); process.exit(1); }
  if (mode === 'publish-token') { console.error('publication rejected token ' + process.env.NMC_GPU_TOKEN); process.exit(1); }
  if (mode === 'publish-timeout') await sleep(5000);
  if (mode === 'publish-empty') { console.log(JSON.stringify({data:{}})); process.exit(0); }
  console.log(mode === 'publish-line' ? 'page_ref: host-page-123' : JSON.stringify({data:{page_ref:'host-page-123'}}));
  process.exit(0);
}
const body = JSON.parse(readFileSync(input, 'utf8'));
const out = args[args.indexOf('-o') + 1];
mkdirSync(out, { recursive: true });
const write = doc => writeFileSync(join(out, 'result.json'), JSON.stringify(doc));
if (mode === 'job-failure') { console.error('controlled job failure'); process.exit(3); }
if (mode === 'job-token') { console.error('job rejected token ' + process.env.NMC_GPU_TOKEN); process.exit(3); }
if (mode === 'waiting') { console.error('GPU job fake: waiting for the GPU slot (0 s)'); await sleep(5000); }
if (mode === 'running') { console.error('GPU job fake: running (0 s)'); await sleep(5000); }
if (mode === 'download') { console.error('GPU job fake: running (0 s)'); console.error('GPU job fake: done'); await sleep(5000); }
if (mode === 'done-window') { console.error('GPU job fake: running (0 s)'); console.error('GPU job fake: done'); await sleep(120); }
if (mode === 'no-result') process.exit(0);
if (mode === 'invalid-json') { writeFileSync(join(out, 'result.json'), 'not json'); process.exit(0); }
if (mode === 'array-reply') { write([]); process.exit(0); }
if (mode === 'missing-ids') { write({result: body}); process.exit(0); }
if (mode === 'missing-result') { write({job_id: 'job-1', client_id: 'client-1'}); process.exit(0); }
if (mode === 'numeric-ids') { write({job_id: 17, client_id: 23, result: {ok: true}}); process.exit(0); }
if (mode === 'nested' || mode === 'array') { write({job_id: 'job-1', client_id: 'client-1', result: body.value}); process.exit(0); }
write({job_id: 'job-1', client_id: 'client-1', result: body});
`;

function setup(t, mode = 'ok', env = {}) {
  const root = mkdtempSync(join(tmpdir(), 'gpu-job-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const page = join(root, 'dist/page');
  mkdirSync(page, { recursive: true });
  writeFileSync(join(page, 'nmc.gpu.json'), JSON.stringify({ page: 'index.html', ops: ['render', 'capture'] }));
  writeFileSync(join(page, 'index.html'), '<!doctype html>\n');
  const client = join(root, 'client.mjs');
  const trace = join(root, 'trace.jsonl');
  writeFileSync(client, CLIENT);
  const saved = Object.fromEntries(ENV.map(name => [name, process.env[name]]));
  t.after(() => {
    for (const name of ENV) {
      if (saved[name] === undefined) delete process.env[name];
      else process.env[name] = saved[name];
    }
  });
  for (const name of ENV) delete process.env[name];
  Object.assign(process.env, { NMC_GPU_CLIENT: client, GPU_JOB_TEST_CASE: mode, GPU_JOB_TEST_TRACE: trace }, env);
  return {
    root, page, client, out: join(root, 'out'),
    calls: () => existsSync(trace) ? readFileSync(trace, 'utf8').trim().split('\n').filter(Boolean).map(line => JSON.parse(line)) : [],
  };
}

test('publication refuses missing configuration and unverified built page directories', async t => {
  const f = setup(t);
  delete process.env.NMC_GPU_CLIENT;
  await assert.rejects(publishGpuPage(f.page, { timeoutMs: 5000 }), /NMC_GPU_CLIENT is missing/);
  process.env.NMC_GPU_CLIENT = f.client;
  await assert.rejects(publishGpuPage(join(f.root, 'dist/missing'), { timeoutMs: 5000 }), /page directory is missing/);
  const bare = join(f.root, 'dist/bare'); mkdirSync(bare);
  await assert.rejects(publishGpuPage(bare, { timeoutMs: 5000 }), /nmc\.gpu\.json is missing/);
  const broken = join(f.root, 'dist/broken'); mkdirSync(broken);
  writeFileSync(join(broken, 'nmc.gpu.json'), '{}');
  await assert.rejects(publishGpuPage(broken, { timeoutMs: 5000 }), /must declare \{page, ops\}/);
  const absent = join(f.root, 'dist/absent'); mkdirSync(absent);
  writeFileSync(join(absent, 'nmc.gpu.json'), JSON.stringify({ page: 'index.html', ops: ['render'] }));
  await assert.rejects(publishGpuPage(absent, { timeoutMs: 5000 }), /declared page index\.html is missing/);
  assert.deepEqual(f.calls(), [], 'no publication before a verified built directory');
});

test('both operations require a positive bounded work timeout', async t => {
  const f = setup(t);
  for (const timeoutMs of [undefined, 0, -1, 300001, Number.NaN]) {
    await assert.rejects(publishGpuPage(f.page, { timeoutMs }), /work timeoutMs must be a positive number of at most 300000 ms/);
    await assert.rejects(runGpuPageOp(REF, 'capture', {}, { timeoutMs }), /work timeoutMs/);
  }
  assert.deepEqual(f.calls(), []);
  const reply = await runGpuPageOp(REF, 'capture', { ok: 1 }, { timeoutMs: 300000 });
  assert.equal(reply.jobId, 'job-1', 'the 300000 ms running cap is accepted');
});

test('queue availability stays a separate NMC_GPU_WAIT_S allowance', async t => {
  const f = setup(t, 'ok', { NMC_GPU_WAIT_S: '0' });
  await assert.rejects(publishGpuPage(f.page, { timeoutMs: 1000 }), /NMC_GPU_WAIT_S must be positive seconds/);
  await assert.rejects(runGpuPageOp(REF, 'capture', {}, { timeoutMs: 1000 }), /NMC_GPU_WAIT_S must be positive seconds/);
  process.env.NMC_GPU_WAIT_S = '0.05';
  process.env.NMC_GPU_CLIENT = '/missing-controlled-client';
  await assert.rejects(runGpuPageOp(REF, 'capture', {}, { timeoutMs: 1000 }), /NMC_GPU_CLIENT unavailable/);
  assert.deepEqual(f.calls(), []);
});

test('publication parses host references and attributes publication failures', async t => {
  const f = setup(t);
  assert.deepEqual(await publishGpuPage(f.page, { timeoutMs: 5000 }), { pageRef: REF });
  const call = f.calls().at(-1);
  assert.equal(call.op, 'publish');
  assert.equal(call.input, f.page);
  process.env.GPU_JOB_TEST_CASE = 'publish-line';
  assert.deepEqual(await publishGpuPage(f.page, { timeoutMs: 5000 }), { pageRef: REF }, 'page_ref line convention');
  process.env.GPU_JOB_TEST_CASE = 'publish-empty';
  await assert.rejects(publishGpuPage(f.page, { timeoutMs: 5000 }), /returned no page_ref/);
  process.env.GPU_JOB_TEST_CASE = 'publish-failure';
  await assert.rejects(publishGpuPage(f.page, { timeoutMs: 5000 }), /controlled publication failure/);
  process.env.GPU_JOB_TEST_CASE = 'publish-token';
  process.env.NMC_GPU_TOKEN = 'publication-secret';
  await assert.rejects(publishGpuPage(f.page, { timeoutMs: 5000 }),
    (error) => /\[redacted\]/.test(error.message) && !error.message.includes('publication-secret'));
  process.env.GPU_JOB_TEST_CASE = 'publish-timeout';
  await assert.rejects(publishGpuPage(f.page, { timeoutMs: 150 }), /publication timeout/);
});

test('runGpuPageOp requires host identifiers and a declared operation', async t => {
  const f = setup(t);
  await assert.rejects(runGpuPageOp('', 'capture', {}, { timeoutMs: 5000 }), /nonempty host-issued page reference/);
  await assert.rejects(runGpuPageOp('   ', 'capture', {}, { timeoutMs: 5000 }), /nonempty host-issued page reference/);
  await assert.rejects(runGpuPageOp(REF, '', {}, { timeoutMs: 5000 }), /nonempty declared operation name/);
  assert.deepEqual(f.calls(), []);
});

test('runGpuPageOp preserves the complete nested direct result without json extraction', async t => {
  const f = setup(t, 'nested');
  const value = {
    json: { legacy: true },
    nested: { items: [1, { deep: 'x' }, [2, 3]], flag: false },
    remainingMs: 270000,
  };
  const reply = await runGpuPageOp(REF, 'capture', { value }, { timeoutMs: 5000 });
  assert.deepEqual(reply, { jobId: 'job-1', clientId: 'client-1', result: value });
  assert.equal(reply.result.json.legacy, true, 'a nested json member is result data, not an extraction path');
  const call = f.calls().at(-1);
  assert.equal(call.op, 'capture');
  assert.deepEqual(call.args, ['--page-ref', REF, '-o', call.args[3]]);
});

test('runGpuPageOp forwards operation input unchanged, including remainingMs', async t => {
  const f = setup(t);
  const input = { args: { remainingMs: 270000, selection: { nested: [1, { x: 2 }] } } };
  const reply = await runGpuPageOp(REF, 'capture', input, { timeoutMs: 5000, out: f.out });
  assert.deepEqual(reply.result, input);
  assert.deepEqual(JSON.parse(readFileSync(join(f.out, 'input.json'), 'utf8')), input);
  assert.equal(JSON.parse(readFileSync(join(f.out, 'input.json'), 'utf8')).args.remainingMs, 270000);
  assert.ok(existsSync(join(f.out, 'result.json')), 'a provided output directory retains the client reply');
});

test('runGpuPageOp preserves a direct array result', async t => {
  setup(t, 'array');
  const value = [1, { nested: ['a', 2] }];
  const reply = await runGpuPageOp(REF, 'capture', { value }, { timeoutMs: 5000 });
  assert.deepEqual(reply.result, value);
});

test('runGpuPageOp attributes job failures, redacts tokens and refuses malformed replies', async t => {
  setup(t, 'job-failure');
  await assert.rejects(runGpuPageOp(REF, 'capture', {}, { timeoutMs: 5000 }), /managed client failed \(3\): controlled job failure/);
  process.env.GPU_JOB_TEST_CASE = 'job-token';
  process.env.NMC_GPU_TOKEN = 'job-secret';
  await assert.rejects(runGpuPageOp(REF, 'capture', {}, { timeoutMs: 5000 }),
    (error) => /\[redacted\]/.test(error.message) && !error.message.includes('job-secret'));
  for (const [mode, pattern] of [
    ['no-result', /result\.json is missing/],
    ['invalid-json', /not valid JSON/],
    ['array-reply', /not a host reply object/],
    ['missing-ids', /missing nonempty managed job\/client provenance/],
    ['missing-result', /missing its complete result/],
  ]) {
    process.env.GPU_JOB_TEST_CASE = mode;
    await assert.rejects(runGpuPageOp(REF, 'capture', {}, { timeoutMs: 5000 }), pattern, mode);
  }
  process.env.GPU_JOB_TEST_CASE = 'numeric-ids';
  assert.deepEqual(await runGpuPageOp(REF, 'capture', {}, { timeoutMs: 5000 }),
    { jobId: '17', clientId: '23', result: { ok: true } });
});

test('queue, running and download phases use separately attributed clocks', async t => {
  setup(t, 'waiting', { NMC_GPU_WAIT_S: '0.15' });
  await assert.rejects(runGpuPageOp(REF, 'capture', {}, { timeoutMs: 5000 }), /GPU waiting timeout/);
  process.env.GPU_JOB_TEST_CASE = 'running';
  process.env.NMC_GPU_WAIT_S = '2';
  await assert.rejects(runGpuPageOp(REF, 'capture', {}, { timeoutMs: 300 }), /GPU job running timeout after 0\.3s/);
  process.env.GPU_JOB_TEST_CASE = 'download';
  process.env.NMC_GPU_WAIT_S = '0.3';
  await assert.rejects(runGpuPageOp(REF, 'capture', {}, { timeoutMs: 5000 }), /managed output download timeout/);
  process.env.GPU_JOB_TEST_CASE = 'done-window';
  process.env.NMC_GPU_WAIT_S = '2';
  const reply = await runGpuPageOp(REF, 'capture', { value: { ok: true } }, { timeoutMs: 50 });
  assert.equal(reply.result.value.ok, true, 'a completed job keeps its separate download window');
});
