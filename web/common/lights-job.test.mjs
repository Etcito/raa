import { captureEvidence } from './lights-evidence.js';
// node --test web/common/lights-job.test.mjs: CPU-only selection and visibility checks.
import assert from 'node:assert/strict';
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync, spawn } from 'node:child_process';
import test from 'node:test';
import { check, inView, main, select, variantProblems } from './lights-job.mjs';

const CAM = { eye: [0, 2, 5], target: [0, 0, 0], fovYDeg: 60, near: 0.1, far: 100 };
const FLOOR = { min: [-50, -1, -50], max: [50, 0, 50] };
const BEHIND = { min: [0, 0, 10], max: [1, 1, 11] };
const EDGE = { min: [4.5, 0, -1], max: [5, 1, 1] };
const scene = (id = 'point', subjects = [FLOOR], milestone = 'm28') => ({ id, milestone, variants: ['on', 'off'].map((variant) =>
  ({ id: variant, description: `${id} ${variant}`, judge: true, timeMs: 1000, selection: { camera: CAM, subjects } })) });
const read = (path) => JSON.parse(readFileSync(path, 'utf8'));
const source = (def) => `export default { ...${JSON.stringify(def)}, build: async () => {} };`;

function writePage(t, defs, base = tmpdir()) {
  const root = mkdtempSync(join(base, 'lights-job-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const page = join(root, 'dist/lights');
  mkdirSync(join(page, 'scenes'), { recursive: true });
  const index = defs.map((def) => {
    const file = `scenes/${def.id}.js`;
    writeFileSync(join(page, file), source(def));
    return { id: def.id, milestone: def.milestone, file };
  });
  writeFileSync(join(page, 'scenes.json'), JSON.stringify(index));
  return { root, page, out: join(root, 'out/lights') };
}

test('inView accepts an enclosing floor and rejects hidden or off-axis boxes', () => {
  assert.equal(inView(CAM, FLOOR), true);
  assert.equal(inView(CAM, BEHIND), false);
  assert.equal(inView(CAM, { min: [0, 20, -1], max: [1, 21, 1] }), false);
  assert.equal(inView(CAM, EDGE), true);
  assert.equal(inView({ ...CAM, aspect: 1 }, EDGE), false);
  assert.equal(inView({ ...CAM, eye: [0, 0, 0], target: [0, 0, 0] }, FLOOR), false);
});

test('select reads only the requested build and writes its on/off pairs', async (t) => {
  const point = writePage(t, [scene()]);
  const spot = writePage(t, [scene('spot')]);
  writeFileSync(join(point.root, 'scene.js'), source(scene('ghost')));
  const pairs = await select(point.page, point.out);
  assert.deepEqual(pairs.map((p) => `${p.scene}--${p.variant}`), ['point--on', 'point--off']);
  const doc = read(join(point.out, 'selection.json'));
  assert.equal(doc.page, point.page);
  assert.deepEqual(doc.pairs.map((p) => p.description), ['point on', 'point off']);
  assert.deepEqual((await check(spot.page)).map((p) => p.scene), ['spot', 'spot']);
});

test('m28 defaults and explicit m28 stay separate from metadata-selected m03', async t => {
  const m3 = scene('mama-lights', [FLOOR], 'm03');
  m3.variants.forEach(variant => { variant.evidence = { filename: `mama-${variant.id}.png`, label: `MAMA ${variant.id} lights` }; });
  const f = writePage(t, [scene(), m3]);
  assert.deepEqual((await check(f.page)).map(pair => pair.scene), ['point', 'point']);
  assert.deepEqual((await main(['check', '--page', f.page, '--milestone', 'm28'])), undefined);
  assert.deepEqual((await main(['check', '--page', f.page, '--milestone', 'm03'])), undefined);
  const m28out = join(f.root, 'out/m28'), m03out = join(f.root, 'out/m03');
  await main(['select', '--page', f.page, '--out', m28out]);
  await main(['select', '--page', f.page, '--out', m03out, '--milestone', 'm03']);
  const oldSelection = read(join(m28out, 'selection.json'));
  const newSelection = read(join(m03out, 'selection.json'));
  assert.equal(oldSelection.milestone, 'm28');
  assert.deepEqual(oldSelection.pairs.map(pair => pair.scene), ['point', 'point']);
  assert.equal(newSelection.milestone, 'm03');
  assert.deepEqual(newSelection.pairs.map(pair => pair.evidence.filename), ['mama-on.png', 'mama-off.png']);
  for (const milestone of ['28', 'm3', 'm300', '']) {
    await assert.rejects(main(['check', '--page', f.page, '--milestone', milestone]), /invalid milestone selector/);
  }
  await assert.rejects(main(['check', '--page', f.page, '--milestone', 'm99']), /selection is empty/);
});

test('selector mismatch is rejected before cleanup and changed metadata invalidates the selected inventory', async t => {
  const f = writePage(t, [scene(), scene('mama-lights', [FLOOR], 'm03')]);
  await select(f.page, f.out);
  mkdirSync(join(f.out, '.lights-runs/00000000-0000-4000-8000-000000000000'), { recursive: true });
  writeFileSync(join(f.out, '.lights-current.json'), JSON.stringify({attempt:'.lights-runs/00000000-0000-4000-8000-000000000000'}));
  await assert.rejects(main(['clean', '--page', f.page, '--out', f.out, '--milestone', 'm03']), /selection milestone m28 does not match/);
  assert.equal(existsSync(join(f.out, '.lights-current.json')), true);
  writeFileSync(join(f.page, 'scenes/late-scene.js'), source(scene('late-scene')));
  const index = read(join(f.page, 'scenes.json'));
  index.push({id:'late-scene',milestone:'m28',file:'scenes/late-scene.js'});
  writeFileSync(join(f.page, 'scenes.json'), JSON.stringify(index));
  await assert.rejects(main(['clean', '--page', f.page, '--out', f.out]), /selection differs from produced metadata/);
  assert.equal(existsSync(join(f.out, '.lights-current.json')), true);
});

test('select refuses missing, empty, malformed and escaping scene indexes', async (t) => {
  const missing = writePage(t, [scene()]);
  rmSync(join(missing.page, 'scenes.json'));
  await assert.rejects(select(missing.page, missing.out), /scenes\.json is missing: build the page first/);
  const empty = writePage(t, [scene()]);
  writeFileSync(join(empty.page, 'scenes.json'), '[]');
  await assert.rejects(check(empty.page), /nonempty array/);
  const malformed = writePage(t, [scene()]);
  writeFileSync(join(malformed.page, 'scenes.json'), '[{"id":"point"}]');
  await assert.rejects(check(malformed.page), /entry 0 is not \{ id, milestone, file \}/);
  const escaping = writePage(t, [scene()]);
  writeFileSync(join(escaping.root, 'outside.js'), 'export default {};');
  writeFileSync(join(escaping.page, 'scenes.json'), JSON.stringify([{ id: 'point', milestone: 'm28', file: '../../outside.js' }]));
  await assert.rejects(check(escaping.page), /is not in the page build/);
  const absent = writePage(t, [scene()]);
  rmSync(join(absent.page, 'scenes/point.js'));
  await assert.rejects(check(absent.page), /module scenes\/point\.js is not in the page build/);
});

test('select refuses invalid scene metadata', async (t) => {
  const mismatch = writePage(t, [scene()]);
  writeFileSync(join(mismatch.page, 'scenes/point.js'), source(scene('ghost')));
  await assert.rejects(check(mismatch.page), /metadata is invalid/);
  const noBuild = writePage(t, [scene()]);
  writeFileSync(join(noBuild.page, 'scenes/point.js'), `export default ${JSON.stringify(scene())};`);
  await assert.rejects(check(noBuild.page), /metadata is invalid/);
  const duplicate = writePage(t, [scene(), scene()]);
  await assert.rejects(check(duplicate.page), /duplicate scene/);
  const wrongVariant = writePage(t, [scene()]);
  const def = scene();
  def.variants[0].id = 'both';
  writeFileSync(join(wrongVariant.page, 'scenes/point.js'), source(def));
  await assert.rejects(check(wrongVariant.page), /variant both is not on or off/);
  const missingOff = writePage(t, [scene()]);
  const onlyOn = scene(); onlyOn.variants.pop();
  writeFileSync(join(missingOff.page, 'scenes/point.js'), source(onlyOn));
  await assert.rejects(check(missingOff.page), /must include on and off variants/);
});

test('select refuses invalid variant metadata', async (t) => {
  const cases = [
    [(s) => { s.variants[0].description = ' '; }, /no description/],
    [(s) => { s.variants[0].judge = false; }, /judge must be true/],
    [(s) => { s.variants[0].timeMs = -1; }, /timeMs is not a duration/],
    [(s) => { s.variants[0].selection.camera = { eye: [0, 0, 0], target: [0, 0, 0] }; }, /camera/],
    [(s) => { s.variants[0].selection.subjects = []; }, /at least one subject/],
    [(s) => { s.variants[0].selection.subjects = [{ min: [0, 0, 0], max: [-1, 1, 1] }]; }, /min<=max/],
  ];
  for (const [mutate, pattern] of cases) {
    const def = scene();
    mutate(def);
    const page = writePage(t, [def]);
    await assert.rejects(select(page.page, page.out), pattern);
  }
});

test('select refuses subjects outside the camera view', async (t) => {
  const behind = writePage(t, [scene('point', [FLOOR, BEHIND])]);
  await assert.rejects(select(behind.page, behind.out), /subjects\[1\] is outside the camera view/);
  const above = writePage(t, [scene('point', [{ min: [0, 30, 0], max: [1, 31, 1] }])]);
  await assert.rejects(check(above.page), /subjects\[0\] is outside the camera view/);
});

test('check validates without writing and the CLI runs both commands', async (t) => {
  const f = writePage(t, [scene()]);
  assert.equal((await check(f.page)).length, 2);
  assert.equal(existsSync(f.out), false);
  await main(['check', '--page', f.page]);
  await assert.rejects(main(['select', '--page', f.page]), /usage/);
  await main(['select', '--page', f.page, '--out', f.out]);
  assert.equal(read(join(f.out, 'selection.json')).pairs.length, 2);
});

test('every lights scene is discovered with each preset and on/off variant selectable', async () => {
  const scenes = new URL('../../examples/lights/scenes/', import.meta.url);
  const ids = [];
  for (const file of readdirSync(scenes).filter((name) => name.endsWith('.js')).sort()) {
    const scene = (await import(new URL(file, scenes))).default;
    assert.equal(scene.id, file.slice(0, -3), 'build-lights.sh discovers a scene by its file name');
    assert.match(scene.milestone, /^m\d\d$/);
    assert.deepEqual(scene.variants.map((variant) => variant.id), ['on', 'off']);
    for (const variant of scene.variants) {
      assert.deepEqual(variantProblems(variant), [], `${scene.id}--${variant.id}`);
    }
    ids.push(scene.id);
  }
  for (const id of ['point', 'spot', 'directional', 'area', 'many-lights', 'attached-light', 'motion']) {
    assert.ok(ids.includes(id), `${id} is discovered`);
  }
});

const WORKTREE = fileURLToPath(new URL('../../', import.meta.url));
const JOB = fileURLToPath(new URL('./lights-job.mjs', import.meta.url));
const COMMIT = execFileSync('git', ['-C', WORKTREE, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();

// This is a controlled CLI producer, not a browser or a GPU/timing substitute.
// It uses the installed client's actual publication/result-file conventions.
const CLIENT = String.raw`
import { captureEvidence } from '${new URL('./lights-evidence.js',import.meta.url).href}';
import { mkdirSync, readFileSync, writeFileSync, appendFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { deflateSync } from 'node:zlib';
import { randomUUID } from 'node:crypto';
const [op, input, ...args] = process.argv.slice(2);
const mode = process.env.LIGHTS_TEST_CASE;
appendFileSync(process.env.LIGHTS_TEST_TRACE, JSON.stringify({op, input, args, wait: process.env.NMC_GPU_WAIT_S}) + '\n');
if (mode === 'publish-timeout' && op === 'publish') await new Promise(r => setTimeout(r, 5000));
if (op === 'publish') {
  if (mode === 'publish-failure') { console.error('controlled publication failure'); process.exit(1); }
  console.log('publication progress');
  const ref = 'host-page-123';
  console.log(mode === 'publish-line' ? 'page_ref: ' + ref : JSON.stringify({data:{page_ref:ref}}));
  process.exit(0);
}
const pairs = JSON.parse(readFileSync(input)).args.pairs;
if (!Array.isArray(pairs)) throw Error('expected whole selection');
if (args[args.indexOf('--page-ref') + 1] !== 'host-page-123') throw Error('wrong page reference');
const batchDir = args[args.indexOf('-o') + 1];
const steps = [], diagnostics = [];
const job_id = randomUUID();
for (const pair of pairs) {
if (Object.keys(pair).sort().join(',') !== 'scene,variant') throw Error('unexpected render arguments');
const dir = join(batchDir, 's-' + encodeURIComponent(pair.scene), pair.variant);
mkdirSync(dir, { recursive:true });
const old = process.env.LIGHTS_TEST_OLD;
if (mode === 'failed-old') {
  if (!old || !readFileSync(old).length) throw Error('test needs old PNG');
  console.error('controlled failed producer; old PNG retained'); process.exit(1);
}
if (mode === 'waiting-timeout') { console.error('GPU job fake: waiting for the GPU slot (0 s)'); await new Promise(r => setTimeout(r, 5000)); }
const measured = pair.scene === 'many-lights';
const width = measured ? 1920 : 1280, height = measured ? 1080 : 720;
// Every pixel is initialized here, including its filter byte; no other pass writes it.
function png() {
  const chunk = (name, data) => {
    const type = Buffer.from(name), body = Buffer.concat([type, data]);
    let crc = 0xffffffff;
    for (const byte of body) { crc ^= byte; for (let i=0;i<8;i++) crc=(crc>>>1)^((crc&1)?0xedb88320:0); }
    const size=Buffer.alloc(4), checksum=Buffer.alloc(4); size.writeUInt32BE(data.length); checksum.writeUInt32BE((crc^0xffffffff)>>>0);
    return Buffer.concat([size,body,checksum]);
  };
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(width); ihdr.writeUInt32BE(height,4); ihdr[8]=8; ihdr[9]=6;
  const raw=Buffer.alloc((width*4+1)*height); // zeroed scanlines, legal RGBA PNG
  return Buffer.concat([Buffer.from('89504e470d0a1a0a','hex'),chunk('IHDR',ihdr),chunk('IDAT',deflateSync(raw)),chunk('IEND',Buffer.alloc(0))]);
}
const base = variant => ({scene:pair.scene, variant, adapter:{vendor:'NVIDIA',architecture:'ampere',device:'RTX 3090',description:'NVIDIA hardware',isFallbackAdapter:false}, hardware:true, nonEmpty:true, canvas:[width,height], warmup:measured?60:30, frames:measured?300:1});
const browser={userAgent:'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/142.0.0.0 Safari/537.36',platform:'Win32',devicePixelRatio:1};
const pageConfig = JSON.parse(readFileSync(process.env.LIGHTS_TEST_CONFIG));
const primaryName = pageConfig.captures?.[pair.scene]?.[pair.variant] ?? 'capture.png';
const lifecycleNames = pair.scene === 'attached-light'
  ? ['initial','translated','rotated','reattached','detached','reattached-before-delete','parent-deleted','light-removed','attached-final']
  : ['supplement'];
let result = {...base(pair.variant),browser,gpuErrors:[],lifecycle:{frames:lifecycleNames.map(name => ({name,stats:{}}))}};
if (measured) {
  const page = JSON.parse(readFileSync(process.env.LIGHTS_TEST_CONFIG));
  const {summarizeFrameIntervals} = await import(page.runner);
  const frameIntervalsMs = Array(300).fill(10);
  const timing = {...summarizeFrameIntervals(frameIntervalsMs),frameIntervalsMs};
  const runs={on:{...base('on'),...timing},off:{...base('off'),...timing}};
  result={...result,...runs[pair.variant],runs,measurement_for:page.measurementFor};
  if (mode === 'no-off') delete runs.off;
  if (mode === 'bad-interval') runs.off.frameIntervalsMs=[0,...frameIntervalsMs.slice(1)];
  if (mode === 'short-intervals') runs.off.frameIntervalsMs=frameIntervalsMs.slice(1);
  if (mode === 'bad-fps') runs.off.fpsMedian=999;
  if (mode === 'slow') {
    const slow=Array(300).fill(20); Object.assign(runs.on,summarizeFrameIntervals(slow),{frameIntervalsMs:slow});
    if (pair.variant === 'on') Object.assign(result,runs.on);
  }
  if (mode === 'bad-off-adapter') runs.off.adapter={...runs.off.adapter,isFallbackAdapter:true};
}
if (mode === 'wrong-pair') result.variant='wrong';
if (mode === 'empty-scene') result.nonEmpty=false;
if (mode === 'software') result.adapter.description='SwiftShader software';
if (mode === 'blank-identity') for (const key of ['vendor','architecture','device','description']) result.adapter[key]='';
if (mode === 'fallback') result.adapter.isFallbackAdapter=true;
if (mode === 'wrong-platform') browser.platform='Linux x86_64';
if (mode === 'wrong-browser') browser.userAgent=browser.userAgent.replace('Chrome/','Firefox/');
if (mode === 'device-errors') result.gpuErrors=['GPU device lost: reset; shader validation failed'];
if (mode === 'missing-errors') delete result.gpuErrors;
if (mode === 'wrong-dpr') browser.devicePixelRatio=2;
if (mode === 'wrong-size') result.canvas=[32,32];
if (mode === 'wrong-counts') result.frames=0;
if (mode !== 'no-output') {
  steps.push({...pair,result});
  if (mode !== 'no-capture') writeFileSync(join(dir,mode === 'arbitrary-png'?'other.png':primaryName), mode === 'corrupt-png'?Buffer.from('not PNG'):mode === 'empty-png'?Buffer.alloc(0):png());
  if (mode !== 'missing-lifecycle') for (const name of lifecycleNames) writeFileSync(join(dir,pair.scene === 'attached-light' ? captureEvidence(pair.scene,pair.variant,name).filename : name + '.png'),png());
}
}
if (mode === 'page-errors') diagnostics.push({pair:pairs[0],kind:'pageerror',message:'all frames still attempted'});
if (mode === 'missing-step') steps.pop();
if (mode !== 'no-result' && mode !== 'no-output') writeFileSync(join(batchDir,'result.json'),JSON.stringify({job_id,client_id:'host-windows-client',result:{steps,diagnostics}}));
`;

async function transportFixture(t, measured = false, definitions = null) {
  const base = join(WORKTREE, 'out');
  mkdirSync(base, { recursive: true });
  const defs = definitions ?? [scene(measured ? 'many-lights' : 'point')];
  const manyLights = defs.find(definition => definition.id === 'many-lights');
  if (measured || manyLights) (manyLights ?? defs[0]).measure = (await import('../../examples/lights/scenes/many-lights.js')).default.measure;
  // Place the build fixture in the real worktree: provenance reads real Git HEAD.
  const f = writePage(t, defs, base);
  mkdirSync(join(f.page, 'lib'));
  writeFileSync(join(f.page, 'lib/runner.js'), readFileSync(new URL('../../examples/lookdev/lib/runner.js', import.meta.url)));
  writeFileSync(join(f.page, 'build.json'), JSON.stringify({buildId:'build-fixture',commit:COMMIT,builtAt:new Date().toISOString()}));
  f.client = join(f.root, 'client.mjs');
  f.trace = join(f.root, 'trace.jsonl');
  f.config = join(f.root, 'config.json');
  writeFileSync(f.client, CLIENT);
  writeFileSync(f.config, JSON.stringify({runner:new URL('../../examples/lookdev/lib/runner.js', import.meta.url).href,
    measurementFor:(manyLights ?? defs[0]).measure?.measurementFor,
    captures:Object.fromEntries(defs.map(definition => [definition.id, Object.fromEntries(definition.variants.map(variant =>
      [variant.id, variant.evidence?.filename ?? captureEvidence(definition.id, variant.id)?.filename ?? 'capture.png']))]))}));
  await select(f.page, f.out);
  return f;
}

function cli(f, op, mode = 'ok', extra = {}, milestone) {
  return new Promise((resolveRun, reject) => {
    const child = spawn(process.execPath, [JOB, op, '--page', f.page, '--out', f.out, ...(milestone ? ['--milestone', milestone] : [])], {
      env: {...process.env, NMC_GPU_CLIENT:f.client, NMC_GPU_REVISION:COMMIT, NMC_GPU_WAIT_S:'2', NMC_CPUS:'1',
        LIGHTS_TEST_TRACE:f.trace,LIGHTS_TEST_CONFIG:f.config,LIGHTS_TEST_CASE:mode,NMC_JUDGE_CLIENT:f.judgeClient,...extra},
      stdio:['ignore','pipe','pipe'],shell:false,
    });
    let stdout='',stderr='';
    child.stdout.on('data', d => { stdout+=d; }); child.stderr.on('data', d => { stderr+=d; });
    child.on('error',reject);
    child.on('close',code => resolveRun({code,stdout,stderr}));
  });
}
const passed = result => assert.equal(result.code, 0, result.stderr);
const failed = (result, pattern) => { assert.notEqual(result.code,0,result.stdout); assert.match(result.stderr,pattern); };

test('managed publication/render/fresh preserve explicit captures, page JSON and provenance', async t => {
  const f = await transportFixture(t);
  passed(await cli(f, 'render'));
  passed(await cli(f, 'fresh'));
  const record = read(join(f.out, 'run.json'));
  assert.equal(record.complete, true);
  assert.equal(record.buildId, 'build-fixture');
  assert.equal(record.commit, COMMIT);
  assert.equal(record.runId, String(read(join(f.out,record.pairs[0].result)).job_id));
  assert.equal(new Date(record.startedAt).toISOString(), record.startedAt);
  assert.equal(new Date(record.completedAt).toISOString(), record.completedAt);
  assert.deepEqual(record.pairs.map(p => p.variant), ['on', 'off']);
  assert.equal(new Set(record.pairs.map(p => p.capture)).size, 2);
  for (const pair of record.pairs) {
    assert.match(pair.capture, /^\.lights-runs\/[\da-f-]+\/s-point\/(on|off)\/point-(on|off)-distance-attenuation\.png$/);
    assert.ok(existsSync(join(f.out,pair.capture)));
    assert.ok(existsSync(join(f.out,pair.capture.replace(/[^/]+$/, 'supplement.png'))));
    const doc = read(join(f.out,pair.result));
    assert.ok(doc.job_id);
    assert.equal(doc.client_id,'host-windows-client');
    assert.equal(doc.result.variant,pair.variant);
  }
  const calls = readFileSync(f.trace,'utf8').trim().split('\n').map(JSON.parse);
  assert.deepEqual(calls.map(c => c.op), ['publish','render']);
  assert.equal(calls[0].input,f.page);
  for (const call of calls.slice(1)) assert.equal(call.args[1],record.pageRef);
});

test('all CLI operations carry explicit m03 and judge only its metadata inventory', async t => {
  const m3 = scene('mama-lights', [FLOOR], 'm03');
  m3.variants.forEach(variant => { variant.evidence = { filename: `mama-${variant.id}.png`, label: `MAMA ${variant.id}` }; });
  const f = await transportFixture(t, false, [scene(), m3]);
  f.judgeClient = join(f.root, 'judge.py');
  writeFileSync(f.judgeClient, JUDGE_CLIENT);
  await main(['check', '--page', f.page, '--milestone', 'm03']);
  await main(['select', '--page', f.page, '--out', f.out, '--milestone', 'm03']);
  assert.equal(read(join(f.out, 'selection.json')).milestone, 'm03');
  passed(await cli(f, 'clean', 'ok', {}, 'm03'));
  passed(await cli(f, 'render', 'ok', {}, 'm03'));
  passed(await cli(f, 'fresh', 'ok', {}, 'm03'));
  passed(await cli(f, 'judge', 'ok', {}, 'm03'));
  const run = read(join(f.out, 'run.json'));
  const judged = read(join(f.out, 'judge.json'));
  assert.equal(run.milestone, 'm03');
  assert.equal(judged.milestone, 'm03');
  assert.deepEqual(run.pairs.map(pair => pair.scene), ['mama-lights', 'mama-lights']);
  assert.deepEqual(judged.images.map(image => image.description), ['MAMA on', 'MAMA off']);
  assert.equal(judged.images.length, 2);
  const renderInput = JSON.parse(readFileSync(join(f.out, read(join(f.out, '.lights-current.json')).attempt, 'input.json')));
  assert.equal(renderInput.args.milestone, 'm03');
  for (const pair of run.pairs) assert.equal(read(join(f.out, pair.result.replace('result.json', 'completion.json'))).milestone, 'm03');
  const missingCapture = join(f.out, run.pairs[0].capture);
  rmSync(missingCapture);
  failed(await cli(f, 'judge', 'ok', {}, 'm03'), /image judgment failures/);
  const partial = read(join(f.out, 'judge.json'));
  assert.equal(partial.complete, false);
  assert.equal(partial.images.length, 2, 'judgment continues for every selected image');
  assert.ok(partial.images.some(image => image.error === 'missing required image'));
  assert.ok(partial.images.some(image => image.score === 8));
  failed(await cli(f, 'fresh'), /selection milestone m03 does not match invocation m28/);
});

test('publication accepts the reference client page_ref line convention', async t => {
  const f = await transportFixture(t);
  passed(await cli(f,'render','publish-line'));
});

test('measured transport preserves and validates both on/off runs and raw intervals', async t => {
  const f = await transportFixture(t,true);
  passed(await cli(f,'render'));
  passed(await cli(f,'fresh'));
  for (const pair of read(join(f.out,'run.json')).pairs) {
    const {result} = read(join(f.out,pair.result));
    assert.equal(result.runs.on.frameIntervalsMs.length,300);
    assert.equal(result.runs.off.frameIntervalsMs.length,300);
    assert.equal(result.runs.on.fpsMean,100);
  }
});

test('missing/unavailable clients and publication failures invalidate previous completion', async t => {
  for (const [mode, env, pattern] of [
    ['ok',{NMC_GPU_CLIENT:''},/NMC_GPU_CLIENT is missing/],
    ['ok',{NMC_GPU_CLIENT:'/missing-controlled-client',NMC_GPU_WAIT_S:'0.02'},/unavailable/],
    ['publish-failure',{},/controlled publication failure/],
    ['publish-timeout',{NMC_GPU_WAIT_S:'0.03'},/publication timeout/],
    ['waiting-timeout',{NMC_GPU_WAIT_S:'0.1'},/GPU waiting timeout/],
  ]) {
    const f = await transportFixture(t);
    passed(await cli(f,'render'));
    failed(await cli(f,'render',mode,env),pattern);
    failed(await cli(f,'fresh'),/run\.json is missing|incomplete/);
  }
});

test('running timeout uses the 300s running clock independently of the queue clock', async t => {
  const f = await transportFixture(t);
  // A subprocess-only clock acceleration proves the exact production 300000ms
  // deadline without a five-minute unit test or a production timeout override.
  const clock = join(f.root,'clock.mjs');
  writeFileSync(clock, 'const original=globalThis.setTimeout; globalThis.setTimeout=(fn,ms,...a)=>original(fn,ms===300000?20:ms,...a);');
  writeFileSync(f.client, CLIENT.replace("if (mode === 'waiting-timeout')", "if (mode === 'running-timeout') { console.error('GPU job fake: running (0 s)'); await new Promise(r => setTimeout(r,5000)); }\nif (mode === 'waiting-timeout')"));
  failed(await cli(f,'render','running-timeout',{NODE_OPTIONS:`--import=${clock}`}),/running timeout after 300s/);
  failed(await cli(f,'fresh'),/incomplete/);
});

test('failed producer cannot reuse an old valid PNG or completion', async t => {
  const f = await transportFixture(t);
  passed(await cli(f,'render'));
  const old = read(join(f.out,'run.json'));
  const archive = join(f.out,'archive'); mkdirSync(archive);
  const retained = join(archive,'capture.png');
  writeFileSync(retained,readFileSync(join(f.out,old.pairs[0].capture)));
  failed(await cli(f,'render','failed-old',{LIGHTS_TEST_OLD:retained}),/controlled failed producer/);
  assert.ok(existsSync(retained));
  assert.equal(existsSync(join(f.out,old.pairs[0].capture)),false);
  const incomplete = read(join(f.out,'run.json'));
  assert.equal(incomplete.complete,false);
  assert.equal(incomplete.completedAt,null);
  assert.deepEqual(incomplete.pairs,[]);
  failed(await cli(f,'fresh'),/incomplete/);
});

test('transport rejects wrong identity, absent/corrupt outputs and invalid hardware/browser evidence', async t => {
  for (const [mode,pattern] of [
    ['wrong-pair',/wrong scene\/variant/], ['empty-scene',/nonEmpty/],
    ['no-output',/result\.json/], ['no-result',/result\.json/], ['no-capture',/capture\.png/],
    ['arbitrary-png',/capture\.png/], ['empty-png',/empty\/missing output/], ['corrupt-png',/corrupt\/empty PNG/],
    ['software',/hardware adapter/], ['fallback',/hardware adapter/], ['blank-identity',/adapter identity is blank/],
    ['wrong-platform',/Chrome on Windows/], ['wrong-browser',/Chrome on Windows/],
    ['device-errors',/device lost: reset; shader validation failed/], ['missing-errors',/gpuErrors/],
    ['wrong-dpr',/DPR 1/], ['wrong-size',/dimensions/], ['wrong-counts',/frame counts/],
    ['missing-lifecycle',/missing lifecycle output/],
  ]) {
    const f = await transportFixture(t);
    failed(await cli(f,'render',mode),pattern);
    failed(await cli(f,'fresh'),/incomplete/);
  }
});

test('measured transport refuses missing runs, invalid timing and on FPS below 60', async t => {
  for (const [mode,pattern] of [
    ['no-off',/wrong scene\/variant/], ['bad-interval',/raw frame intervals/],
    ['short-intervals',/raw frame intervals/], ['bad-fps',/invalid timing fpsMedian/],
    ['slow',/fpsMean must be >= 60/], ['bad-off-adapter',/hardware adapter/],
  ]) {
    const f = await transportFixture(t,true);
    failed(await cli(f,'render',mode),pattern);
    failed(await cli(f,'fresh'),/incomplete/);
  }
});

test('fresh rejects stale provenance, incomplete records, missing outputs and forged pair paths', async t => {
  const mutations = [
    [f => { const p=join(f.page,'build.json'), b=read(p); b.buildId='next-build'; writeFileSync(p,JSON.stringify(b)); }, /provenance/],
    [f => { const p=join(f.out,'run.json'), r=read(p); r.complete=false; writeFileSync(p,JSON.stringify(r)); }, /incomplete/],
    [f => { const p=join(f.out,'run.json'), r=read(p); r.pairs[0].capture='../neighbor/capture.png'; writeFileSync(p,JSON.stringify(r)); }, /unsafe pair paths/],
    [f => { const r=read(join(f.out,'run.json')); rmSync(join(f.out,r.pairs[0].capture)); }, /capture\.png/],
    [f => { const r=read(join(f.out,'run.json')); writeFileSync(join(f.out,r.pairs[0].capture),''); }, /empty\/missing output/],
    [f => { const r=read(join(f.out,'run.json')); const p=join(f.out,r.pairs[0].result.replace('result.json','completion.json')); const c=read(p); c.runId='old-run'; writeFileSync(p,JSON.stringify(c)); }, /producer completion/],
  ];
  for (const [mutate,pattern] of mutations) {
    const f = await transportFixture(t);
    passed(await cli(f,'render'));
    mutate(f);
    failed(await cli(f,'fresh'),pattern);
  }
});

test('clean retains selection, unrelated files, sibling runs and archives', async t => {
  const f = await transportFixture(t), neighbor = await transportFixture(t);
  passed(await cli(f,'render')); passed(await cli(neighbor,'render'));
  const selection = readFileSync(join(f.out,'selection.json'),'utf8');
  const record = read(join(f.out,'run.json'));
  const archive=join(f.out,'.lights-runs','archived-attempt'); mkdirSync(archive);
  writeFileSync(join(archive,'keep.txt'),'archived');
  writeFileSync(join(f.out,'notes.txt'),'keep');
  passed(await cli(f,'clean'));
  assert.equal(readFileSync(join(f.out,'selection.json'),'utf8'),selection);
  assert.equal(readFileSync(join(f.out,'notes.txt'),'utf8'),'keep');
  assert.equal(readFileSync(join(archive,'keep.txt'),'utf8'),'archived');
  assert.equal(existsSync(join(f.out,record.pairs[0].capture)),false);
  assert.equal(existsSync(join(f.out,'run.json')),false);
  passed(await cli(f,'clean'));
  passed(await cli(neighbor,'fresh'));
});

test('the build staging step records real Git/build provenance and discovered metadata', async t => {
  const f = await transportFixture(t);
  const script = readFileSync(new URL('../../scripts/build-lights.sh',import.meta.url),'utf8');
  const staging = script.split("<<'JS'\n")[1].split('\nJS\n')[0];
  execFileSync(process.execPath,['--input-type=module','-',f.page],{
    input:staging,cwd:WORKTREE,env:{...process.env,NMC_GPU_REVISION:COMMIT},
  });
  const build = read(join(f.page,'build.json'));
  assert.equal(build.commit,COMMIT);
  assert.equal(build.worktree,WORKTREE.replace(/\/$/,''));
  assert.equal(build.source,'scripts/build-lights.sh');
  assert.equal(typeof build.dirty,'boolean');
  assert.match(build.buildId,/^[\da-f-]{36}$/);
  passed(await cli(f,'render'));
});

test('missing build/commit provenance, wrong host revision and empty selection fail closed', async t => {
  for (const [mutate,env,pattern] of [
    [f => rmSync(join(f.page,'build.json')),{},/build\.json is missing/],
    [f => writeFileSync(join(f.page,'build.json'),'{}'),{},/missing build provenance/],
    [f => { const p=join(f.page,'build.json'),b=read(p); b.commit='not-a-commit'; writeFileSync(p,JSON.stringify(b)); },{},/stale build commit/],
    [() => {},{NMC_GPU_REVISION:'different-host-commit'},/host NMC_GPU_REVISION differs/],
    [f => writeFileSync(join(f.out,'selection.json'),JSON.stringify({page:f.page,pairs:[]})),{},/selection is empty/],
  ]) {
    const f=await transportFixture(t); mutate(f);
    failed(await cli(f,'render','ok',env),pattern);
    assert.equal(existsSync(f.trace),false,'no publication before valid provenance/selection');
    failed(await cli(f,'fresh','ok',env),pattern);
  }
});

test('fresh revalidates PNG CRC/scanlines and requires the producer receipt/job identity', async t => {
  for (const [mutate,pattern] of [
    [(f,r) => { const p=join(f.out,r.pairs[0].capture),b=readFileSync(p); b[b.length-5]^=1; writeFileSync(p,b); },/corrupt\/empty PNG/],
    [(f,r) => rmSync(join(f.out,r.pairs[0].result.replace('result.json','completion.json'))),/completion\.json is missing/],
    [(f,r) => { const p=join(f.out,r.pairs[0].result),d=read(p); d.job_id='previous-job'; writeFileSync(p,JSON.stringify(d)); },/managed (job|run) provenance/],
  ]) {
    const f=await transportFixture(t); passed(await cli(f,'render'));
    mutate(f,read(join(f.out,'run.json')));
    failed(await cli(f,'fresh'),pattern);
  }
});

test('a completed GPU job keeps its separate output download window', async t => {
  const f=await transportFixture(t);
  const clock=join(f.root,'clock.mjs');
  writeFileSync(clock,'const original=globalThis.setTimeout; globalThis.setTimeout=(fn,ms,...a)=>original(fn,ms===300000?20:ms,...a);');
  writeFileSync(f.client,CLIENT.replace('const measured =',
    "console.error('GPU job fake: running (0 s)'); console.error('GPU job fake: done'); await new Promise(r => setTimeout(r,40));\nconst measured ="));
  passed(await cli(f,'render','ok',{NODE_OPTIONS:`--import=${clock}`}));
  passed(await cli(f,'fresh'));
});

const JUDGE_CLIENT = String.raw`import json, os, sys
from pathlib import Path
capture, *args = sys.argv[1:]
assert Path(capture).is_file() and Path(capture).stat().st_size > 0
assert args[0] == '--description' and args[1]
assert args[2:] == ['--min-score', '6', '--aspect', 'functional']
assert 'prepass' not in args[1] and 'metre' not in args[1]
mode = os.environ.get('LIGHTS_TEST_JUDGE_CASE', 'ok')
if mode == 'no-verdict':
    print('judgment finished without a verdict')
elif mode == 'failed-score':
    print('{"score": 8}')
    sys.exit(1)
elif mode == 'one-low' and Path(capture).name.startswith('point-on-'):
    print('JUDGE FAIL: 5/10')
    sys.exit(1)
elif mode == 'score-low':
    print('JUDGE FAIL: 5/10')
else:
    print('{"score": 8}')
`;

test('judge fails closed across primary and attachment lifecycle captures', async t => {
  const ids = ['point','spot','directional','area','many-lights','attached-light','motion'];
  const f = await transportFixture(t, false, ids.map(id => scene(id)));
  f.judgeClient = join(f.root, 'judge.py');
  writeFileSync(f.judgeClient, JUDGE_CLIENT);
  passed(await cli(f, 'render'));

  const judged = await cli(f, 'judge');
  passed(judged);
  const record = read(join(f.out, 'judge.json'));
  assert.equal(record.complete, true);
  assert.equal(record.images.length, 14 + 18);
  assert.deepEqual(record.images.slice(0, 2).map(image => image.variant), ['on', 'off']);
  const lifecycle = record.images.filter(image => image.scene === 'attached-light');
  assert.equal(new Set(record.images.map(image => image.capture.split('/').at(-1))).size, 32);
  assert.ok(record.images.every(image => image.description && image.aspect === 'functional'));
  assert.ok(lifecycle.every(image => image.score === 8 && image.exitCode === 0));

  for (const [mode, pattern] of [
    ['score-low', /score 5 is outside/],
    ['failed-score', /judge process failed/],
    ['no-verdict', /no parseable score verdict/],
    ]) {
    failed(await cli(f, 'judge', 'ok', {LIGHTS_TEST_JUDGE_CASE:mode}), pattern);
    const rejected = read(join(f.out, 'judge.json'));
    assert.equal(rejected.complete, false);
    assert.equal(rejected.images.length, 32, 'every image is attempted despite earlier rejections');
    assert.equal(rejected.errors.length, 32);
  }

  failed(await cli(f, 'judge', 'ok', {LIGHTS_TEST_JUDGE_CASE:'one-low'}), /image judgment failures \(1\/32\)/);
  const oneFailure = read(join(f.out, 'judge.json'));
  assert.equal(oneFailure.images.length, 32);
  assert.equal(oneFailure.errors.length, 1);
  assert.equal(oneFailure.images.at(-1).score, 8);
  assert.match(oneFailure.images[0].stdout, /JUDGE FAIL/);

  const runPath = join(f.out, 'run.json'), savedRun = readFileSync(runPath, 'utf8');
  const run = JSON.parse(savedRun); run.complete = false; writeFileSync(runPath, JSON.stringify(run));
  failed(await cli(f, 'judge'), /unsuccessful or incomplete/);
  writeFileSync(runPath, savedRun);

  const buildPath = join(f.page, 'build.json'), savedBuild = readFileSync(buildPath, 'utf8');
  const build = JSON.parse(savedBuild); build.buildId = 'stale-build'; writeFileSync(buildPath, JSON.stringify(build));
  failed(await cli(f, 'judge'), /stale build commit provenance|stale\/missing run\/build\/commit provenance/);
  writeFileSync(buildPath, savedBuild);

  const attachedPair = read(runPath).pairs.find(pair => pair.scene === 'attached-light' && pair.variant === 'on');
  const stagePath = join(f.out, attachedPair.capture.replace(captureEvidence('attached-light','on').filename, captureEvidence('attached-light','on','rotated').filename));
  const savedStage = readFileSync(stagePath);
  rmSync(stagePath);
  failed(await cli(f, 'judge'), /missing lifecycle output: rotated\.png/);
  writeFileSync(stagePath, savedStage);
});


test('whole-flow diagnostics fail the run after every pair is collected in one job', async t => {
  const f = await transportFixture(t);
  failed(await cli(f, 'render', 'page-errors'), /pageerror: all frames still attempted/);
  const record = read(join(f.out, 'run.json'));
  const attempt = read(join(f.out, '.lights-current.json')).attempt;
  assert.equal(record.complete, false);
  assert.equal(record.pairs.length, 2);
  const batch = read(join(f.out, attempt, 'result.json'));
  assert.deepEqual(batch.result.steps.map(s => s.variant), ['on', 'off']);
  assert.deepEqual(readFileSync(f.trace, 'utf8').trim().split('\n').map(JSON.parse).map(c => c.op), ['publish', 'render']);
  failed(await cli(f, 'fresh'), /incomplete/);
  failed(await cli(f, 'render', 'missing-step'), /step count differs/);
  failed(await cli(f, 'fresh'), /incomplete/);
  failed(await cli(f, 'render', 'software'), /point\/on.*hardware adapter[\s\S]*point\/off.*hardware adapter/);
});


test('real staged metadata resolves functional labels and 32 unique filenames', async t => {
  const f = writePage(t, [scene()]);
  cpSync(new URL('../../examples/lights/scenes/', import.meta.url), join(f.page, 'scenes'), { recursive: true });
  mkdirSync(join(f.page, 'common'), { recursive: true });
  cpSync(new URL('./lights-evidence.js', import.meta.url), join(f.page, 'common/lights-evidence.js'));
  const script = readFileSync(new URL('../../scripts/build-lights.sh', import.meta.url), 'utf8');
  const imports = script.split("<<'IMPORTS'\n")[1].split('\nIMPORTS\n')[0];
  execFileSync(process.execPath, ['--input-type=module', '-', join(f.page, 'scenes')], { input: imports });
  const staging = script.split("<<'JS'\n")[1].split('\nJS\n')[0];
  execFileSync(process.execPath, ['--input-type=module', '-', f.page], { input: staging, cwd: WORKTREE });
  const m03 = scene('mama-produced-light', [FLOOR], 'm03');
  m03.variants.forEach(variant => { variant.evidence = { filename: `mama-produced-${variant.id}.png`, label: `MAMA-produced ${variant.id} light` }; });
  writeFileSync(join(f.page, 'scenes/mama-produced-light.js'), source(m03));
  const index = read(join(f.page, 'scenes.json'));
  index.push({ id: m03.id, milestone: m03.milestone, file: 'scenes/mama-produced-light.js' });
  writeFileSync(join(f.page, 'scenes.json'), JSON.stringify(index));
  const pairs = await check(f.page);
  assert.equal(pairs.length, 14);
  const images = pairs.flatMap(pair => [pair.evidence,
    ...(pair.scene === 'attached-light' ? ['initial','translated','rotated','reattached','detached',
      'reattached-before-delete','parent-deleted','light-removed','attached-final'].map(stage => captureEvidence(pair.scene,pair.variant,stage)) : []),
  ]);
  assert.equal(images.length, 32);
  assert.equal(new Set(images.map(image => image.filename)).size, 32);
  assert.ok(images.every(image => image.label && !/prepass|metre|intensity equality|dark background/.test(image.label)));
  const m03out = join(f.root, 'out/m03');
  await main(['select', '--page', f.page, '--out', m03out, '--milestone', 'm03']);
  const m03Pairs = read(join(m03out, 'selection.json')).pairs;
  assert.deepEqual(m03Pairs.map(pair => pair.evidence.filename), ['mama-produced-on.png', 'mama-produced-off.png']);
});
