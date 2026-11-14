// node --test scripts/build-page.test.mjs: stage a throwaway fixture topic through the
// page-kit builder. No git command, cargo or GPU: captureInputs is a stub and every
// input is a small text file under a temporary root.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { buildPage, parseArgs } from './build-page.mjs';

const TOPIC = 'fixture';
const SHARED = 'examples/other/lib/shared-one.js';
const EXPECTED_FILES = [
  'common/nmc-gpu.js',
  'index.html',
  'lib/a.js',
  'main.js',
  'nmc.gpu.json',
  'raa/index.d.ts',
  'raa/index.js',
  'raa/pkg/raa_js_bg.wasm',
  'scenes.json',
  'scenes/one.js',
  'shared/capture.js',
  'shared/evidence.mjs',
  'shared/measure.js',
  'shared/readiness.js',
  'shared/shared-one.js',
];

function write(root, relative, text) {
  const file = path.join(root, relative);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, text);
}

async function fixture(t, { milestone = 'alpha', extraLib = null } = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'build-page-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  write(root, 'dist/package/raa/index.js', 'export {};\n');
  write(root, 'dist/package/raa/index.d.ts', 'export {};\n');
  write(root, 'dist/package/raa/pkg/raa_js_bg.wasm', 'wasm\n');
  write(root, 'examples/fixture/index.html', '<!doctype html>\n');
  write(root, 'examples/fixture/main.js', "import './lib/a.js';\n");
  write(root, 'examples/fixture/nmc.gpu.json', '{"page":"index.html","ops":[]}\n');
  write(root, 'examples/fixture/lib/a.js', [
    "import capture from '../../sky-weather/lib/capture.js';",
    "import gpu from '../../../web/common/nmc-gpu.js';",
    "import shared from '../../other/lib/shared-one.js';",
    '',
  ].join('\n'));
  write(root, 'examples/fixture/scenes/one.js', `export default { id: 'one', milestone: '${milestone}' };\n`);
  write(root, 'examples/other/lib/shared-one.js', 'export const sharedOne = true;\n');
  for (const name of ['capture.js', 'measure.js', 'readiness.js']) {
    write(root, `examples/sky-weather/lib/${name}`, 'export {};\n');
  }
  fs.copyFileSync(new URL('../examples/sky-weather/lib/evidence.mjs', import.meta.url),
    path.join(root, 'examples/sky-weather/lib/evidence.mjs'));
  write(root, 'web/common/nmc-gpu.js', 'export {};\n');
  if (extraLib) write(root, `examples/fixture/lib/${extraLib.name}`, extraLib.text);
  return root;
}

function build(root) {
  return buildPage({
    root,
    topic: TOPIC,
    stages: ['alpha'],
    shared: [SHARED],
    captureInputs: () => ({ stub: true }),
  });
}

test('stages the fixture layout and rewrites its imports', async (t) => {
  const root = await fixture(t);
  const { out, scenes, receipt } = await build(root);
  assert.equal(out, path.join(root, 'dist', TOPIC));
  assert.deepEqual(scenes, [{ id: 'one', milestone: 'alpha', file: 'scenes/one.js' }]);
  assert.deepEqual(receipt.files, EXPECTED_FILES);
  const manifest = JSON.parse(fs.readFileSync(path.join(out, 'scenes.json'), 'utf8'));
  assert.deepEqual(manifest, scenes);
  const staged = fs.readFileSync(path.join(out, 'lib/a.js'), 'utf8');
  const specifiers = [...staged.matchAll(/from\s*['"]([^'"]+)['"]/g)].map((match) => match[1]);
  assert.deepEqual(specifiers, [
    '../shared/capture.js',
    '../common/nmc-gpu.js',
    '../shared/shared-one.js',
  ]);
});

test('refuses a staged import that still names a source path', async (t) => {
  const root = await fixture(t, {
    extraLib: { name: 'b.js', text: "import extra from '../../sky-weather/lib-extra/x.js';\n" },
  });
  await assert.rejects(build(root), /still names a source path/);
  assert.equal(fs.existsSync(path.join(root, 'dist', TOPIC, 'build.json')), false);
});

test('refuses a scene whose milestone is not a requested stage', async (t) => {
  const root = await fixture(t, { milestone: 'gamma' });
  await assert.rejects(build(root), /is not one of/);
  assert.equal(fs.existsSync(path.join(root, 'dist', TOPIC, 'build.json')), false);
});

test('writes build.json last with a ready receipt', async (t) => {
  const root = await fixture(t);
  const { out, receipt } = await build(root);
  const buildFile = path.join(out, 'build.json');
  const manifest = JSON.parse(fs.readFileSync(buildFile, 'utf8'));
  assert.ok(fs.statSync(buildFile).mtimeMs >= fs.statSync(path.join(out, 'scenes.json')).mtimeMs);
  assert.equal(manifest.status, 'ready');
  assert.equal(manifest.error, null);
  assert.equal(manifest.files.includes('build.json'), false);
  assert.equal(manifest.files.includes('scenes.json'), true);
  assert.deepEqual(manifest.files, receipt.files);
  assert.deepEqual(manifest.inputs, { stub: true });
});

test('parseArgs refuses a missing --stages and an unknown option', () => {
  assert.throws(() => parseArgs([TOPIC]), { message: /^build-page: / });
  assert.throws(() => parseArgs([TOPIC, '--stages', 'alpha', '--unknown']), { message: /^build-page: / });
  assert.deepEqual(parseArgs([TOPIC, '--stages', 'alpha', '--shared', SHARED]), {
    topic: TOPIC,
    stages: ['alpha'],
    shared: [SHARED],
    root: null,
  });
});
