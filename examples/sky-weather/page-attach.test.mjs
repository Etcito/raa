import test from 'node:test';
import assert from 'node:assert/strict';
import { discoverScenes, attachSkyPage } from './main.js';
import { createSkyPageRunner } from './lib/runtime.js';
import * as atmosphere from './scenes/atmosphere.js';

// Regression for run 13630: the managed host bridge answered 'Render operation unavailable'
// because window.nmcGpu existed with an empty ops object while built scenes were discovered.
// The atmosphere stage (current preset mappings and legacy comparison) must expose `run`
// in every published nmcGpu state, through the real discovery and the real page runner.

const index = [
  { id: 'atmosphere', milestone: 'atmosphere', file: 'scenes/atmosphere.js' },
  { id: 'foundation', milestone: 'foundation', file: 'scenes/foundation.js' },
];
const discoverBuilt = () => discoverScenes({
  fetchIndex: async () => ({ ok: true, json: async () => structuredClone(index) }),
  importScene: file => import(`./${file}`),
});

function pageElements() {
  const document = { createElement };
  function createElement(tag) {
    return {
      tag, ownerDocument: document, children: [], textContent: '', value: '', disabled: false, listeners: {},
      appendChild(child) { this.children.push(child); if (tag === 'select' && this.children.length === 1) this.value = child.value; },
      replaceChildren() { this.children = []; },
      addEventListener(name, handler) { this.listeners[name] = handler; },
    };
  }
  return {
    target: { crypto: { randomUUID: () => 'id' }, listeners: {}, addEventListener(name, handler) { this.listeners[name] = handler; } },
    canvas: createElement('canvas'), status: createElement('div'), picker: createElement('select'),
    controls: createElement('div'), runButton: createElement('button'),
  };
}

const input = mode => ({ selectionId: 's', attemptId: 'a', buildId: 'b', scenario: 'atmosphere', mode,
  width: 1280, height: 720, remainingMs: 270000 });

test('atmosphere page publishes the run operation from the first nmcGpu state and attaches the real runner', async () => {
  const page = pageElements();
  let runner = null;
  const preparation = attachSkyPage({ ...page, discover: discoverBuilt, load: async () => { throw new Error('no GPU in tests'); },
    createRunner: options => (runner = createSkyPageRunner(options)) });
  // Synchronously after attachment starts, before discovery resolves.
  assert.equal(page.target.nmcGpu.ready, false);
  assert.equal(typeof page.target.nmcGpu.ops.run, 'function');
  assert.equal(await preparation, runner);
  assert.equal(page.target.nmcGpu.ready, true);
  assert.equal(page.target.nmcGpu.ops.run, runner.run);
  assert.equal(typeof runner.run, 'function');
  assert.equal(runner.engine(), null);
  // Every current atmosphere mode, preset mappings and legacy comparison included, is offered.
  const modes = atmosphere.inventory.map(row => row.mode);
  assert.deepEqual(modes, ['day', 'twilight', 'night', 'later', 'snow', 'jungle', 'legacy']);
  const offered = page.picker.children.map(option => option.textContent);
  for (const mode of modes) assert.ok(offered.some(text => text.endsWith(`/ ${mode}`)), `picker lacks atmosphere/${mode}`);
  page.target.listeners.pagehide();
  runner.dispose();
});

test('a managed call arriving during discovery reaches the attached runner instead of failing as unavailable', async () => {
  const page = pageElements();
  let resolve;
  const pending = new Promise(r => { resolve = r; });
  const calls = [];
  const runner = { run: async value => { calls.push(value); return { mode: value.mode }; }, engine: () => null, dispose() {} };
  const preparation = attachSkyPage({ ...page, discover: () => pending, createRunner: () => runner });
  const early = page.target.nmcGpu.ops.run(input('legacy'));
  assert.equal(calls.length, 0);
  resolve(await discoverBuilt());
  await preparation;
  assert.deepEqual(await early, { mode: 'legacy' });
  assert.deepEqual(calls, [input('legacy')]);
  page.target.listeners.pagehide();
});

test('a managed call during a discovery that fails rejects with the originating error', async () => {
  const page = pageElements();
  let reject;
  const pending = new Promise((_, r) => { reject = r; });
  const preparation = attachSkyPage({ ...page, discover: () => pending });
  const early = page.target.nmcGpu.ops.run(input('day'));
  const cause = new Error('scene import failed');
  reject(cause);
  await assert.rejects(preparation, error => error === cause);
  await assert.rejects(early, error => error === cause);
  await assert.rejects(page.target.nmcGpu.ops.run(input('day')), error => error === cause);
});
