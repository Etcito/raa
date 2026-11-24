import { registerNmcGpu } from '../../web/common/nmc-gpu.js';
import * as content from './lib/content.js';
import { validateInventory } from './lib/inventory.js';
import { captureAreaPilot, createGiPageRunner } from './lib/runtime.js';
import { runNeuralBaseline } from './lib/neural-baseline-runner.js';

const stages = ['volume', 'lighting', 'virtual'];
const name = /^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$/;
const message = error => {
  const text = error?.message ?? String(error);
  return error?.code && !text.startsWith(`${error.code}:`) ? `${error.code}: ${text}` : text;
};
const id = () => globalThis.crypto.randomUUID();
// The staged package is the only engine source; the import stays dynamic so node can
// import this page module for discovery metadata without the built brick present.
const loadRaa = async () => (await import('./raa/index.js')).loadRaa();

// The two stage modules read room geometry and the shared virtual-geometry content
// through this namespace; nothing is built until the runner calls a scene's build.
function createLib(raa) {
  return { raa, content };
}

// Import full namespaces, never just defaults: observe, inventory and factNames survive
// discovery so the runner and the page's picker see the same modules.
export async function discoverScenes({ fetchIndex = () => fetch('./scenes.json'), importScene = file => import(`./${file}`) } = {}) {
  const response = await fetchIndex();
  if (!response.ok) throw new Error(`scenes.json: HTTP ${response.status}`);
  const index = await response.json();
  if (!Array.isArray(index) || index.length === 0) throw new Error('scenes.json: nonempty array required');
  const namespaces = [];
  let previous = '';
  for (const row of index) {
    if (!row || !name.test(row.id ?? '') || !stages.includes(row.milestone)
        || typeof row.file !== 'string' || !/^scenes\/[A-Za-z0-9_-]+\.js$/.test(row.file)
        || row.id <= previous) throw new Error(`scenes.json: invalid, duplicate or unsorted entry ${row?.id}`);
    previous = row.id;
    const namespace = await importScene(row.file);
    if (namespace?.default?.id !== row.id || namespace.default.milestone !== row.milestone) {
      throw new Error(`scenes.json: namespace identity mismatch for ${row.id}`);
    }
    namespaces.push(namespace);
  }
  // The highest stage present: virtual when a scene has it, else lighting, else volume.
  // This throw is shown in the status element by attachGiPage and rethrown, never swallowed.
  const stage = stages[Math.max(...namespaces.map(namespace => stages.indexOf(namespace.default.milestone)))];
  validateInventory(namespaces, stage);
  return new Map(namespaces.map(namespace => [namespace.default.id, namespace]));
}

// No GPU work starts at module import. The managed operation and the picker share the
// resident runner; window.nmcGpu is created by the shared helper only once run exists.
export async function attachGiPage({ target, canvas, status, scenePicker, variantPicker, runButton,
  baselineButton, baselineDownload,
  discover = discoverScenes, load = loadRaa, createRunner = createGiPageRunner }) {
  let runner = null;
  let busy = false;
  let downloadUrl = null;
  status.textContent = 'pending: discovering built scenes';
  try {
    const scenes = await discover();
    const lib = createLib(await load());
    const newRunner = () => createRunner({ canvas, status, load, scenes, lib });
    runner = newRunner();
    const controls = [scenePicker, variantPicker, runButton, baselineButton].filter(Boolean);
    async function exclusively(work) {
      if (busy) throw new Error('A run is already in progress');
      busy = true;
      for (const control of controls) control.disabled = true;
      try {
        return await work();
      } finally {
        busy = false;
        for (const control of controls) control.disabled = false;
      }
    }
    const run = input => exclusively(() => {
      runner ??= newRunner();
      return runner.run(input);
    });
    const baseline = input => exclusively(() => {
      runner?.dispose();
      runner = null;
      return runNeuralBaseline({ target, canvas, status, raa: lib.raa }, input?.args ?? input);
    });
    registerNmcGpu({ run });
    target.nmcGpu.ops.baseline = async input => {
      try {
        return { json: await baseline(input), files: [] };
      } catch (error) {
        if (!busy) status.textContent = `failed: ${message(error)}`;
        throw error;
      }
    };
    target.nmcGpu.ops['area-pilot'] = input => exclusively(() => {
      runner?.dispose();
      runner = null;
      return captureAreaPilot({ target, canvas, raa: lib.raa }, input?.args ?? input);
    });
    const entries = [...scenes.values()];
    for (const namespace of entries) {
      const option = target.document.createElement('option');
      option.value = namespace.default.id;
      option.textContent = namespace.default.title ?? namespace.default.id;
      scenePicker.appendChild(option);
    }
    function fillVariants() {
      const namespace = scenes.get(scenePicker.value);
      variantPicker.replaceChildren();
      for (const variant of namespace.default.variants) {
        if (!namespace.inventory.some(row => row.mode === variant.id)) continue;
        const option = target.document.createElement('option');
        option.value = variant.id;
        option.textContent = variant.id;
        variantPicker.appendChild(option);
      }
    }
    async function runPicked() {
      if (busy || variantPicker.value === '') return;
      try {
        await run({
          selectionId: id(), attemptId: id(), buildId: id(),
          scenario: scenePicker.value, mode: variantPicker.value,
          width: 1280, height: 720, remainingMs: 300000,
        });
      } catch (error) {
        // The runner already published `failed: <message>`; keep the originating text.
        status.textContent = `failed: ${message(error)}`;
      }
    }
    baselineButton?.addEventListener('click', async () => {
      if (busy) return;
      if (downloadUrl !== null) {
        target.URL.revokeObjectURL(downloadUrl);
        downloadUrl = null;
      }
      baselineDownload.hidden = true;
      try {
        const trials = await baseline({ mode: 'keyboard' });
        downloadUrl = target.URL.createObjectURL(new Blob([JSON.stringify(trials, null, 2)], { type: 'application/json' }));
        baselineDownload.href = downloadUrl;
        baselineDownload.hidden = false;
      } catch (error) {
        status.textContent = `failed: ${message(error)}`;
      }
    });
    scenePicker.value = entries[0].default.id;
    fillVariants();
    for (const control of controls) control.disabled = false;
    scenePicker.addEventListener('change', () => { fillVariants(); runPicked(); });
    variantPicker.addEventListener('change', runPicked);
    runButton.addEventListener('click', runPicked);
    target.addEventListener('pagehide', () => {
      runner?.dispose();
      if (downloadUrl !== null) target.URL.revokeObjectURL(downloadUrl);
    }, { once: true });
    status.textContent = 'pending: choose a scene and variant to prepare the GPU';
    return runner;
  } catch (error) {
    status.textContent = `failed: ${message(error)}`;
    throw error;
  }
}

if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  // The rejection stays visible: the status element shows it and the promise is not caught.
  attachGiPage({ target: window, canvas: document.getElementById('c'), status: document.getElementById('status'),
    scenePicker: document.getElementById('scene'), variantPicker: document.getElementById('variant'),
    runButton: document.getElementById('run'), baselineButton: document.getElementById('baseline'),
    baselineDownload: document.getElementById('baseline-download') });
}
