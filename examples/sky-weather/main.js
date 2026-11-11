import { createSkyPageRunner } from './lib/runtime.js';
import { validateInventory } from './lib/inventory.js';
import { buildSkySample, sampleSelection } from './lib/scene.js';

const stages = ['foundation', 'atmosphere', 'weather'];
const name = /^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$/;
const message = error => {
  const text = error?.message ?? String(error);
  return error?.code && !text.startsWith(`${error.code}:`) ? `${error.code}: ${text}` : text;
};
const lib = Object.freeze({ buildSkySample, sampleSelection });
const loadRaa = async () => (await import('./raa/index.js')).loadRaa();
const readBuildId = async () => {
  const response = await fetch('./build.json');
  if (!response.ok) throw new Error(`build.json: HTTP ${response.status}`);
  const receipt = await response.json();
  if (receipt?.status !== 'ready' || receipt.error !== null || typeof receipt.buildId !== 'string' || !receipt.buildId.trim()) {
    throw new Error('build.json: a ready build receipt is required');
  }
  return receipt.buildId;
};

// Import full namespaces, never just defaults: observe and the selector registry
// must survive discovery for the runner and browser-free selection to agree.
export async function discoverScenes({ fetchIndex = () => fetch('./scenes.json'), importScene = file => import(`./${file}`) } = {}) {
  const response = await fetchIndex();
  if (!response.ok) throw new Error(`scenes.json: HTTP ${response.status}`);
  const index = await response.json();
  if (!Array.isArray(index) || index.length === 0) throw new Error('scenes.json: nonempty array required');
  const scenes = new Map();
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
    // Each delivered stage validates its own exports; the final validation also
    // catches duplicate filenames, image identities and cross-scene pair errors.
    validateInventory([namespace], row.milestone);
    scenes.set(namespace.default.id, namespace);
  }
  const stage = stages[Math.max(...[...scenes.values()].map(ns => stages.indexOf(ns.default.milestone)))];
  validateInventory([...scenes.values()], stage);
  return scenes;
}

// No GPU work starts at module import. The managed operation and the Run button
// share the resident runner, while controls edit its actual accepted Engine state.
export async function attachSkyPage({ target, canvas, status, picker, controls, runButton,
  discover = discoverScenes, load = loadRaa, readBuild = readBuildId, createRunner = createSkyPageRunner }) {
  // The managed host bridge calls ops[op] as soon as window.nmcGpu exists; an ops object
  // without run made it answer 'Render operation unavailable' (run 13630, legacy job).
  // Publish run from the first moment: during discovery it waits for the attached runner
  // (or the originating discovery failure) and then delegates exactly once.
  let settle;
  const attached = new Promise(resolve => { settle = resolve; });
  const pendingRun = async input => (await attached)(input);
  target.nmcGpu = { ready: false, ops: { run: pendingRun } };
  status.textContent = 'pending: discovering built scenes';
  picker.disabled = true;
  runButton.disabled = true;
  try {
    const scenes = await discover();
    const runner = createRunner({ canvas, status, scenes, lib, load });
    target.nmcGpu.ops.run = runner.run;
    settle(runner.run);
    target.nmcGpu.ready = true;
    const selections = [];
    for (const namespace of scenes.values()) {
      for (const variant of namespace.default.variants) {
        if (!namespace.inventory.some(row => row.mode === variant.id)) continue;
        const option = picker.ownerDocument.createElement('option');
        option.value = String(selections.length);
        option.textContent = `${namespace.default.title} / ${variant.id}`;
        picker.appendChild(option);
        selections.push({ namespace, variant });
      }
    }
    let selected = null;
    let busy = false;
    const fields = [];
    const refresh = () => {
      const engine = runner.engine();
      for (const { control, input } of fields) {
        const value = control.read(engine);
        if (control.kind === 'checkbox') input.checked = value;
        else input.value = String(value);
      }
    };
    const render = () => {
      const engine = runner.engine();
      const { namespace, variant } = selected;
      const timeMs = variant.timeMs ?? 1000;
      namespace.default.frame?.(engine, timeMs, variant);
      engine.render(timeMs);
    };
    function showControls() {
      controls.replaceChildren();
      fields.length = 0;
      const { namespace, variant } = selected;
      for (const control of variant.controls ?? namespace.default.controls ?? []) {
        const document = controls.ownerDocument;
        const label = document.createElement('label');
        label.textContent = `${control.label}${control.unit ? ` (${control.unit})` : ''}`;
        const input = document.createElement(control.kind === 'select' ? 'select' : control.kind === 'button' ? 'button' : 'input');
        input.id = `control-${control.id}`;
        if (control.kind === 'select') {
          for (const choice of control.choices) {
            const option = document.createElement('option');
            option.value = choice.value; option.textContent = choice.label; input.appendChild(option);
          }
        } else if (control.kind === 'button') input.textContent = control.label;
        else { input.type = control.kind; if (control.kind === 'number') input.step = 'any'; }
        input.addEventListener(control.kind === 'button' ? 'click' : 'change', () => {
          if (busy) return;
          try {
            const value = control.kind === 'checkbox' ? input.checked : control.kind === 'number'
              ? (input.value.trim() === '' ? NaN : Number(input.value)) : input.value;
            control.apply(runner.engine(), value);
            refresh(); render();
            status.textContent = `ready: ${namespace.default.id}/${variant.id} — ${control.label} applied`;
          } catch (error) {
            // The setter owns atomic validation; restore from its accepted getter.
            refresh();
            status.textContent = `failed: ${message(error)}\nLast accepted settings remain displayed. Correct the value and retry.`;
          }
        });
        label.appendChild(input); controls.appendChild(label); fields.push({ control, input });
      }
      refresh();
    }
    async function runPicked() {
      if (busy) return;
      busy = true; picker.disabled = true; runButton.disabled = true;
      controls.replaceChildren();
      try {
        selected = selections[Number(picker.value)];
        const { namespace, variant } = selected;
        status.textContent = 'pending: reading current build receipt';
        const buildId = await readBuild();
        await runner.run({ selectionId: target.crypto.randomUUID(), attemptId: target.crypto.randomUUID(),
          buildId, scenario: namespace.default.id, mode: variant.id,
          width: 1280, height: 720, remainingMs: 270000 });
        showControls();
      } catch (error) { status.textContent = `failed: ${message(error)}`; }
      finally { busy = false; picker.disabled = false; runButton.disabled = false; }
    }
    picker.disabled = false; runButton.disabled = false;
    picker.addEventListener('change', runPicked);
    runButton.addEventListener('click', runPicked);
    // Direct managed calls retain runner.run's identity and still populate the UI
    // when the runner publishes its actual ready state.
    const observer = typeof target.MutationObserver === 'function' ? new target.MutationObserver(() => {
      const match = /^ready: ([^/]+)\/(\S+)$/.exec(status.textContent);
      if (match) {
        const index = selections.findIndex(item => item.namespace.default.id === match[1] && item.variant.id === match[2]);
        if (index >= 0) { selected = selections[index]; picker.value = String(index); showControls(); }
      } else if (status.textContent.startsWith('pending:')) controls.replaceChildren();
    }) : null;
    observer?.observe(status, { childList: true, characterData: true, subtree: true });
    target.addEventListener('pagehide', () => { observer?.disconnect(); runner.dispose(); }, { once: true });
    status.textContent = 'pending: scenes discovered; choose a scene and press Run to prepare the GPU';
    return runner;
  } catch (error) {
    status.textContent = `failed: ${message(error)}`;
    const rejectRun = async () => { throw error; };
    target.nmcGpu.ops.run = rejectRun;
    settle(rejectRun);
    target.nmcGpu.ready = true;
    throw error;
  }
}

if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  attachSkyPage({ target: window, canvas: document.getElementById('c'), status: document.getElementById('status'),
    picker: document.getElementById('picker'), controls: document.getElementById('controls'), runButton: document.getElementById('run') })
    .catch(() => {}); // The originating failure is displayed and remains the operation's rejection.
}
