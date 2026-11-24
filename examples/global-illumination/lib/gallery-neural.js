import { digest } from './gallery-path-model.js';

const graphPath = 'shared/real-frame-model.json';
const optionId = 'neural-upscaling';
const original = 'v1';
const joint = 'joint-v2';
const packages = {
  [original]: { id: 'gallery-temporal', directory: 'lib/gallery-temporal-fit', heads: ['display'] },
  [joint]: { id: 'gallery-temporal-joint-v2', directory: 'lib/gallery-temporal-joint-fit', heads: ['display', 'reflection'] },
};

export const neuralOption = {
  id: optionId, label: 'Neural upscaling (pilot)', enabled: false, unavailable: neuralUnavailable,
};
export const neuralModelOption = {
  id: 'neural-model', label: 'Upscaling model', value: original,
  choices: [[original, 'Original Windows fit'], [joint, 'V2 SR + reflections (experimental)']],
};

function selectedModel(state) {
  return state?.neuralModels?.[state.values[neuralModelOption.id]];
}

function neuralUnavailable(state) {
  if (!selectedModel(state)) return state?.neuralError ?? state?.neuralModels?.jointError ?? 'The temporal pilot is unavailable.';
  if (state.enabled['path-reference'] || state.enabled['model-assisted-path']
      || state.values['gi-grid'] === 'spherical' || state.values['lighting-view'] !== 'combined') {
    return 'Requires combined raster lighting with the volume grid.';
  }
  return state.neuralError;
}

async function read(path) {
  const response = await fetch(new URL('../' + path, import.meta.url));
  if (!response.ok) throw new Error('Gallery neural asset ' + path + ' returned HTTP ' + response.status);
  return response;
}

export async function loadGalleryNeural(engine, { includeJoint = false } = {}) {
  if (typeof engine.registerNeuralModel !== 'function') return null;
  const receipt = await (await read('build.json')).json();
  if (!receipt.files?.includes(packages[original].directory + '/manifest.json')) return null;
  const graph = await (await read(graphPath)).json();

  async function load(key) {
    const { id, directory, heads } = packages[key];
    const manifestPath = directory + '/manifest.json';
    if (!receipt.files.includes(manifestPath)) return null;
    const [manifestFile, weightsFile] = await Promise.all([read(manifestPath), read(directory + '/weights.fp32le')]);
    const text = await manifestFile.text(), manifest = JSON.parse(text);
    const weights = new Uint8Array(await weightsFile.arrayBuffer());
    if (manifest.format !== 'raa.real-frame-model/1' || manifest.kind !== 'windows-capture-fit'
        || !heads.every(head => manifest.trainedHeads?.includes(head)) || manifest.weightsByteLength !== weights.byteLength
        || manifest.weightsSha256 !== await digest(weights)) throw new Error('Invalid trained gallery upscaling package');
    engine.registerNeuralModel(id, text, weights);
    return { id, graph, manifest, reflection: key === joint };
  }

  const stable = await load(original);
  if (!includeJoint) return stable;
  const models = { [original]: stable };
  try {
    models[joint] = await load(joint);
  } catch (error) {
    models.jointError = error.message;
  }
  return models;
}

function requireSelection(result) {
  if (!result.ok) throw new Error(result.issues.find(issue => issue.severity === 'error')?.message ?? 'Neural graph selection refused');
}

export function applyGalleryNeural(engine, state, width, height) {
  const requested = state.enabled[optionId];
  const available = !neuralUnavailable(state);
  const model = selectedModel(state);
  const size = requested && available ? model.id + ':' + width + 'x' + height : undefined;
  if (requested && !available) state.enabled[optionId] = false;
  if (size === state.neuralSize) return;
  try {
    if (!size) {
      engine.setNeuralReconstruction(null);
      requireSelection(engine.setRenderGraph(null));
    } else {
      const { id, graph, reflection } = model;
      requireSelection(engine.setRenderGraphSelection(graph, undefined, Math.ceil(width / 2), Math.ceil(height / 2)));
      engine.setNeuralReconstruction({ model: id, sr: true, reflection });
    }
    state.neuralSize = size;
  } catch (error) {
    engine.setNeuralReconstruction(null);
    requireSelection(engine.setRenderGraph(null));
    state.neuralSize = undefined;
    state.enabled[optionId] = false;
    state.neuralError = error.message;
  }
}

export function syncNeuralControls(toggle, select, state) {
  const reason = neuralUnavailable(state);
  toggle.disabled = Boolean(reason);
  toggle.checked = state.enabled[optionId];
  toggle.title = reason ?? 'Experimental 2× reconstruction';
  select.disabled = !state.neuralModels?.[original];
  select.value = state.values[neuralModelOption.id];
  select.options[1].disabled = !state.neuralModels?.[joint];
  select.title = state.neuralModels?.jointError ?? '';
}

export async function verifyGalleryNeuralUi() {
  const control = document.querySelector('[data-option="neural-upscaling"]');
  const models = document.querySelector('[data-option="neural-model"]');
  const status = () => document.getElementById('status').textContent;
  const text = () => document.getElementById('live-stats').textContent;
  if (!control || control.checked || control.disabled) throw new Error('Gallery neural control unavailable: ' + control?.title);
  const initial = models?.value;
  if (initial !== original) throw new Error('The gallery must open with the original model selected');
  const change = (element, key, value) => {
    element[key] = value;
    element.dispatchEvent(new Event('change', { bubbles: true }));
  };
  const line = prefix => text().split('\n').find(value => value.startsWith(prefix));
  async function wait(check, label) {
    const deadline = performance.now() + 25000;
    while (!check()) {
      if (status().startsWith('failed:') || performance.now() >= deadline) {
        throw new Error('Gallery neural ' + label + ' failed: ' + status() + ' ' + control.title);
      }
      await new Promise(resolve => setTimeout(resolve, 40));
    }
  }
  change(control, 'checked', true);
  const originalSha = (await (await read(packages[original].directory + '/manifest.json')).json()).weightsSha256.slice(0, 12);
  const originalLine = 'Neural model · gallery-temporal · ' + originalSha + ' · SR';
  await wait(() => /Neural upscaling · 320 × 180 → 640 × 360/.test(text()) &&
    line('Neural model · gallery-temporal ·') === originalLine, 'original activation');
  const active = line('Neural upscaling ·');
  let jointModel;
  if (!models.options[1].disabled) {
    const expected = (await (await read(packages[joint].directory + '/manifest.json')).json()).weightsSha256.slice(0, 12);
    change(models, 'value', joint);
    await wait(() => line('Neural model · gallery-temporal-joint-v2 · ' + expected + ' · SR + reflections'), 'V2 activation');
    jointModel = line('Neural model · gallery-temporal-joint-v2');
    change(models, 'value', original);
    await wait(() => line('Neural model · gallery-temporal ·') === originalLine, 'original restoration');
  }
  change(control, 'checked', false);
  await wait(() => !text().includes('Neural upscaling ·'), 'bypass');
  return { active, initial, firstModel: originalLine, jointModel, restored: models.value, disabled: !control.checked, status: status() };
}
