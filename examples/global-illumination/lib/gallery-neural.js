import { digest } from './gallery-path-model.js';

const manifestPath = 'lib/gallery-temporal-fit/manifest.json';
const weightsPath = 'lib/gallery-temporal-fit/weights.fp32le';
const graphPath = 'shared/real-frame-model.json';
const optionId = 'neural-upscaling';

export const neuralOption = { id: optionId, label: 'Neural upscaling (pilot)', enabled: false,
  unavailable: state => neuralUnavailable(state) };

function neuralUnavailable(state) {
  if (!state?.neuralModel) return state?.neuralError ?? 'The temporal pilot is unavailable.';
  if (state.enabled['path-reference'] || state.enabled['model-assisted-path']
      || state.values['gi-grid'] === 'spherical' || state.values['lighting-view'] !== 'combined') {
    return 'Requires combined raster lighting with the volume grid.';
  }
  return state.neuralError;
}

export async function loadGalleryNeural(engine) {
  if (typeof engine.registerNeuralModel !== 'function') return null;
  const read = async path => {
    const response = await fetch(new URL(`../${path}`, import.meta.url));
    if (!response.ok) throw new Error(`Gallery neural asset ${path} returned HTTP ${response.status}`);
    return response;
  };
  const receipt = await (await read('build.json')).json();
  if (!receipt.files?.includes(manifestPath)) return null;
  const [manifestFile, weightsFile, graphFile] = await Promise.all([read(manifestPath), read(weightsPath), read(graphPath)]);
  const text = await manifestFile.text(), manifest = JSON.parse(text);
  const weights = new Uint8Array(await weightsFile.arrayBuffer());
  if (manifest.format !== 'raa.real-frame-model/1' || manifest.kind !== 'windows-capture-fit'
      || !manifest.trainedHeads?.includes('sr') || manifest.weightsByteLength !== weights.byteLength
      || manifest.weightsSha256 !== await digest(weights)) throw new Error('Invalid trained gallery upscaling package');
  const id = 'gallery-temporal';
  engine.registerNeuralModel(id, text, weights);
  return { id, graph: await graphFile.json() };
}

function requireSelection(result) {
  if (!result.ok) throw new Error(result.issues.find(issue => issue.severity === 'error')?.message ?? 'Neural graph selection refused');
}

export function applyGalleryNeural(engine, state, width, height) {
  const requested = state.enabled[optionId];
  const available = !neuralUnavailable(state);
  const size = requested && available ? `${width}x${height}` : undefined;
  if (requested && !available) state.enabled[optionId] = false;
  if (size === state.neuralSize) return;
  try {
    if (!size) {
      engine.setNeuralReconstruction(null);
      requireSelection(engine.setRenderGraph(null));
    } else {
      const { id, graph } = state.neuralModel;
      requireSelection(engine.setRenderGraphSelection(graph, undefined, Math.ceil(width / 2), Math.ceil(height / 2)));
      engine.setNeuralReconstruction({ model: id, sr: true });
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

export function syncNeuralControl(control, state) {
  const reason = neuralUnavailable(state);
  control.disabled = Boolean(reason);
  control.checked = state.enabled[optionId];
  control.title = reason ?? 'Experimental 2× reconstruction';
}

export async function verifyGalleryNeuralUi() {
  const selector = '[data-option="neural-upscaling"]';
  const control = document.querySelector(selector);
  const status = () => document.getElementById('status').textContent;
  const text = () => document.getElementById('live-stats').textContent;
  if (!control || control.checked || control.disabled) throw new Error(`Gallery neural control unavailable: ${control?.title}`);
  const toggle = checked => {
    control.checked = checked;
    control.dispatchEvent(new Event('change', { bubbles: true }));
  };
  async function wait(check, label) {
    const deadline = performance.now() + 25000;
    while (!check()) {
      if (status().startsWith('failed:') || performance.now() >= deadline) {
        throw new Error(`Gallery neural ${label} failed: ${status()} ${control.title}`);
      }
      await new Promise(resolve => setTimeout(resolve, 40));
    }
  }
  toggle(true);
  await wait(() => /Neural upscaling · 320 × 180 → 640 × 360/.test(text()), 'activation');
  const active = text().split('\n').find(line => line.startsWith('Neural upscaling ·'));
  toggle(false);
  await wait(() => !text().includes('Neural upscaling ·'), 'bypass');
  return { active, disabled: !control.checked, status: status() };
}
