// Publish alongside common/ (web/common) and raa-anim/ (dist/bricks/raa-anim).
// The shared helper exposes run(args) and the GPU client's render({args}) envelope.
import { registerNmcGpu } from './common/nmc-gpu.js';

const status = document.getElementById('status');
const adapterText = document.getElementById('adapter');
const resultsText = document.getElementById('results');
const errorText = document.getElementById('error');
const HARDWARE_DENY = /swiftshader|llvmpipe|lavapipe|software|basic render/i;
let resident;
let gpuFailure;

function showError(error) {
  status.textContent = 'Failed';
  resultsText.textContent = '';
  errorText.textContent = String(error?.message || error);
}

function setup() {
  // Cache the promise as well as the resources, including concurrent first calls.
  return resident ??= (async () => {
    if (!navigator.gpu) throw new Error('WebGPU is unavailable; use Chrome with WebGPU enabled.');
    const gpuAdapter = await navigator.gpu.requestAdapter();
    if (!gpuAdapter) throw new Error('WebGPU did not provide an adapter.');
    const info = gpuAdapter.info ?? await gpuAdapter.requestAdapterInfo();
    const isFallbackAdapter = info.isFallbackAdapter ?? gpuAdapter.isFallbackAdapter;
    if (typeof isFallbackAdapter !== 'boolean') {
      throw new Error('WebGPU adapter did not report its fallback status.');
    }
    const adapter = {
      vendor: info.vendor,
      architecture: info.architecture,
      device: info.device,
      description: info.description,
      isFallbackAdapter,
    };
    const labels = [adapter.vendor, adapter.architecture, adapter.device, adapter.description];
    if (!labels.every(value => typeof value === 'string')) {
      throw new Error('WebGPU adapter information is incomplete.');
    }
    const hardware = !isFallbackAdapter && !labels.some(value => HARDWARE_DENY.test(value));
    adapterText.textContent = `Adapter classification: ${hardware ? 'hardware' : 'software'}\n`
      + JSON.stringify(adapter, null, 2);
    const device = await gpuAdapter.requestDevice();
    device.lost.then(info => {
      gpuFailure = new Error(`WebGPU device lost: ${info.message}`);
      showError(gpuFailure);
    });
    device.addEventListener('uncapturederror', event => {
      gpuFailure = new Error(`WebGPU error: ${event.error.message}`);
      showError(gpuFailure);
    });
    // Dynamic import keeps the operation registered and package failures visible.
    const { loadRaaAnim } = await import('./raa-anim/index.js');
    const api = await loadRaaAnim();
    const brick = api.brickInfo();
    if (Object.keys(brick).length !== 2 || brick.name !== 'raa-anim' || brick.version !== '0.5.0') {
      throw new Error('Unexpected animation brick metadata; expected raa-anim 0.5.0.');
    }
    return { gpuAdapter, device, adapter, hardware, api, brick };
  })();
}

function isU32(value) {
  return Number.isInteger(value) && value >= 0 && value <= 0xffffffff;
}

function validate(args) {
  if (!args || typeof args !== 'object' || Array.isArray(args)
      || args.case !== 'det-selftest' || !isU32(args.ticks)
      || !Array.isArray(args.seeds) || args.seeds.length === 0
      || !Array.from(args.seeds).every(isU32)) {
    throw new TypeError('Expected {case:"det-selftest",seeds:[u32, ...],ticks:u32}; seeds must be nonempty.');
  }
  return { case: args.case, seeds: [...args.seeds], ticks: args.ticks };
}

async function run(args) {
  try {
    errorText.textContent = '';
    resultsText.textContent = '';
    const request = validate(args);
    status.textContent = 'Running browser WebAssembly…';
    const { adapter, hardware, api, brick } = await setup();
    if (gpuFailure) throw gpuFailure;
    const results = request.seeds.map(seed => {
      const result = api.selftest(request.case, seed, request.ticks);
      if (!result || Object.keys(result).sort().join(',') !== 'case,hash,seed,ticks'
          || result.case !== request.case || result.seed !== seed || result.ticks !== request.ticks
          || typeof result.hash !== 'string' || !/^[0-9a-f]{8}$/.test(result.hash)) {
        throw new Error('Animation brick returned an invalid selftest record.');
      }
      return result;
    });
    resultsText.textContent = JSON.stringify(results, null, 2);
    status.textContent = `${brick.name} ${brick.version}: ${results.length} browser-computed hash(es)`
      + `; adapter is ${hardware ? 'hardware' : 'software'}.`;
    return { ...request, brick, adapter, hardware, results };
  } catch (error) {
    showError(error);
    throw error;
  }
}

registerNmcGpu({ run });
document.getElementById('request').addEventListener('submit', async event => {
  event.preventDefault();
  try {
    await run(JSON.parse(document.getElementById('input').value));
  } catch (error) {
    showError(error);
  }
});
setup().then(() => {
  if (!gpuFailure && status.textContent === 'Loading diagnostic…') status.textContent = 'Ready';
}).catch(showError);
