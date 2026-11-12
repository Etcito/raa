import { validateInventory, validateRequiredFacts } from './inventory.js';
import { createReadiness } from './readiness.js';
import { collectFeatureMeasurement } from './measure.js';
import { captureCurrent } from './capture.js';

const MAX_REMAINING_MS = 300000;
const MEASUREMENT_WIDTH = 1280;
const MEASUREMENT_HEIGHT = 720;
const WARMUP_FRAMES = 30;
const DEFAULT_TIME_MS = 1000;

const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const finite = value => typeof value === 'number' && Number.isFinite(value);
const nonempty = value => typeof value === 'string' && value.trim().length > 0;

function fail(message) { throw new Error(`sky page runner: ${message}`); }

export function createOperationBudget(remainingMs, clock) {
  if (!finite(remainingMs) || remainingMs <= 0) fail('remainingMs must be a finite positive duration');
  if (!record(clock) || typeof clock.now !== 'function') fail('clock must provide now');
  const startedAt = clock.now();
  if (!finite(startedAt)) fail('clock.now() must return a finite millisecond timestamp');
  const deadlineMs = startedAt + remainingMs;
  return {
    deadlineMs,
    remainingMs: () => {
      const nowMs = clock.now();
      return finite(nowMs) ? Math.max(0, deadlineMs - nowMs) : 0;
    },
  };
}

function browserClock() {
  if (typeof globalThis.performance !== 'object' || !globalThis.performance ||
      typeof globalThis.performance.now !== 'function') fail('performance.now is unavailable');
  if (typeof globalThis.requestAnimationFrame !== 'function') fail('requestAnimationFrame is unavailable');
  return {
    now: () => globalThis.performance.now(),
    requestFrame: callback => globalThis.requestAnimationFrame(callback),
    cancelFrame: handle => {
      if (typeof globalThis.cancelAnimationFrame === 'function') globalThis.cancelAnimationFrame(handle);
    },
  };
}

function describeError(error) {
  const message = error && error.message ? error.message : String(error);
  const code = error && typeof error.code === 'string' ? error.code : '';
  return code && !message.startsWith(`${code}: `) ? `${code}: ${message}` : message;
}

const yieldTask = () => new Promise(resolve => setTimeout(resolve, 0));

function variantTimeMs(variant) {
  return variant && variant.timeMs !== undefined ? variant.timeMs : DEFAULT_TIME_MS;
}

function renderAt(engine, scene, variant, timeMs) {
  if (typeof scene.frame === 'function') scene.frame(engine, timeMs, variant);
  engine.render(timeMs);
}

function applyPasses(engine, passes) {
  if (!passes) return;
  for (const name of passes.disable ?? []) engine.setPassEnabled(name, false);
  if (passes.order) engine.setPassOrder(passes.order);
}

function setCanvasSize(canvas, width, height) {
  canvas.width = width;
  canvas.height = height;
  if (canvas.style) {
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
  }
}

function validateInput(input) {
  if (!record(input)) fail('input must be an object');
  for (const name of ['selectionId', 'attemptId', 'buildId', 'scenario', 'mode']) {
    if (!nonempty(input[name])) fail(`input.${name} must be a nonempty string`);
  }
  for (const name of ['width', 'height']) {
    if (!Number.isSafeInteger(input[name]) || input[name] <= 0) fail(`input.${name} must be a positive integer pixel count`);
  }
  if (!finite(input.remainingMs) || input.remainingMs <= 0 || input.remainingMs > MAX_REMAINING_MS) {
    fail(`input.remainingMs must be a finite positive duration of at most ${MAX_REMAINING_MS} ms`);
  }
}

function resolveSelection(scenes, input) {
  if (!(scenes instanceof Map)) fail('options.scenes must be a Map');
  let entry = scenes.get(input.scenario);
  if (entry === undefined) {
    for (const candidate of scenes.values()) {
      if (record(candidate) && record(candidate.default) && candidate.default.id === input.scenario) {
        entry = candidate;
        break;
      }
    }
  }
  if (entry === undefined) fail(`unknown scenario ${JSON.stringify(input.scenario)}`);
  if (!record(entry)) fail(`scenario ${input.scenario}: the scene namespace must be an object`);
  const scene = entry.default;
  if (!record(scene) || scene.id !== input.scenario) fail(`scenario ${input.scenario}: namespace default.id must match`);
  const rows = validateInventory([entry], scene.milestone);
  const variant = Array.isArray(scene.variants) ? scene.variants.find(candidate => candidate?.id === input.mode) : null;
  if (!variant) fail(`scenario ${input.scenario} has no mode ${JSON.stringify(input.mode)}`);
  const selected = rows.filter(row => row.scenario === input.scenario && row.mode === input.mode);
  if (selected.length === 0) fail(`no inventory image for ${input.scenario}/${input.mode}`);
  return { entry, scene, variant, rows: selected };
}

function viewSizesFor(engine, canvas) {
  const width = canvas.width;
  const height = canvas.height;
  if (!Number.isSafeInteger(width) || width <= 0 || !Number.isSafeInteger(height) || height <= 0) {
    fail('canvas must have positive integer pixel dimensions');
  }
  if (typeof engine.viewports === 'function') {
    const rects = engine.viewports();
    if (Array.isArray(rects) && rects.length > 0) {
      return rects.map((rect, view) => {
        if (!record(rect) || !finite(rect.width) || !finite(rect.height) || rect.width <= 0 || rect.height <= 0) {
          fail('engine.viewports() must return positive viewport rectangles');
        }
        const viewWidth = Math.round(rect.width * width);
        const viewHeight = Math.round(rect.height * height);
        if (viewWidth <= 0 || viewHeight <= 0) fail('engine.viewports() produced an empty view');
        return { view, width: viewWidth, height: viewHeight };
      });
    }
  }
  return [{ view: 0, width, height }];
}

function adapterFacts(value) {
  const source = record(value) ? value : {};
  const text = key => typeof source[key] === 'string' ? source[key] : '';
  return {
    vendor: text('vendor'),
    architecture: text('architecture'),
    device: text('device'),
    description: text('description'),
    isFallbackAdapter: source.isFallbackAdapter === true,
  };
}

function browserFacts() {
  const source = typeof globalThis.navigator === 'object' && globalThis.navigator ? globalThis.navigator : {};
  const ratio = finite(globalThis.devicePixelRatio) && globalThis.devicePixelRatio > 0 ? globalThis.devicePixelRatio : 1;
  return {
    userAgent: typeof source.userAgent === 'string' ? source.userAgent : '',
    platform: typeof source.platform === 'string' ? source.platform : '',
    devicePixelRatio: ratio,
  };
}

export function createSkyPageRunner(options) {
  if (!record(options)) fail('options must be an object');
  if (!record(options.canvas)) fail('options.canvas is required');
  if (!record(options.status)) fail('options.status is required');
  if (typeof options.load !== 'function') fail('options.load must be a function');
  if (!(options.scenes instanceof Map)) fail('options.scenes must be a Map');
  if (!Object.hasOwn(options, 'lib')) fail('options.lib is required');

  const canvas = options.canvas;
  const status = options.status;
  const readiness = createReadiness('page');
  const errors = { console: [], page: [], gpu: [] };
  let engine = null;
  let raa = null;
  let running = false;
  let disposed = false;
  let restoreGpu = null;
  let deviceAdapter = null;
  let pageListening = false;

  const onWindowError = event => {
    const error = event && event.error !== undefined ? event.error : event;
    errors.page.push(error && error.message ? error.message : String(error));
  };
  const onUnhandledRejection = event => {
    const error = event && event.reason !== undefined ? event.reason : event;
    errors.page.push(error && error.message ? error.message : String(error));
  };

  if (typeof globalThis.window === 'object' && globalThis.window &&
      typeof globalThis.window.addEventListener === 'function') {
    globalThis.window.addEventListener('error', onWindowError);
    globalThis.window.addEventListener('unhandledrejection', onUnhandledRejection);
    pageListening = true;
  }

  function installGpuErrors() {
    if (typeof globalThis.navigator !== 'object' || !globalThis.navigator) return;
    const gpu = globalThis.navigator.gpu;
    if (!gpu || typeof gpu.requestAdapter !== 'function' || gpu.__nmcSkyWeatherErrors) return;
    const original = gpu.requestAdapter;
    const requestAdapter = original.bind(gpu);
    gpu.requestAdapter = async descriptor => {
      const adapter = await requestAdapter(descriptor);
      if (!adapter || typeof adapter.requestDevice !== 'function') return adapter;
      const requestDevice = adapter.requestDevice.bind(adapter);
      adapter.requestDevice = async deviceDescriptor => {
        const device = await requestDevice(deviceDescriptor);
        // wgpu 30 retains description but drops the browser's vendor/architecture.
        // Chrome may leave description blank. Preserve the SAME adapter that
        // created the Engine device; a separate request is not its provenance.
        const info = adapter.info;
        if (info) deviceAdapter = adapterFacts({
          vendor: info.vendor, architecture: info.architecture,
          device: info.device, description: info.description,
          isFallbackAdapter: info.isFallbackAdapter ?? adapter.isFallbackAdapter,
        });
        if (device && typeof device.addEventListener === 'function') {
          device.addEventListener('uncapturederror', event => {
            const error = event && event.error !== undefined ? event.error : event;
            errors.gpu.push(error && error.message ? error.message : String(error));
          });
        }
        // A lost device (GPU process reset) blanks every canvas; only an explicit dispose is expected.
        if (device && device.lost && typeof device.lost.then === 'function') {
          device.lost.then(info => {
            if (info && info.reason === 'destroyed') return;
            errors.gpu.push(`GPU device lost: ${info && info.reason ? info.reason : 'unknown'}: ${info && info.message ? info.message : ''}`.trim());
          }, () => {});
        }
        return device;
      };
      return adapter;
    };
    gpu.__nmcSkyWeatherErrors = true;
    restoreGpu = () => {
      gpu.requestAdapter = original;
      delete gpu.__nmcSkyWeatherErrors;
      restoreGpu = null;
    };
  }

  installGpuErrors();

  function installConsoleErrors() {
    if (typeof console !== 'object' || !console || typeof console.error !== 'function') return () => {};
    const original = console.error;
    const wrapper = (...args) => {
      errors.console.push(args.map(value => value && value.message ? value.message : String(value)).join(' '));
      original.apply(console, args);
    };
    console.error = wrapper;
    return () => { if (console.error === wrapper) console.error = original; };
  }

  function showStatus(text) { status.textContent = text; }

  function checkBudget(budget, label) {
    if (disposed) fail('runner is disposed');
    if (budget.remainingMs() <= 0) fail(`operation deadline expired during ${label}`);
  }

  async function loadModule() {
    if (!raa) {
      const loaded = await options.load();
      if (!record(loaded) || typeof loaded.createEngine !== 'function') fail('load() must resolve a RaaModule with createEngine');
      raa = loaded;
    }
    return raa;
  }

  async function createEngineFor(scene, variant, width, height) {
    if (engine) {
      const previous = engine;
      engine = null;
      previous.dispose();
    }
    setCanvasSize(canvas, width, height);
    const module = await loadModule();
    deviceAdapter = null;
    const created = await module.createEngine(canvas, { ...(scene.engine ?? {}), ...(variant.engine ?? {}) });
    if (disposed) {
      if (created && typeof created.dispose === 'function') created.dispose();
      fail('runner is disposed');
    }
    if (!record(created) || typeof created.resize !== 'function' || typeof created.render !== 'function' ||
        typeof created.stats !== 'function' || typeof created.passes !== 'function' || typeof created.dispose !== 'function') {
      fail('createEngine did not resolve a usable Engine');
    }
    engine = created;
    created.resize(width, height);
    return created;
  }

  function factsFor(observation, target) {
    return {
      ...observation,
      passes: target.passes(),
      viewSizes: viewSizesFor(target, canvas),
      readiness: readiness.snapshot(),
    };
  }

  // Readiness, publication and submittedInvocation are CPU-side submission facts: the GPU may
  // still be executing the prepared work (atmosphere LUTs, Milky Way bake, the unpaced warm-up
  // and observation frames). Chrome delivers no requestAnimationFrame while that backlog drains.
  // Modes whose observation awaits the IBL readback drain it there; env-off (the cleared
  // environment never publishes) settled at once and started the 30/300 rAF collector behind
  // seconds of GPU work, so warm-up frame 1 never arrived (run 13659). probeNonEmpty resolves
  // from a map readback queued after every earlier submission, i.e. once the GPU completed it.
  async function awaitGpuCompletion(target, scene, variant, budget) {
    if (typeof target.probeNonEmpty !== 'function') fail('engine.probeNonEmpty is required to await GPU completion');
    const remaining = budget.remainingMs();
    if (remaining <= 0) fail('operation deadline expired before the GPU completed the prepared frames');
    let timer = null;
    const expired = new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(
        `sky page runner: the GPU did not complete the prepared frames within the remaining ${Math.round(remaining)} ms of the operation deadline`)), remaining);
    });
    try {
      const timeMs = variantTimeMs(variant);
      if (typeof scene.frame === 'function') scene.frame(target, timeMs, variant);
      await Promise.race([target.probeNonEmpty(timeMs), expired]);
    } finally {
      clearTimeout(timer);
    }
  }

  async function collectMeasurements(target, scene, variant, clock, budget) {
    if (variant.measurement === undefined) return [];
    if (canvas.width !== MEASUREMENT_WIDTH || canvas.height !== MEASUREMENT_HEIGHT) {
      fail(`feature measurement requires a ${MEASUREMENT_WIDTH}x${MEASUREMENT_HEIGHT} canvas at device pixel ratio 1`);
    }
    checkBudget(budget, 'feature measurement');
    await awaitGpuCompletion(target, scene, variant, budget);
    checkBudget(budget, 'feature measurement');
    const measurement = await collectFeatureMeasurement(
      () => renderAt(target, scene, variant, variantTimeMs(variant)),
      { ...variant.measurement, deadlineMs: budget.deadlineMs },
      clock,
    );
    checkBudget(budget, 'feature measurement');
    return [measurement];
  }

  async function run(input) {
    if (disposed) fail('runner is disposed');
    if (running) fail('a run is already in progress');
    const restoreConsole = installConsoleErrors();
    try {
      let clock = null;
      let budget = null;
      try {
        validateInput(input);
        clock = browserClock();
        budget = createOperationBudget(input.remainingMs, clock);
      } catch (error) {
        showStatus(`failed: ${describeError(error)}`);
        throw error;
      }
      errors.console.length = 0;
      errors.page.length = 0;
      errors.gpu.length = 0;
      running = true;
      try {
        let selection = null;
        try {
          selection = resolveSelection(options.scenes, input);
        } catch (error) {
          readiness.fail('selection', error);
          showStatus(`failed: ${describeError(error)}`);
          throw error;
        }
        const stage = selection.scene.milestone;
        readiness.pending(stage);
        showStatus(`pending: preparing ${input.scenario}/${input.mode}`);
        try {
          const target = await createEngineFor(selection.scene, selection.variant, input.width, input.height);
          const stats = target.stats();
          if (!record(stats)) fail('engine.stats() must return frame statistics');
          const adapter = deviceAdapter ?? adapterFacts(stats.adapter);
          checkBudget(budget, 'Engine creation');
          await selection.scene.build(target, options.lib, selection.variant);
          checkBudget(budget, 'scene preparation');
          applyPasses(target, selection.variant.passes);
          for (let index = 0; index < WARMUP_FRAMES; index++) {
            checkBudget(budget, `warm-up frame ${index + 1}`);
            renderAt(target, selection.scene, selection.variant, variantTimeMs(selection.variant));
            await yieldTask();
          }
          const observation = await selection.entry.observe(target, selection.variant);
          checkBudget(budget, 'scene observation');
          if (!record(observation)) fail('observe must resolve a SceneObservation object');
          readiness.ready();
          showStatus(`ready: ${input.scenario}/${input.mode}`);
          const facts = factsFor(observation, target);
          for (const row of selection.rows) validateRequiredFacts(row, facts, selection.entry.factNames);
          const measurements = await collectMeasurements(target, selection.scene, selection.variant, clock, budget);
          checkBudget(budget, 'capture');
          renderAt(target, selection.scene, selection.variant, variantTimeMs(selection.variant));
          const data = await captureCurrent(canvas, status);
          checkBudget(budget, 'capture');
          return {
            provenance: {
              selectionId: input.selectionId,
              attemptId: input.attemptId,
              buildId: input.buildId,
              scenario: input.scenario,
              mode: input.mode,
            },
            browser: browserFacts(),
            adapter,
            hardware: stats.hardware === true && !adapter.isFallbackAdapter &&
              ![adapter.vendor, adapter.architecture, adapter.device, adapter.description].some(value =>
                /software|swiftshader|llvmpipe|lavapipe|softpipe|fallback|basic render|microsoft basic|warp\b/i.test(value)),
            images: selection.rows.map(row => ({
              id: row.id,
              filename: row.filename,
              mime: 'image/png',
              data,
              observation: row.expectedVisibleBehavior,
            })),
            facts: factsFor(observation, target),
            measurements,
            errors: {
              console: [...errors.console],
              page: [...errors.page],
              gpu: [...errors.gpu],
            },
          };
        } catch (error) {
          readiness.fail(stage, error);
          showStatus(`failed: ${describeError(error)}`);
          throw error;
        }
      } finally {
        running = false;
      }
    } finally {
      restoreConsole();
    }
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    if (pageListening && globalThis.window) {
      globalThis.window.removeEventListener('error', onWindowError);
      globalThis.window.removeEventListener('unhandledrejection', onUnhandledRejection);
      pageListening = false;
    }
    if (restoreGpu) restoreGpu();
    const current = engine;
    engine = null;
    if (current && typeof current.dispose === 'function') current.dispose();
  }

  return {
    run,
    engine: () => engine,
    dispose,
  };
}
