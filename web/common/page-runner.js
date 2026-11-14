// Shares 1 and 2 of contract 'page-runner': the constants, the standalone helpers and the runner.
// Ported from examples/virtual-geometry/lib/runtime.js. The throwing helpers come from
// helperSet(prefix), so a standalone call refuses with 'page runner: <message>' while
// createPageRunner passes helperSet('<topic> runtime'). No top-level side effect.
import { createReadiness } from '../../examples/sky-weather/lib/readiness.js';
import { collectFeatureMeasurement } from '../../examples/sky-weather/lib/measure.js';
import { captureCurrent } from '../../examples/sky-weather/lib/capture.js';

export const MAX_REMAINING_MS = 300000;
export const WARMUP_FRAMES = 30;
export const DEFAULT_TIME_MS = 1000;

const INPUT_KEYS = ['selectionId', 'attemptId', 'buildId', 'scenario', 'mode', 'width', 'height', 'remainingMs'];
const SOFTWARE_ADAPTER = /software|swiftshader|llvmpipe|lavapipe|softpipe|fallback|basic render|microsoft basic|warp\b/i;

const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const finite = value => typeof value === 'number' && Number.isFinite(value);
const nonblank = value => typeof value === 'string' && value.trim().length > 0;

// Handoff: helperSet(prefix) is called by page-kit: page-runner.js share 2, createPageRunner, which
// calls helperSet(topic + ' runtime') so that its run refusals carry '<topic> runtime: '. Today the
// only call is the module export at the end of this file, with 'page runner'.
function helperSet(prefix) {
  const fail = message => { throw new Error(prefix + ': ' + message); };

  // Handoff: createOperationBudget(remainingMs, clock) and validateInput(input) are called directly by
  // page-kit: page-runner.test.mjs, which passes its own clock. Today createOperationBudget refuses a
  // remainingMs outside (0, MAX_REMAINING_MS] and a clock without now(), reads clock.now() once and
  // returns { deadlineMs, remainingMs() }; validateInput keeps no state. Nothing crosses but the clock.
  function createOperationBudget(remainingMs, clock) {
    if (!finite(remainingMs) || remainingMs <= 0 || remainingMs > MAX_REMAINING_MS) {
      fail(`remainingMs must be a finite number in (0, ${MAX_REMAINING_MS}]`);
    }
    if (!record(clock) || typeof clock.now !== 'function') fail('clock must provide now()');
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
    const perf = globalThis.performance;
    if (!perf || typeof perf.now !== 'function') fail('performance.now is unavailable');
    if (typeof globalThis.requestAnimationFrame !== 'function') fail('requestAnimationFrame is unavailable');
    return {
      now: () => perf.now(),
      requestFrame: callback => globalThis.requestAnimationFrame(callback),
      cancelFrame: id => {
        if (typeof globalThis.cancelAnimationFrame === 'function') globalThis.cancelAnimationFrame(id);
      },
    };
  }

  function validateInput(input) {
    if (!record(input)) fail('input must be an object');
    if (Object.keys(input).length !== INPUT_KEYS.length || !INPUT_KEYS.every(key => Object.hasOwn(input, key))) {
      fail(`input must have exactly the keys ${INPUT_KEYS.join(', ')}`);
    }
    for (const name of ['selectionId', 'attemptId', 'buildId', 'scenario', 'mode']) {
      if (!nonblank(input[name])) fail(`input.${name} must be a nonblank string`);
    }
    for (const name of ['width', 'height']) {
      if (!Number.isSafeInteger(input[name]) || input[name] <= 0) fail(`input.${name} must be a positive integer pixel count`);
    }
    if (!finite(input.remainingMs) || input.remainingMs <= 0 || input.remainingMs > MAX_REMAINING_MS) {
      fail(`input.remainingMs must be a finite positive duration of at most ${MAX_REMAINING_MS} ms`);
    }
  }

  // `entry` is the scene namespace, `scene` its default export; `rows` are the scene's inventory rows
  // of this scenario and mode (at least one).
  function resolveSelection(scenes, input) {
    if (!(scenes instanceof Map)) fail('scenes must be a Map');
    let entry = scenes.get(input.scenario);
    if (entry === undefined) {
      entry = [...scenes.values()].find(candidate => record(candidate) && record(candidate.default) &&
        candidate.default.id === input.scenario);
    }
    if (!record(entry) || !record(entry.default)) fail(`unknown scenario ${JSON.stringify(input.scenario)}`);
    const scene = entry.default;
    if (scene.id !== input.scenario) fail(`scenario ${input.scenario}: default.id must match the scenario`);
    const variant = Array.isArray(scene.variants) ? scene.variants.find(candidate => record(candidate) && candidate.id === input.mode) : undefined;
    if (!variant) fail(`scenario ${input.scenario} has no mode ${JSON.stringify(input.mode)}`);
    const rows = Array.isArray(entry.inventory)
      ? entry.inventory.filter(row => record(row) && row.scenario === input.scenario && row.mode === input.mode)
      : [];
    if (rows.length === 0) fail(`no inventory image for ${input.scenario}/${input.mode}`);
    return { entry, scene, variant, rows };
  }

  // One viewport: the canvas size is the only view (the variant's engine is not consulted).
  function viewSizesFor(engine, canvas) {
    const { width, height } = canvas;
    if (!Number.isSafeInteger(width) || width <= 0 || !Number.isSafeInteger(height) || height <= 0) {
      fail('canvas must have positive integer pixel dimensions');
    }
    return [{ view: 0, width, height }];
  }

  return { createOperationBudget, browserClock, validateInput, resolveSelection, viewSizesFor };
}

export const {
  createOperationBudget, browserClock, validateInput, resolveSelection, viewSizesFor,
} = helperSet('page runner');

export function describeError(error) {
  const message = error && error.message ? error.message : String(error);
  const code = error && typeof error.code === 'string' ? error.code : '';
  return code && !message.startsWith(`${code}: `) ? `${code}: ${message}` : message;
}

export function variantTimeMs(variant) {
  return variant && variant.timeMs !== undefined ? variant.timeMs : DEFAULT_TIME_MS;
}

// `state` is the operation's own object: the optional frame hook receives it as its fourth
// argument, so a scene can keep per-operation values between frames.
export function renderAt(engine, scene, variant, timeMs, state) {
  if (typeof scene.frame === 'function') scene.frame(engine, timeMs, variant, state);
  engine.render(timeMs);
}

export function setCanvasSize(canvas, width, height) {
  canvas.width = width;
  canvas.height = height;
  if (canvas.style) {
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
  }
}

export function adapterFacts(value) {
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

// Hardware needs the Engine's own flag, no fallback adapter and no software name in the four strings.
export function isHardware(statsHardware, adapter) {
  if (statsHardware !== true || !record(adapter) || adapter.isFallbackAdapter === true) return false;
  return ![adapter.vendor, adapter.architecture, adapter.device, adapter.description]
    .some(value => typeof value === 'string' && SOFTWARE_ADAPTER.test(value));
}

export function browserFacts() {
  const source = globalThis.navigator && typeof globalThis.navigator === 'object' ? globalThis.navigator : {};
  const ratio = finite(globalThis.devicePixelRatio) && globalThis.devicePixelRatio > 0 ? globalThis.devicePixelRatio : 1;
  return {
    userAgent: typeof source.userAgent === 'string' ? source.userAgent : '',
    platform: typeof source.platform === 'string' ? source.platform : '',
    devicePixelRatio: ratio,
  };
}

// Collects the page, console and GPU errors of one page session. Restores everything on dispose.
export function createErrorCollectors() {
  const errors = { console: [], page: [], gpu: [] };
  const win = globalThis.window;
  const listening = !!win && typeof win.addEventListener === 'function';
  let deviceAdapter = null;
  let restoreGpu = null;
  let disposed = false;

  const note = (list, error) => { if (!disposed) list.push(describeError(error)); };
  const onError = event => note(errors.page, event && event.error !== undefined ? event.error : event);
  const onRejection = event => note(errors.page, event && event.reason !== undefined ? event.reason : event);

  if (listening) {
    win.addEventListener('error', onError);
    win.addEventListener('unhandledrejection', onRejection);
  }

  const gpu = globalThis.navigator?.gpu;
  if (gpu && typeof gpu.requestAdapter === 'function' && !gpu.__nmcPageErrors) {
    const original = gpu.requestAdapter;
    gpu.requestAdapter = async descriptor => {
      const adapter = await original.call(gpu, descriptor);
      if (!adapter || typeof adapter.requestDevice !== 'function') return adapter;
      const requestDevice = adapter.requestDevice.bind(adapter);
      adapter.requestDevice = async deviceDescriptor => {
        const device = await requestDevice(deviceDescriptor);
        // The adapter that created this device is its provenance; a separate request would not be.
        const info = adapter.info;
        if (info) {
          deviceAdapter = adapterFacts({
            vendor: info.vendor, architecture: info.architecture,
            device: info.device, description: info.description,
            isFallbackAdapter: info.isFallbackAdapter ?? adapter.isFallbackAdapter,
          });
        }
        if (device && typeof device.addEventListener === 'function') {
          device.addEventListener('uncapturederror', event => note(errors.gpu, event?.error ?? event));
        }
        // Only an explicit destroy is expected; any other loss blanks every canvas.
        if (device?.lost?.then) {
          device.lost.then(info => {
            if (info?.reason === 'destroyed') return;
            note(errors.gpu, `GPU device lost: ${info?.reason ?? 'unknown'}: ${info?.message ?? ''}`.trim());
          }, () => {});
        }
        return device;
      };
      return adapter;
    };
    gpu.__nmcPageErrors = true;
    restoreGpu = () => {
      gpu.requestAdapter = original;
      delete gpu.__nmcPageErrors;
    };
  }

  function installConsole() {
    const target = globalThis.console;
    if (!target || typeof target.error !== 'function') return () => {};
    const original = target.error;
    const wrapper = (...args) => {
      note(errors.console, args.map(value => describeError(value)).join(' '));
      original.apply(target, args);
    };
    target.error = wrapper;
    return () => { if (target.error === wrapper) target.error = original; };
  }

  function reset() {
    for (const list of Object.values(errors)) list.length = 0;
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    if (listening) {
      win.removeEventListener('error', onError);
      win.removeEventListener('unhandledrejection', onRejection);
    }
    if (restoreGpu) restoreGpu();
    restoreGpu = null;
    deviceAdapter = null;
  }

  return { errors, deviceAdapter: () => deviceAdapter, installConsole, reset, dispose };
}

// Handoff: createPageRunner(options) is called by card vg-page: createVgPageRunner on createPageRunner
// and by card gi-page: lib/runtime.js and lib/facts.js. Each calls createPageRunner(options) with its own
// topic, collectFacts, validateRequiredFacts and measurementExtras; the GI page also passes settle.
// It continues the share-1 handoff at web/common/page-runner.js:23 helperSet: every refusal of the runner
// goes through helperSet('<topic> runtime'), except the topic check, which throws 'page runner: ...'.
// The returned runner validates the topic first, then canvas, status, load, scenes, lib and the fact
// callbacks; run() validates the input, builds a fresh Engine, warms up WARMUP_FRAMES frames, runs the
// optional settle phase, observes, validates every inventory row, measures a measured variant, captures,
// and answers the S26 result with facts.readiness. dispose() cancels a pending frame and disposes the Engine.
export function createPageRunner(options) {
  if (!record(options) || !nonblank(options.topic)) {
    throw new Error('page runner: options.topic must be a nonblank string');
  }
  const prefix = `${options.topic} runtime`;
  const fail = message => { throw new Error(`${prefix}: ${message}`); };
  const { createOperationBudget, browserClock, validateInput, resolveSelection, viewSizesFor } = helperSet(prefix);

  if (!record(options.canvas)) fail('options.canvas is required');
  if (!record(options.status) || !('textContent' in options.status)) fail('options.status must be an object with textContent');
  if (typeof options.load !== 'function') fail('options.load must be a function');
  if (!(options.scenes instanceof Map)) fail('options.scenes must be a Map');
  if (!Object.hasOwn(options, 'lib')) fail('options.lib is required');
  if (typeof options.collectFacts !== 'function') fail('options.collectFacts must be a function');
  if (typeof options.validateRequiredFacts !== 'function') fail('options.validateRequiredFacts must be a function');
  if (options.capture != null && typeof options.capture !== 'function') fail('options.capture must be a function');
  if (options.measurementExtras != null && typeof options.measurementExtras !== 'function') fail('options.measurementExtras must be a function');
  if (options.settle != null && typeof options.settle !== 'function') fail('options.settle must be a function');
  if (options.clock != null && (!record(options.clock) || typeof options.clock.now !== 'function')) fail('options.clock must provide now()');

  const { canvas, status } = options;
  const capture = options.capture ?? captureCurrent;
  const readiness = createReadiness('page');
  const collectors = createErrorCollectors();
  let engine = null;
  let raa = null;
  let running = false;
  let disposed = false;
  let pendingFrame = null;

  const showStatus = text => { status.textContent = text; };
  const checkDisposed = () => { if (disposed) fail('runner is disposed'); };
  const checkBudget = (budget, label) => { if (budget.remainingMs() <= 0) fail(`operation deadline expired before ${label}`); };

  async function loadModule() {
    if (!raa) {
      const loaded = await options.load();
      if (!record(loaded) || typeof loaded.createEngine !== 'function') fail('load() must resolve a RaaModule with createEngine');
      raa = loaded;
    }
    return raa;
  }

  // A fresh Engine of the scene and variant options, resized to the input; the previous one is disposed first.
  async function createEngineFor(scene, variant, width, height) {
    if (engine) {
      const previous = engine;
      engine = null;
      previous.dispose();
    }
    setCanvasSize(canvas, width, height);
    const module = await loadModule();
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

  // The next animation frame of the clock. Dispose cancels the wait and rejects it.
  function nextFrame(clock) {
    return new Promise((resolve, reject) => {
      const id = clock.requestFrame(() => {
        pendingFrame = null;
        resolve();
      });
      pendingFrame = () => {
        pendingFrame = null;
        clock.cancelFrame(id);
        reject(new Error(`${prefix}: runner is disposed`));
      };
    });
  }

  // One PAIR_RUN record of the variant's on/off state, extended by measurementExtras.
  async function measureVariant(target, scene, variant, state, clock, budget) {
    if (variant.measurement === undefined) return [];
    const render = () => renderAt(target, scene, variant, variantTimeMs(variant), state);
    const measurement = await collectFeatureMeasurement(render, { ...variant.measurement, deadlineMs: budget.deadlineMs }, clock);
    return [{ ...measurement, ...(options.measurementExtras?.(target, variant) ?? {}) }];
  }

  async function run(input) {
    if (disposed) fail('runner is disposed');
    if (running) fail('a run is already in progress');
    running = true;
    const restoreConsole = collectors.installConsole();
    // The stage a failure is recorded under: 'selection', 'build', 'frames' or 'settle' (the scene's own
    // stage), 'observe' or 'capture'.
    let phase = 'selection';
    let sceneStage = 'page';
    let clock = null;
    let budget = null;
    try {
      validateInput(input);
      clock = options.clock ?? browserClock();
      budget = createOperationBudget(input.remainingMs, clock);
      collectors.reset();
      const { scene, variant, entry, rows } = resolveSelection(options.scenes, input);
      sceneStage = scene.milestone;
      readiness.pending(sceneStage);
      showStatus(`pending: preparing ${input.scenario}/${input.mode}`);

      phase = 'build';
      const target = await createEngineFor(scene, variant, input.width, input.height);
      const state = await scene.build(target, options.lib, variant);

      // Frames until the warm-up is done and the scene's ready hook (if any) reports ready.
      phase = 'frames';
      for (let frames = 1; ; frames++) {
        await nextFrame(clock);
        checkDisposed();
        renderAt(target, scene, variant, variantTimeMs(variant), state);
        const verdict = typeof scene.ready === 'function' ? scene.ready(target, state, variant) : { ready: true, stage: sceneStage };
        sceneStage = verdict.stage;
        if (frames >= WARMUP_FRAMES && verdict.ready) break;
        if (budget.remainingMs() <= 0) fail(`readiness: ${sceneStage} not reached before the operation deadline`);
      }

      // The optional settle phase renders more frames of the same Engine; its verdict gates readiness.
      if (options.settle) {
        phase = 'settle';
        const verdict = await options.settle(target, {
          render: async () => {
            await nextFrame(clock);
            checkDisposed();
            renderAt(target, scene, variant, variantTimeMs(variant), state);
          },
          budget,
          scene,
          variant,
          state,
        });
        if (!record(verdict) || verdict.status !== 'ready') {
          fail(`readiness: ${record(verdict) && typeof verdict.error === 'string' ? verdict.error : 'settle did not report ready'}`);
        }
      }

      phase = 'observe';
      const observation = await entry.observe(target, variant, state);
      if (!record(observation)) fail('observe must resolve an Observation object');
      readiness.ready();
      const facts = {
        ...options.collectFacts(target, observation, variant),
        passes: target.passes(),
        viewSizes: viewSizesFor(target, canvas),
        readiness: readiness.snapshot(),
      };
      showStatus(`ready: ${input.scenario}/${input.mode}${typeof state?.status === 'string' ? ` ${state.status}` : ''}`);
      for (const row of rows) options.validateRequiredFacts(row, facts, entry.factNames);

      phase = 'capture';
      checkBudget(budget, 'the measurements');
      const measurements = await measureVariant(target, scene, variant, state, clock, budget);
      checkBudget(budget, 'the capture');
      renderAt(target, scene, variant, variantTimeMs(variant), state);
      const data = await capture(canvas, status);
      checkBudget(budget, 'the result');

      const stats = target.stats();
      const adapter = collectors.deviceAdapter() ?? adapterFacts(stats.adapter);
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
        hardware: isHardware(stats.hardware === true, adapter),
        images: rows.map(row => ({
          id: row.id,
          filename: row.filename,
          mime: 'image/png',
          data,
          observation: row.expectedVisibleBehavior,
        })),
        facts,
        measurements,
        errors: {
          console: [...collectors.errors.console],
          page: [...collectors.errors.page],
          gpu: [...collectors.errors.gpu],
        },
      };
    } catch (error) {
      readiness.fail(phase === 'frames' || phase === 'settle' ? sceneStage : phase, error);
      showStatus(`failed: ${describeError(error)}`);
      throw error;
    } finally {
      running = false;
      restoreConsole();
    }
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    if (pendingFrame) pendingFrame();
    collectors.dispose();
    const current = engine;
    engine = null;
    if (current && typeof current.dispose === 'function') current.dispose();
  }

  return { run, engine: () => engine, dispose };
}
