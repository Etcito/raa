// Contract 'vg-runtime'. The constants and standalone helpers are re-exported from
// web/common/page-runner.js (contract 'page-runner'); createVgPageRunner keeps its own fail and record.
import {
  MAX_REMAINING_MS, WARMUP_FRAMES, DEFAULT_TIME_MS, createOperationBudget, browserClock, describeError,
  variantTimeMs, renderAt, setCanvasSize, validateInput, resolveSelection, viewSizesFor, adapterFacts,
  isHardware, browserFacts, createErrorCollectors,
} from '../../../web/common/page-runner.js';
import { validateRequiredFacts } from './inventory.js';
import { collectVgFacts } from './facts.js';
import { createReadiness } from '../../sky-weather/lib/readiness.js';
import { collectFeatureMeasurement } from '../../sky-weather/lib/measure.js';
import { captureCurrent } from '../../sky-weather/lib/capture.js';

export {
  MAX_REMAINING_MS, WARMUP_FRAMES, DEFAULT_TIME_MS, createOperationBudget, browserClock, describeError,
  variantTimeMs, renderAt, setCanvasSize, validateInput, resolveSelection, viewSizesFor, adapterFacts,
  isHardware, browserFacts, createErrorCollectors,
};

const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);

function fail(message) { throw new Error(`vg runtime: ${message}`); }

function checkBudget(budget, label) {
  if (budget.remainingMs() <= 0) fail(`operation deadline expired before ${label}`);
}

// Share 2 of contract 'vg-runtime'. Form: examples/sky-weather/lib/runtime.js lines 159-460.
// Handoff: createVgPageRunner(options) is rewritten by card vg-page: createVgPageRunner on createPageRunner,
// which builds it on createPageRunner (web/common/page-runner.js:256 createPageRunner). Today it is this
// standalone runner, calling the page-kit helpers re-exported above.
export function createVgPageRunner(options) {
  if (!record(options)) fail('options must be an object');
  if (!record(options.canvas)) fail('options.canvas is required');
  if (!record(options.status)) fail('options.status is required');
  if (typeof options.load !== 'function') fail('options.load must be a function');
  if (!(options.scenes instanceof Map)) fail('options.scenes must be a Map');
  if (!Object.hasOwn(options, 'lib')) fail('options.lib is required');

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
        reject(new Error('vg runtime: runner is disposed'));
      };
    });
  }

  // One PAIR_RUN record of the variant's on/off state, extended with the cluster and triangle counts.
  async function measureVariant(target, scene, variant, state, clock, budget) {
    if (variant.measurement === undefined) return [];
    const render = () => renderAt(target, scene, variant, variantTimeMs(variant), state);
    const measurement = await collectFeatureMeasurement(render, { ...variant.measurement, deadlineMs: budget.deadlineMs }, clock);
    const stats = target.stats();
    const vg = variant.measurement.state === 'on' && typeof target.virtualGeometryStats === 'function'
      ? target.virtualGeometryStats()
      : null;
    return [{
      ...measurement,
      drawnClusters: vg ? vg.drawnClusters : 0,
      drawnTriangles: vg ? vg.drawnTriangles : stats.triangles,
      hardwareClusters: vg ? vg.hardwareClusters : 0,
      softwareClusters: vg ? vg.softwareClusters : 0,
    }];
  }

  async function run(input) {
    if (disposed) fail('runner is disposed');
    if (running) fail('a run is already in progress');
    running = true;
    const restoreConsole = collectors.installConsole();
    // The stage a failure is recorded under: 'selection', 'build', 'frames' (the scene's own stage), 'observe' or 'capture'.
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

      phase = 'observe';
      const observation = await entry.observe(target, variant, state);
      if (!record(observation)) fail('observe must resolve an Observation object');
      const collected = collectVgFacts(target, observation.meshes, observation.instances);
      readiness.ready();
      const facts = {
        settings: collected.settings,
        stats: collected.stats,
        meshes: collected.meshes,
        instances: collected.instances,
        assertions: observation.assertions,
        edits: observation.edits,
        passes: target.passes(),
        viewSizes: viewSizesFor(target, canvas),
        readiness: readiness.snapshot(),
      };
      showStatus(`ready: ${input.scenario}/${input.mode}${typeof state?.status === 'string' ? ` ${state.status}` : ''}`);
      for (const row of rows) validateRequiredFacts(row, facts, entry.factNames);

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
      readiness.fail(phase === 'frames' ? sceneStage : phase, error);
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
