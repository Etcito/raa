// Page runtime of the render-graph page (contract rg-page-runtime, module render-graph-page).
// createRgPageRunner wraps createPageRunner of the page kit: its settle phase runs RG_RUN for every
// half of the variant (one 'single' half, or the pair's 'left' and 'right'), copies the GPU canvas
// right after frame 240, composes the half images and keeps the base64 PNG for the capture option.
// No top-level side effect: importing this module touches no DOM, canvas or engine.
import { createPageRunner } from '../../../web/common/page-runner.js';
// Consumed upstream, not re-implemented here: createPageRunner (web/common/page-runner.js:265, its
// Handoff doc comment at :256), applyCamera (examples/render-graph/lib/scene.js:62) and cameraAt
// (:56) from the Handoff doc comment at examples/render-graph/lib/scene.js:31, and
// validateRequiredFacts (examples/render-graph/lib/inventory.js:137).
import { RG_CAPTURE_T_S, RG_DT_MS, RG_WARMUP_FRAMES, applyCamera } from './scene.js';
import { validateRequiredFacts } from './inventory.js';

const CAPTURE_WIDTH = 1280;
const CAPTURE_HEIGHT = 720;
const PAIR_WIDTH = CAPTURE_WIDTH * 2;
const LABEL_BAND = 28;
const LABEL_FONT = 'bold 16px monospace';
const PADDING = 8;
const LINE_HEIGHT = 18;
const FONT = '14px monospace';
const BACKGROUND = '#0b0d12';
const FOREGROUND = '#f2f2f2';
const LABEL_COLOR = '#ffffff';

function fail(message) { throw new Error(`render-graph runtime: ${message}`); }

// The camera time of frame f: f x RG_DT_MS / 1000 s. The captured frame is RG_CAPTURE_T_S itself,
// the exact second the rg-room-scene constant states.
function frameTimeSeconds(f) {
  return f === RG_WARMUP_FRAMES ? RG_CAPTURE_T_S : f * RG_DT_MS / 1000;
}

/** The 241 camera times of RG_RUN: frame f is at {@link frameTimeSeconds}. */
export function halfFrameTimes() {
  return Array.from({ length: RG_WARMUP_FRAMES + 1 }, (_, f) => frameTimeSeconds(f));
}

/**
 * Handoff: render-graph page 3/5: interpreter scene and its test calls `captionText(issue) -> string`
 * exported by examples/render-graph/lib/runtime.js, attaching it by ES import in
 * examples/render-graph/scenes/interpreter.js for the caption of its refused row; state/keys: a
 * RenderGraphIssue of contract rg-page-facts, read for code, nodeId, param and message. Today it
 * returns `refused: <code> on node <nodeId>[ (param <param>)]: <message>`.
 */
export function captionText(issue) {
  const { code, nodeId, param, message } = issue ?? {};
  const parameter = param == null ? '' : ` (param ${param})`;
  return `refused: ${code} on node ${nodeId}${parameter}: ${message}`;
}

/**
 * `collectRgFacts(engine, observation, variant)` builds the render-graph facts of contract
 * rg-page-facts from the observation: `renderGraphActive`, `graph`, `issues`, `assertions` and
 * `edits`, each null when the observation does not provide it. The page runner adds `passes`,
 * `viewSizes` and `readiness`. The engine and variant are not read.
 */
export function collectRgFacts(engine, observation, variant) {
  const source = observation !== null && typeof observation === 'object' ? observation : {};
  const field = key => Object.hasOwn(source, key) ? source[key] ?? null : null;
  return {
    renderGraphActive: field('renderGraphActive'),
    graph: field('graph'),
    issues: field('issues'),
    assertions: field('assertions'),
    edits: field('edits'),
  };
}

function measure(context, text) {
  if (typeof context.measureText === 'function') return context.measureText(text).width;
  return text.length * 8;
}

// The wrapping of examples/sky-weather/lib/capture.js wrapLines, which is not exported there.
function wrapLines(context, text, maxWidth) {
  const source = text.replace(/\r\n?/g, '\n').trim();
  if (!source) return [];
  const lines = [];
  for (const paragraph of source.split('\n')) {
    const words = paragraph.split(/\s+/).filter(Boolean);
    if (words.length === 0) { lines.push(''); continue; }
    let line = '';
    for (const word of words) {
      const candidate = line ? `${line} ${word}` : word;
      if (line && measure(context, candidate) > maxWidth) {
        lines.push(line);
        line = word;
      } else {
        line = candidate;
      }
    }
    if (line) lines.push(line);
  }
  return lines;
}

function createCanvas(source) {
  const document = source && source.ownerDocument ? source.ownerDocument
    : (typeof globalThis.document === 'object' ? globalThis.document : null);
  if (!document || typeof document.createElement !== 'function') fail('a document is required to compose the capture');
  const output = document.createElement('canvas');
  if (!output || typeof output.getContext !== 'function') fail('cannot create the capture canvas');
  return output;
}

// A 1280x720 copy of the GPU canvas, drawn in the same task as the frame-240 render.
function copyFrame(source) {
  const output = createCanvas(source);
  output.width = CAPTURE_WIDTH;
  output.height = CAPTURE_HEIGHT;
  const context = output.getContext('2d', { alpha: false, willReadFrequently: true });
  if (!context) fail('2D canvas context is unavailable');
  try {
    context.drawImage(source, 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT);
  } catch (error) {
    fail(`cannot copy the GPU canvas: ${error && error.message ? error.message : String(error)}`);
  }
  return output;
}

function debugGridLabels(graph) {
  if (!graph || !Array.isArray(graph.nodes) || !Array.isArray(graph.links)) return null;
  const grid = graph.nodes.find(node => node.type === 'view.grid' && node.params?.labels !== false);
  if (!grid) return null;
  const nodes = new Map(graph.nodes.map(node => [node.id, node]));
  const ordered = [];
  for (let index = 0; index < 8; index++) {
    const link = graph.links.find(item => item.to === `${grid.id}.t${index}`);
    const source = link && nodes.get(link.from.slice(0, link.from.lastIndexOf('.')));
    const label = source?.params?.label;
    if (typeof label !== 'string' || !label.trim()) return null;
    ordered.push(label === 'ao' ? 'ambient occlusion' : label);
  }
  return ordered;
}

function annotateDebugGrid(context, offset, labels) {
  if (!labels) return;
  context.font = '12px monospace';
  context.textBaseline = 'top';
  for (let index = 0; index < labels.length; index++) {
    const column = index % 4, row = Math.floor(index / 4);
    const x = offset + column * 320;
    const y = Math.max(LABEL_BAND, row * 360);
    const width = Math.min(320, Math.ceil(measure(context, labels[index])) + 8);
    const height = 18;
    context.fillStyle = '#000000';
    context.fillRect(x, y, width, height);
    context.fillStyle = '#ffffff';
    context.fillText(labels[index], x + 4, y + 2, 312);
  }
}

// The single 1280x720 image, or the 2560x720 pair (left at x 0, right at x 1280) with its labels,
// plus the optional caption strip; returns the base64 PNG and its 2D context.
function composeCapture(copies, pair, caption, graphs = new Map()) {
  const first = copies.get(pair ? 'left' : 'single');
  if (!first) fail(pair ? 'the RG_RUN did not copy the left half' : 'the RG_RUN did not copy the single half');
  const second = pair ? copies.get('right') : null;
  if (pair && !second) fail('the RG_RUN did not copy the right half');
  const width = pair ? PAIR_WIDTH : CAPTURE_WIDTH;
  const output = createCanvas(first);
  const measureContext = output.getContext('2d', { alpha: false, willReadFrequently: true });
  if (!measureContext) fail('2D canvas context is unavailable');
  measureContext.font = FONT;
  const lines = typeof caption === 'string' ? wrapLines(measureContext, caption, Math.max(1, width - PADDING * 2)) : [];
  const captionHeight = lines.length > 0 ? lines.length * LINE_HEIGHT + PADDING * 2 : 0;
  output.width = width;
  output.height = CAPTURE_HEIGHT + captionHeight;
  const context = output.getContext('2d', { alpha: false, willReadFrequently: true });
  if (!context) fail('2D canvas context is unavailable');
  context.fillStyle = BACKGROUND;
  context.fillRect(0, 0, output.width, output.height);
  context.drawImage(first, 0, 0);
  if (second) context.drawImage(second, CAPTURE_WIDTH, 0);
  if (pair) {
    context.font = LABEL_FONT;
    context.fillStyle = BACKGROUND;
    context.fillRect(0, 0, output.width, LABEL_BAND);
    context.fillStyle = LABEL_COLOR;
    context.textBaseline = 'middle';
    context.fillText(String(pair.left), PADDING, LABEL_BAND / 2);
    context.fillText(String(pair.right), CAPTURE_WIDTH + PADDING, LABEL_BAND / 2);
  }
  annotateDebugGrid(context, 0, debugGridLabels(graphs.get(pair ? 'left' : 'single')));
  if (second) annotateDebugGrid(context, CAPTURE_WIDTH, debugGridLabels(graphs.get('right')));
  if (captionHeight > 0) {
    context.fillStyle = FOREGROUND;
    context.textBaseline = 'top';
    context.font = FONT;
    lines.forEach((line, index) => context.fillText(line, PADDING, CAPTURE_HEIGHT + PADDING + index * LINE_HEIGHT));
  }
  const url = typeof output.toDataURL === 'function' ? output.toDataURL('image/png') : '';
  const prefix = 'data:image/png;base64,';
  if (typeof url !== 'string' || !url.startsWith(prefix) || url.length === prefix.length) {
    fail('the composed canvas did not produce a PNG data URL');
  }
  return { data: url.slice(prefix.length), context };
}

/**
 * Handoff: render-graph page 4/5: page entry and build script calls
 * `createRgPageRunner(options) -> { run(input), engine(), dispose() }` exported by
 * examples/render-graph/lib/runtime.js; the page entry attaches it with an ES import and passes its
 * canvas, status, load, scenes map and lib helpers. State/keys: the rg-page-facts shape of the
 * settle's observation and the RG_DT_MS, RG_WARMUP_FRAMES and RG_CAPTURE_T_S constants. Today it
 * returns createPageRunner's runner with topic 'render-graph', collectRgFacts, the inventory
 * validateRequiredFacts, the RG_RUN settle phase and this file's composed-image capture.
 * Handoff: Build the three frozen comparison scenes calls `createRgPageRunner(options)` in
 * examples/render-graph/lib/runtime.js to consume `scene.isolatedPair`; today it creates isolated
 * pipeline engines, records their immutable `state.isolatedHalves` capture facts and routes final
 * page-kit observation through the interpreted half. Existing page-kit hooks remain local adapters.
 */
export function createRgPageRunner(options) {
  const canvas = options?.canvas;
  const lib = options?.lib;
  let composed = null;
  let composedContext = null;
  const owned = new Set();
  let activeEngine = null;
  let stableHandle = null;
  let disposed = false;
  let generation = 0;
  let moduleValue = null;

  function proxyFor(engine) {
    activeEngine = engine;
    if (stableHandle) return stableHandle;
    stableHandle = new Proxy({}, {
      get(_target, key) {
        if (key === 'then') return undefined;
        if (key === 'dispose') return (...args) => {
          const current = activeEngine;
          if (!current) return;
          owned.delete(current);
          return current.dispose.apply(current, args);
        };
        const value = activeEngine?.[key];
        return typeof value === 'function' ? (...args) => value.apply(activeEngine, args) : value;
      },
      set(_target, key, value) { activeEngine[key] = value; return true; },
    });
    return stableHandle;
  }

  function disposeOwned(engine) {
    if (engine && owned.delete(engine)) engine.dispose();
  }

  const isolatedLoad = async () => {
    moduleValue = await options.load();
    return {
      ...moduleValue,
      async createEngine(targetCanvas, engineOptions) {
        const engine = await moduleValue.createEngine(targetCanvas, engineOptions);
        if (disposed) { engine?.dispose?.(); fail('runner is disposed'); }
        return proxyFor(engine);
      },
    };
  };

  function capture() {
    if (composed === null) fail('no composed image exists for this run');
    if (composedContext && typeof composedContext.isContextLost === 'function' && composedContext.isContextLost()) {
      fail('the capture canvas lost its 2D context (browser GPU process reset); its PNG would be blank');
    }
    return composed;
  }

  async function settle(engine, { budget, scene, variant, state }) {
    if (scene?.isolatedPair !== undefined) {
      await settleIsolated(scene, variant, state, budget);
      return { status: 'ready' };
    }
    if (typeof scene?.prepareHalf !== 'function') fail('scene.prepareHalf must be a function');
    composed = null;
    composedContext = null;
    const pair = variant?.pair ?? null;
    if (pair && !Array.isArray(pair.order)) fail('variant.pair.order must list the halves');
    const halves = pair ? pair.order : ['single'];
    const copies = new Map();
    const graphs = new Map();
    let timeMs = 0;

    // The render and whatever runs after it share one animation frame callback (the frame-240 copy).
    const nextFrame = render => new Promise((resolve, reject) => {
      if (typeof globalThis.requestAnimationFrame !== 'function') fail('requestAnimationFrame is unavailable');
      globalThis.requestAnimationFrame(() => {
        try { render(); resolve(); } catch (error) { reject(error); }
      });
    });

    const renderFrames = async count => {
      if (!Number.isSafeInteger(count) || count < 0) fail('renderFrames(n) must take a non-negative integer');
      for (let index = 0; index < count; index++) {
        await nextFrame(() => {
          if (budget.remainingMs() <= 0) fail(`operation deadline expired before the prepareHalf frames (${index})`);
          engine.render(timeMs);
          timeMs += RG_DT_MS;
        });
      }
    };

    for (const half of halves) {
      await scene.prepareHalf(engine, lib, variant, state, half, { renderFrames });
      for (let f = 0; f <= RG_WARMUP_FRAMES; f++) {
        await nextFrame(() => {
          if (budget.remainingMs() <= 0) {
            fail(`operation deadline expired before the RG_RUN frames (${half}, frame ${f})`);
          }
          applyCamera(engine, frameTimeSeconds(f));
          engine.render(timeMs);
          timeMs += RG_DT_MS;
          if (f === RG_WARMUP_FRAMES) {
            copies.set(half, copyFrame(canvas));
            graphs.set(half, engine.renderGraph?.());
          }
        });
      }
    }

    const caption = typeof scene.caption === 'function' ? scene.caption(engine, state, variant) : null;
    const result = composeCapture(copies, pair, typeof caption === 'string' ? caption : null, graphs);
    composed = result.data;
    composedContext = result.context;
    return { status: 'ready' };
  }

  async function settleIsolated(scene, variant, state, budget) {
    const hooks = scene.isolatedPair;
    if (!hooks || typeof hooks !== 'object' || typeof hooks.engineOptions !== 'function' ||
        typeof hooks.prepare !== 'function' || typeof hooks.seeds !== 'function') {
      fail('isolatedPair requires engineOptions, prepare and seeds hooks');
    }
    if (variant?.measurement || typeof scene.prepareHalf === 'function' || !variant?.pair?.order ||
        variant.pair.order.length !== 2 || variant.pair.order[0] !== 'left' || variant.pair.order[1] !== 'right') {
      fail('unsupported isolatedPair scene or variant hooks');
    }
    const runId = ++generation;
    const module = moduleValue ?? await options.load();
    const snapshots = [];
    let previous = null;
    try {
      for (const half of ['left', 'right']) {
        if (disposed || runId !== generation) fail('isolatedPair run was cancelled');
        if (budget.remainingMs() <= 0) fail('operation deadline expired before isolated engine creation');
        const opts = hooks.engineOptions(variant, state, half);
        if (!opts || typeof opts !== 'object' || Array.isArray(opts)) fail(`isolatedPair engineOptions(${half}) is invalid`);
        const targetCanvas = createCanvas(canvas);
        targetCanvas.width = CAPTURE_WIDTH;
        targetCanvas.height = CAPTURE_HEIGHT;
        const creating = Promise.resolve(module.createEngine(targetCanvas, opts));
        const measured = await creating;
        owned.add(measured);
        if (disposed || runId !== generation) { disposeOwned(measured); fail('isolatedPair run was cancelled'); }
        if (!measured || typeof measured.render !== 'function' || typeof measured.resize !== 'function' || typeof measured.dispose !== 'function') {
          fail('isolatedPair createEngine did not resolve a usable Engine');
        }
        measured.resize(CAPTURE_WIDTH, CAPTURE_HEIGHT);
        await hooks.prepare(measured, lib, variant, state, half);
        if (disposed || runId !== generation) fail('isolatedPair run was cancelled');
        const times = [], cameraTimes = [];
        for (let f = 0; f <= RG_WARMUP_FRAMES; f++) {
          await new Promise((resolve, reject) => {
            if (typeof globalThis.requestAnimationFrame !== 'function') return reject(new Error('requestAnimationFrame is unavailable'));
            globalThis.requestAnimationFrame(() => {
              try {
                if (disposed || runId !== generation) fail('isolatedPair run was cancelled');
                if (budget.remainingMs() <= 0) fail(`operation deadline expired before isolated frame ${f}`);
                applyCamera(measured, f * RG_DT_MS / 1000);
                measured.render(f * RG_DT_MS);
                times.push(f * RG_DT_MS);
                cameraTimes.push(f / 60);
                resolve();
              } catch (error) { reject(error); }
            });
          });
        }
        const image = copyFrame(targetCanvas);
        const seeds = hooks.seeds(measured, variant, state, half);
        const graph = measured.renderGraph();
        const issues = measured.renderGraphIssues();
        const stats = measured.renderGraphStats();
        const passes = measured.passes();
        const active = measured.renderGraphActive();
        const snapshot = structuredClone({ half, graphFrameIndex: stats.frameIndex, renderTimesMs: times,
          cameraTimesS: cameraTimes, seeds, graph, active, issues, stats, passes });
        validateHalf(snapshot);
        snapshots.push(snapshot);
        const destination = state.isolatedImages ??= {};
        destination[half] = image;
        if (previous) disposeOwned(previous);
        previous = measured;
        activeEngine = measured;
        if (half === 'left') copiesForIsolatedLeft = image;
      }
      if (!sameProvenance(snapshots[0].seeds, snapshots[1].seeds) ||
          snapshots[0].graphFrameIndex !== snapshots[1].graphFrameIndex ||
          snapshots[0].renderTimesMs.some((time, index) => time !== snapshots[1].renderTimesMs[index]) ||
          snapshots[0].cameraTimesS.some((time, index) => time !== snapshots[1].cameraTimesS[index])) {
        fail('isolatedPair halves have mismatched frame or seed provenance');
      }
      state.isolatedHalves = Object.freeze(snapshots.map(snapshot => deepFreeze(snapshot)));
      copiesForIsolatedRight = state.isolatedImages.right;
      const pair = variant.pair;
      const result = composeCapture(new Map([['left', copiesForIsolatedLeft], ['right', copiesForIsolatedRight]]), pair, null,
        new Map(snapshots.map(snapshot => [snapshot.half, snapshot.graph])));
      composed = result.data; composedContext = result.context;
      if (previous) owned.add(previous);
    } catch (error) {
      for (const engine of [...owned]) disposeOwned(engine);
      activeEngine = null;
      throw error;
    }
  }

  let copiesForIsolatedLeft = null;
  let copiesForIsolatedRight = null;
  function deepFreeze(value) {
    if (value && typeof value === 'object' && !Object.isFrozen(value)) {
      Object.freeze(value);
      Object.values(value).forEach(deepFreeze);
    }
    return value;
  }
  function sameProvenance(left, right) {
    return Object.keys(left).length === Object.keys(right).length &&
      Object.keys(left).every(key => left[key] === right[key]);
  }
  function validateHalf(value) {
    const { seeds, stats } = value;
    if (!Number.isSafeInteger(value.graphFrameIndex) || value.graphFrameIndex !== 240 ||
        stats?.frameIndex !== value.graphFrameIndex || value.renderTimesMs.length !== 241 ||
        value.cameraTimesS.length !== 241 || value.cameraTimesS.at(-1) !== 4 || !seeds ||
        !Number.isInteger(seeds.weatherSeed) || seeds.weatherSeed < 0 || seeds.weatherSeed > 0xffffffff ||
        typeof seeds.weatherChanging !== 'boolean' || typeof seeds.timePlaying !== 'boolean' ||
        seeds.freshEngine !== true || seeds.initialRenderCount !== 0 || typeof seeds.source !== 'string' || !seeds.source) {
      fail('isolatedPair produced incomplete or invalid frame and seed evidence');
    }
    for (let frame = 0; frame <= 240; frame++) {
      if (value.renderTimesMs[frame] !== frame * RG_DT_MS || value.cameraTimesS[frame] !== frame / 60) {
        fail(`isolatedPair frame trace differs at ${frame}`);
      }
    }
  }

  const adaptedScenes = new Map([...options.scenes].map(([key, entry]) => {
    const source = entry.default;
    const wrapped = source?.isolatedPair === undefined ? entry : {
      ...entry,
      default: {
        ...source,
        async build(engine, helpers, variant) {
          // Isolated setup builds metadata only; room construction belongs to each measured engine.
          return typeof source.buildMetadata === 'function' ? source.buildMetadata(engine, helpers, variant) : {};
        },
      },
    };
    return [key, wrapped];
  }));
  const runner = createPageRunner({
    ...options,
    scenes: adaptedScenes,
    load: async () => {
      const loaded = await isolatedLoad();
      return loaded;
    },
    topic: 'render-graph',
    collectFacts: collectRgFacts,
    validateRequiredFacts,
    settle,
    capture,
  });
  return {
    run: async input => {
      composed = null;
      composedContext = null;
      generation++;
      return runner.run(input);
    },
    engine: () => runner.engine(),
    dispose() {
      disposed = true;
      generation++;
      runner.dispose();
      for (const engine of [...owned]) disposeOwned(engine);
      activeEngine = null;
    },
  };
}
