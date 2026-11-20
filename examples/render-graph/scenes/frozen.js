// Frozen stage of the render-graph page (contract rg-frozen-scene, module
// render-graph-frozen-scene). Each mode pairs a frozen pipeline with the interpreted catalog
// starter of the same source: the left half is created through createEngine's `pipeline`
// option, the right half renders the starter document applied through setRenderGraph. Both
// halves build the same room, set weather seed 1, stop the weather cycle and pause time, then
// run RG_RUN on their own fresh engine. Nothing is copied: the frozen registry names come from
// catalog.frozen and the interpreted documents from catalog.starters, and a missing entry
// refuses the run. Importing this module has no top-level side effect.
import { RG_CAPTURE_T_S, RG_DT_MS, RG_WARMUP_FRAMES, selectionAt } from '../lib/scene.js';

// The isolated-pair runtime of contract m11-isolated-pair: createRgPageRunner at
// examples/render-graph/lib/runtime.js:180 (its Handoff doc comment at :168 and :175) owns the
// fresh left/right engines, the 241-frame traces and the deep-frozen state.isolatedHalves. This
// module supplies engineOptions, prepare and seeds, and observes those immutable snapshots;
// buildRoom is examples/render-graph/lib/scene.js:112 buildRoom.
const WEATHER_SEED = 1;
const HALVES = ['left', 'right'];
const HELD_CONSTANT = ['room geometry', 'materials', 'lights', 'sky', 'camera', 'seeds'];
const PRODUCER = 'Build the three frozen comparison scenes';
const MODES = [
  { mode: 'simple-forward', frozen: 'simple_forward', starter: 'Simple forward' },
  { mode: 'full-gi', frozen: 'full_gi', starter: 'Full GI' },
  { mode: 'debug', frozen: 'debug', starter: 'Debug' },
];
const FACTS = ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:ready'];

const expectedBehavior = {
  'simple-forward': 'Two halves of the same view of the room, labelled frozen simple_forward (left) and interpreted Simple forward (right). Both show the room correctly lit by the sun: surfaces facing the sun are brighter than the others and each box casts a distinct dark sun shadow on the floor under and beside it; the image is neither black nor washed out and no geometry is missing. The two halves match in brightness, colour and shadowing.',
  'full-gi': 'Two halves of the same view of the room, labelled frozen full_gi (left) and interpreted Full GI (right). Both show the room lit by the sun with soft indirect light: shadowed areas dim but not black, sun shadows under the boxes, a reflection on the glossy floor strip and a faint red tint on the floor near the red wall. The two halves match in brightness, colour and indirect light.',
  debug: 'Two halves of the same view of the room, labelled frozen debug (left) and interpreted Debug (right). Both show the same four-column grid of readable, labelled debug tiles instead of the lit final image: depth, normal, motion, ambient occlusion, area shadow mask, a GI raw-versus-denoised comparison, history length and the final image; every tile is a readable labelled image and both halves show the same tiles and labels.',
};

const row = (mode, fact) => ({
  id: `frozen-${mode}`, criterionName: `P5-Q12 the frozen ${mode} pipeline matches its interpreted starter`,
  scenario: 'render-graph-frozen', mode, filename: `frozen-${mode}-left-interpreted-right.png`,
  expectedVisibleBehavior: expectedBehavior[mode], requiredFacts: [...FACTS, `assertion:${fact}`],
  pair: null, heldConstant: [...HELD_CONSTANT], producer: PRODUCER,
});

export const factNames = ['simple-forward-frozen-pair', 'full-gi-frozen-pair', 'debug-frozen-pair'];
export const inventory = [
  row('simple-forward', factNames[0]),
  row('full-gi', factNames[1]),
  row('debug', factNames[2]),
];

function makeVariant({ mode, frozen, starter }) {
  return {
    id: mode, description: `Frozen ${frozen} (left) beside the interpreted ${starter} starter (right)`,
    references: [], judge: true,
    pair: { left: `frozen ${frozen}`, right: `interpreted ${starter}`, order: [...HALVES] },
    selection: selectionAt(RG_CAPTURE_T_S),
  };
}

function fail(message) { throw new Error(`render-graph frozen: ${message}`); }
const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);

// The catalog metadata of the setup engine: frozen pipeline names and starter documents, never
// a copy of the starter JSON. A required name absent from either list refuses the run.
async function buildMetadata(engine) {
  const catalog = engine.renderGraphCatalog();
  if (!record(catalog)) fail('renderGraphCatalog() must return an object');
  const frozen = Array.isArray(catalog.frozen) ? catalog.frozen : [];
  const starters = Array.isArray(catalog.starters) ? catalog.starters : [];
  const mapping = MODES.map(({ mode, frozen: name, starter }) => {
    if (!frozen.includes(name)) fail(`catalog.frozen has no ${name} pipeline`);
    const entry = starters.find(item => item?.name === starter);
    if (!entry || typeof entry.json !== 'string' || entry.json.length === 0) {
      fail(`catalog.starters has no ${starter} document`);
    }
    return { mode, frozen: name, starter, json: entry.json };
  });
  return {
    frozen: [...frozen],
    starters: starters.map(({ name, json }) => ({ name, json })),
    mapping, assertions: [], edits: [],
  };
}

function mappingFor(state, variant) {
  const entry = state?.mapping?.find(item => item.mode === variant?.id);
  if (!entry) fail(`unknown mode ${variant?.id}`);
  return entry;
}

// The left engine is frozen by createEngine({ pipeline }); the right one omits the option and
// applies the catalog starter in prepare.
function engineOptions(variant, state, half) {
  if (!HALVES.includes(half)) fail(`half must be left or right, not ${String(half)}`);
  const entry = mappingFor(state, variant);
  return half === 'left' ? { pipeline: entry.frozen } : {};
}

function edit(state, result) {
  const [first] = result.issues ?? [];
  state.edits.push({
    operation: 'setRenderGraph',
    accepted: result.ok === true,
    error: result.ok === true ? null : { code: first?.code ?? 'graph.refused', message: first?.message ?? 'refused' },
  });
  return result;
}

// Builds the identical room on the measured engine, then sets the explicit seed and paused
// time; the right half also applies the interpreted catalog starter.
async function prepare(engine, lib, variant, state, half) {
  const entry = mappingFor(state, variant);
  await lib.scene.buildRoom(engine);
  engine.setWeather({ seed: WEATHER_SEED, changing: false });
  engine.setTimeOfDay({ playing: false });
  if (half === 'left') {
    if (engine.renderGraphActive() !== true) fail(`frozen pipeline ${entry.frozen} is not active on the left half`);
    return;
  }
  const result = edit(state, engine.setRenderGraph(entry.json));
  if (result.ok !== true) {
    const [first] = result.issues ?? [];
    fail(`interpreted ${entry.starter} was refused: ${first?.code ?? 'graph.refused'} ${first?.message ?? ''}`.trim());
  }
}

// The getter-observed seed provenance of the freshly created measured engine.
function seeds(engine, _variant, _state, half) {
  const weather = engine.weather();
  const time = engine.timeOfDay();
  const provenance = {
    weatherSeed: weather?.seed,
    weatherChanging: weather?.changing,
    timePlaying: time?.playing,
    freshEngine: true,
    initialRenderCount: 0,
    source: `explicit weather seed ${WEATHER_SEED}, changing false and paused time on a fresh engine rendered zero..${RG_WARMUP_FRAMES} from zero`,
  };
  if (provenance.weatherSeed !== WEATHER_SEED || provenance.weatherChanging !== false || provenance.timePlaying !== false) {
    fail(`the ${half} engine did not keep the explicit weather seed 1, changing false and paused time`);
  }
  return provenance;
}

const sameList = (left, right) => Array.isArray(left) && Array.isArray(right)
  && left.length === right.length && left.every((value, index) => value === right[index]);
const sameSeeds = (left, right) => record(left) && record(right)
  && Object.keys(left).length === Object.keys(right).length
  && Object.keys(left).every(key => left[key] === right[key]);

function validateHalf(value) {
  if (!record(value)) fail('isolatedHalves must hold HalfSnapshot objects');
  if (!HALVES.includes(value.half)) fail(`invalid half ${String(value.half)}`);
  if (!Number.isSafeInteger(value.graphFrameIndex) || value.graphFrameIndex !== RG_WARMUP_FRAMES) {
    fail(`${value.half} graph frame index is not ${RG_WARMUP_FRAMES}`);
  }
  if (value.stats?.frameIndex !== value.graphFrameIndex) fail(`${value.half} stats frame index differs`);
  if (!Array.isArray(value.renderTimesMs) || value.renderTimesMs.length !== RG_WARMUP_FRAMES + 1) {
    fail(`${value.half} render trace does not hold ${RG_WARMUP_FRAMES + 1} times`);
  }
  if (!Array.isArray(value.cameraTimesS) || value.cameraTimesS.length !== RG_WARMUP_FRAMES + 1) {
    fail(`${value.half} camera trace does not hold ${RG_WARMUP_FRAMES + 1} times`);
  }
  for (let frame = 0; frame <= RG_WARMUP_FRAMES; frame++) {
    if (value.renderTimesMs[frame] !== frame * RG_DT_MS || value.cameraTimesS[frame] !== frame / 60) {
      fail(`${value.half} frame trace differs at frame ${frame}`);
    }
  }
  if (value.cameraTimesS.at(-1) !== RG_CAPTURE_T_S) fail(`${value.half} captured camera time is not ${RG_CAPTURE_T_S} s`);
  const seed = value.seeds;
  if (!record(seed) || !Number.isInteger(seed.weatherSeed) || seed.weatherSeed < 0 || seed.weatherSeed > 0xffffffff
      || typeof seed.weatherChanging !== 'boolean' || typeof seed.timePlaying !== 'boolean'
      || seed.freshEngine !== true || seed.initialRenderCount !== 0
      || typeof seed.source !== 'string' || seed.source.length === 0) {
    fail(`${value.half} seed provenance is incomplete`);
  }
}

// The immutable capture evidence of both halves; unequal or missing indices, traces or seeds
// refuse the observation instead of claiming parity from the camera time alone.
function evidence(state) {
  const halves = state?.isolatedHalves;
  if (!Array.isArray(halves) || halves.length !== 2) fail('state.isolatedHalves must hold exactly two halves');
  halves.forEach(validateHalf);
  const [left, right] = halves;
  if (left.half !== 'left' || right.half !== 'right') fail('isolatedHalves must be in left, right rank order');
  if (left.graphFrameIndex !== right.graphFrameIndex) fail('the halves report different graph frame indices');
  if (!sameList(left.renderTimesMs, right.renderTimesMs)) fail('the halves report different render times');
  if (!sameList(left.cameraTimesS, right.cameraTimesS)) fail('the halves report different camera times');
  if (!sameSeeds(left.seeds, right.seeds)) fail('the halves report different seed provenance');
  return halves;
}

function snapshotEvidence(snapshot) {
  return {
    half: snapshot.half, graphFrameIndex: snapshot.graphFrameIndex,
    renderTimesMs: snapshot.renderTimesMs, cameraTimesS: snapshot.cameraTimesS,
    seeds: snapshot.seeds, active: snapshot.active, statsFrameIndex: snapshot.stats?.frameIndex,
  };
}

// Readable labelled tiles of the Debug pair: each half's active grid keeps its labels.
function labeledTiles(halves) {
  return halves.map(snapshot => {
    const grid = (snapshot.graph?.nodes ?? []).find(node => node.type === 'view.grid');
    return grid !== undefined && grid.params?.labels !== false;
  });
}

// The SceneObservation of contract rg-page-facts. It reads only the saved, immutable half
// snapshots for the pair evidence and serializes them into the assertion detail, so
// collectRgFacts keeps the frame and seed proof with the facts.
export function observe(engine, variant, state) {
  const halves = evidence(state);
  const fact = factNames[MODES.findIndex(entry => entry.mode === variant?.id)];
  if (!fact) fail(`unknown mode ${variant?.id}`);
  const labels = variant.id === 'debug' ? labeledTiles(halves) : [];
  const passed = variant.id !== 'debug' || labels.every(Boolean);
  const detail = JSON.stringify({ halfSnapshots: halves.map(snapshotEvidence), labeledTiles: labels });
  const assertions = [...(state.assertions ?? [])];
  assertions.push({ name: fact, passed, detail });
  return {
    renderGraphActive: engine.renderGraphActive(),
    graph: engine.renderGraphActive() ? engine.renderGraph().name : null,
    issues: engine.renderGraphIssues(),
    assertions,
    edits: state.edits ?? [],
  };
}

export default {
  id: 'render-graph-frozen', milestone: 'frozen',
  title: 'Render graph frozen pipelines: frozen beside interpreted starter', engine: {},
  variants: MODES.map(makeVariant),
  build: buildMetadata,
  buildMetadata,
  isolatedPair: { engineOptions, prepare, seeds },
};
