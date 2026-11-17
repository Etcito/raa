import test from 'node:test';
import assert from 'node:assert/strict';
import { RG_CAPTURE_T_S, ROOM_SUBJECTS, cameraAt, selectionAt } from './lib/scene.js';
import { validateInventory, validateRequiredFacts } from './lib/inventory.js';
import { captionText, collectRgFacts, halfFrameTimes } from './lib/runtime.js';

// Browser-free contract checks of the render-graph page share (contracts rg-room-scene and
// rg-page-facts): the RG_CAMERA_PATH formula, the room subjects in the capture view, the
// inventory rules and the required-facts gate.

test('cameraAt follows RG_CAMERA_PATH at t = 0 and at the capture time', () => {
  // By hand: a = 0.6 + 0.1 t, position = (9 sin a, 3, 9 cos a), target (0, 1, 0), fov 50.
  const hand = t => {
    const a = 0.6 + 0.1 * t;
    return [9 * Math.sin(a), 3, 9 * Math.cos(a)];
  };
  assert.deepEqual(cameraAt(0), { position: hand(0), target: [0, 1, 0], fovDeg: 50 });
  assert.deepEqual(cameraAt(RG_CAPTURE_T_S), { position: hand(RG_CAPTURE_T_S), target: [0, 1, 0], fovDeg: 50 });
  // The captured frame is 240 warm-up frames later: 240 x (1000 / 60) ms = 4 s.
  assert.equal(RG_CAPTURE_T_S, 4);
  assert.deepEqual(cameraAt(4).position, [9 * Math.sin(1), 3, 9 * Math.cos(1)]);
});

const ROW = {
  id: 'interpreter-room-visible',
  criterionName: 'P5-Q2 lit room',
  scenario: 'interpreter',
  mode: 'room',
  filename: 'interpreter-room-lit-shadowed.png',
  expectedVisibleBehavior: 'The lit room with sun shadows under the boxes on the visible floor.',
  requiredFacts: ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:viewSizes', 'runtime:ready', 'assertion:room-visible'],
  pair: null,
  heldConstant: ['room geometry', 'camera'],
  producer: 'raa-graph',
};

// A minimal scene namespace shaped like the interpreter scene of the S61 page.
function fixture() {
  return {
    default: {
      id: 'interpreter',
      milestone: 'interpreter',
      title: 'fixture',
      variants: [{ id: 'room', selection: selectionAt(RG_CAPTURE_T_S) }],
      build() {},
    },
    factNames: ['room-visible'],
    inventory: [ROW],
    observe() {},
  };
}

test('every ROOM_SUBJECTS box overlaps the capture view volume through validateInventory', () => {
  assert.equal(selectionAt(RG_CAPTURE_T_S).subjects, ROOM_SUBJECTS);
  assert.deepEqual(ROOM_SUBJECTS.map(s => s.name),
    ['floor', 'back-wall', 'left-wall', 'box-a', 'box-b', 'box-c', 'sphere', 'emissive-panel']);
  assert.equal(validateInventory([fixture()], 'interpreter').length, 1);
  for (const subject of ROOM_SUBJECTS) {
    const ns = fixture();
    ns.default.variants[0].selection = { ...selectionAt(RG_CAPTURE_T_S), subjects: [subject] };
    assert.equal(validateInventory([ns], 'interpreter').length, 1, subject.name);
  }
});

test('validateInventory refuses a duplicate filename and a non-descriptive filename', () => {
  const duplicate = fixture();
  duplicate.inventory = [ROW, { ...ROW, id: 'interpreter-room-copy' }];
  assert.throws(() => validateInventory([duplicate], 'interpreter'), /invalid inventory row interpreter-room-copy/);
  const vague = fixture();
  vague.inventory = [{ ...ROW, filename: 'capture.png' }];
  assert.throws(() => validateInventory([vague], 'interpreter'), /invalid inventory row/);
});

test('validateInventory refuses an unknown stage and any measurement', () => {
  assert.throws(() => validateInventory([fixture()], 'volume'), /invalid stage or scenes: volume/);
  const measured = fixture();
  measured.default.variants[0].measurement = { pair: 'room', state: 'on', preset: 'room', features: {} };
  assert.throws(() => validateInventory([measured], 'interpreter'), /measurement is not allowed/);
});

function facts() {
  return {
    renderGraphActive: true,
    graph: 'Simple forward',
    issues: [{ code: 'graph.input_missing', severity: 'warning', nodeId: 'forward', port: 'ao', message: 'unlinked', fix: 'link ao' }],
    assertions: [{ name: 'room-visible', passed: true, detail: 'eight subjects inside the view volume' }],
    edits: [{ operation: 'setRenderGraph', accepted: true, error: null }],
    passes: [{ name: 'forward', role: 'pass', enabled: true, order: 60, required: true }],
    viewSizes: [{ view: 0, width: 1280, height: 720 }],
    readiness: { status: 'ready', stage: 'interpreter', error: null },
  };
}

test('validateRequiredFacts accepts valid facts and refuses a false assertion', () => {
  validateRequiredFacts(ROW, facts(), ['room-visible']);
  const failed = { ...facts(), assertions: [{ name: 'room-visible', passed: false, detail: 'back wall outside the view' }] };
  assert.throws(() => validateRequiredFacts(ROW, failed, ['room-visible']),
    error => /required assertion:room-visible is missing or invalid/.test(error.message)
      && error.message.includes('back wall outside the view'));
});

test('halfFrameTimes() is the 241 RG_RUN camera times from 0 to RG_CAPTURE_T_S', () => {
  const times = halfFrameTimes();
  assert.equal(times.length, 241);
  assert.equal(times[0], 0);
  assert.equal(times.at(-1), RG_CAPTURE_T_S);
  assert.ok(times.every((time, frame) => frame === 0 || time > times[frame - 1]));
});

test('captionText names the refused code, node and parameter', () => {
  assert.equal(
    captionText({ code: 'graph.param_range', nodeId: 'aoGtao', param: 'radius', message: 'm', severity: 'error', fix: 'f' }),
    'refused: graph.param_range on node aoGtao (param radius): m');
  assert.equal(
    captionText({ code: 'graph.cycle', nodeId: 'forward', message: 'link cycle', severity: 'error', fix: 'f' }),
    'refused: graph.cycle on node forward: link cycle');
});

test('collectRgFacts maps an observation and yields nulls for absent fields', () => {
  const observation = {
    renderGraphActive: true,
    graph: 'Simple forward',
    issues: [{ code: 'graph.param_range', severity: 'error', nodeId: 'aoGtao', param: 'radius', message: 'm', fix: 'f' }],
    assertions: [{ name: 'room-visible', passed: true, detail: 'eight subjects inside the view volume' }],
    edits: [{ operation: 'setRenderGraph', accepted: true, error: null }],
  };
  assert.deepEqual(collectRgFacts(null, observation, null), observation);
  assert.deepEqual(collectRgFacts(null, {}, null),
    { renderGraphActive: null, graph: null, issues: null, assertions: null, edits: null });
  assert.deepEqual(collectRgFacts(null, null, null),
    { renderGraphActive: null, graph: null, issues: null, assertions: null, edits: null });
});
