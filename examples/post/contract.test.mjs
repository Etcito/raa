// Contract checks of the post page (S49, module post-page): the inventory and fact validators on fixtures.
// The real exposure scene namespace is checked by examples/post/exposure.test.mjs.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { SETTING_ORDER, validateInventory, validateRequiredFacts } from './lib/inventory.js';

// A camera at the origin looking down -Z, and a 1 m box five metres in front of it (inside the frustum).
const camera = { eye: [0, 0, 0], target: [0, 0, -1], fovYDeg: 60, near: 0.1, far: 50 };
const selection = { camera, subjects: [{ name: 'box', min: [-0.5, -0.5, -5.5], max: [0.5, 0.5, -4.5] }] };
const settings = { toneMapping: { exposureEv: 0 }, passes: { tonemap: true } };
const on = { pair: 'demo', state: 'on', preset: 'p', features: { exposure: true } };
const off = { pair: 'demo', state: 'off', preset: 'p', features: { exposure: false } };
const onVariant = (overrides = {}) => ({ id: 'demo-on', selection, settings, measurement: on, ...overrides });
const offVariant = () => ({ id: 'demo-off', selection, settings, measurement: off });
const names = ['exposure-settled', 'exposure-ev'];

const onRow = {
  id: 'fixture-room-demo-on',
  criterionName: 'exposure',
  scenario: 'fixture-room',
  mode: 'demo-on',
  filename: 'fixture-room-demo-on.png',
  expectedVisibleBehavior: 'The room is readable.',
  requiredFacts: ['runtime:passes', 'runtime:viewSizes', 'runtime:ready', 'fact:exposure-settled', 'fact:exposure-ev', 'assertion:exposure-settled'],
  pair: 'demo',
  heldConstant: ['room geometry'],
  producer: 'fixture',
};
const offRow = { ...onRow, id: 'fixture-room-demo-off', mode: 'demo-off', filename: 'fixture-room-demo-off.png', expectedVisibleBehavior: 'The room is dark.' };

// A scene namespace (default, inventory, factNames, observe) with the fixture room unless overridden.
function namespace({ id = 'fixture-room', milestone = 'exposure', variants = [onVariant(), offVariant()], inventory = [onRow, offRow], factNames = names } = {}) {
  return {
    default: { id, milestone, title: 'Fixture room', variants, build: async () => ({}) },
    inventory,
    factNames,
    observe: async () => ({ values: {}, assertions: [] }),
  };
}

test('validateInventory returns the rows of the stage in namespace then inventory order', () => {
  assert.deepEqual(validateInventory([namespace()], 'exposure'), [onRow, offRow]);
});

test('an unknown stage is refused', () => {
  assert.throws(() => validateInventory([namespace()], 'blur'), { message: 'invalid stage or scenes: blur' });
});

test('a stage with no scene of that milestone is unavailable', () => {
  assert.throws(() => validateInventory([namespace()], 'sharpen'), { message: 'stage sharpen is unavailable' });
});

test('duplicate scene, variant, row and filename are refused, and a bad filename too', () => {
  assert.throws(() => validateInventory([namespace(), namespace()], 'exposure'), { message: 'invalid or duplicate scene fixture-room' });
  assert.throws(() => validateInventory([namespace({ variants: [onVariant(), onVariant()] })], 'exposure'), { message: 'fixture-room: invalid or duplicate variant' });
  assert.throws(() => validateInventory([namespace({ inventory: [onRow, { ...offRow, id: onRow.id }] })], 'exposure'), { message: 'fixture-room: invalid inventory row fixture-room-demo-on' });
  assert.throws(() => validateInventory([namespace({ inventory: [onRow, { ...offRow, filename: onRow.filename }] })], 'exposure'), { message: 'fixture-room: invalid inventory row fixture-room-demo-off' });
  assert.throws(() => validateInventory([namespace({ inventory: [{ ...onRow, filename: 'Bad.png' }, offRow] })], 'exposure'), { message: 'fixture-room: invalid inventory row fixture-room-demo-on' });
});

test('a measurement state that does not match the variant suffix is refused', () => {
  const variants = [onVariant({ measurement: off }), offVariant()];
  assert.throws(() => validateInventory([namespace({ variants })], 'exposure'), { message: 'fixture-room/demo-on: invalid measurement or variant name' });
});

test('a measurement pair with only an on state is refused', () => {
  assert.throws(() => validateInventory([namespace({ variants: [onVariant()], inventory: [onRow] })], 'exposure'), { message: 'measurement pair demo requires on and off states with the same preset' });
});

test('a subject behind the camera is refused', () => {
  const behind = { camera, subjects: [{ name: 'box', min: [-0.5, -0.5, 4.5], max: [0.5, 0.5, 5.5] }] };
  const variants = [onVariant({ selection: behind }), offVariant()];
  assert.throws(() => validateInventory([namespace({ variants })], 'exposure'), { message: 'fixture-room/demo-on: subject box outside camera frustum' });
});

test('unknown settings, a non-boolean settle and an unknown selector are refused', () => {
  assert.throws(() => validateInventory([namespace({ variants: [onVariant({ settings: { blur: 1 } }), offVariant()] })], 'exposure'), { message: 'fixture-room/demo-on: unknown setting blur' });
  assert.throws(() => validateInventory([namespace({ variants: [onVariant({ settle: 'yes' }), offVariant()] })], 'exposure'), { message: 'fixture-room/demo-on: settle must be a boolean' });
  assert.throws(() => validateInventory([namespace({ inventory: [{ ...onRow, requiredFacts: ['snapshot:x'] }, offRow] })], 'exposure'), { message: 'fixture-room-demo-on: unknown selector snapshot:x' });
});

test('a scene of a later milestone is skipped at an earlier stage and included at its own', () => {
  const sharp = namespace({
    id: 'fixture-sharp',
    milestone: 'sharpen',
    variants: [{ id: 'plain', selection, settings }],
    inventory: [{ ...onRow, id: 'fixture-sharp-plain', scenario: 'fixture-sharp', mode: 'plain', filename: 'fixture-sharp-plain.png', pair: null, requiredFacts: [] }],
    factNames: [],
  });
  assert.deepEqual(validateInventory([namespace(), sharp], 'exposure').map(row => row.id), ['fixture-room-demo-on', 'fixture-room-demo-off']);
  assert.deepEqual(validateInventory([namespace(), sharp], 'sharpen').map(row => row.id), ['fixture-room-demo-on', 'fixture-room-demo-off', 'fixture-sharp-plain']);
});

test('SETTING_ORDER is the rank table of the settings', () => {
  assert.deepEqual(SETTING_ORDER, ['toneMapping', 'passes', 'renderScale', 'autoExposure', 'sharpening']);
});

// Facts of a passing exposure run; each refusal below changes one field of it.
const facts = {
  values: { 'exposure-settled': true, 'exposure-ev': null },
  assertions: [{ name: 'exposure-settled', passed: true, detail: '' }],
  passes: [{ name: 'tonemap', role: 'pass', enabled: true, order: 200, required: true }],
  viewSizes: [{ view: 0, width: 1280, height: 720 }],
  readiness: { status: 'ready', stage: 'exposure', error: null },
};

test('validateRequiredFacts accepts the facts of a passing row', () => {
  assert.equal(validateRequiredFacts(onRow, facts, names), undefined);
});

test('validateRequiredFacts refuses a missing value', () => {
  const missing = { ...facts, values: { 'exposure-settled': true } };
  assert.throws(() => validateRequiredFacts(onRow, missing, names), { message: 'fixture-room-demo-on: required fact:exposure-ev is missing or invalid' });
});

test('validateRequiredFacts refuses a false assertion with its detail', () => {
  const failing = { ...facts, assertions: [{ name: 'exposure-settled', passed: false, detail: 'still moving by 0.4 EV' }] };
  assert.throws(() => validateRequiredFacts(onRow, failing, names), { message: 'fixture-room-demo-on: required assertion:exposure-settled is missing or invalid: still moving by 0.4 EV' });
});

test('validateRequiredFacts refuses a failed readiness', () => {
  const broken = { ...facts, readiness: { status: 'failed', stage: 'exposure', error: 'device lost' } };
  assert.throws(() => validateRequiredFacts(onRow, broken, names), { message: 'fixture-room-demo-on: required runtime:ready is missing or invalid' });
});

test('validateRequiredFacts refuses a facts object without values', () => {
  const noValues = { ...facts };
  delete noValues.values;
  assert.throws(() => validateRequiredFacts(onRow, noValues, names), { message: 'fixture-room-demo-on: invalid PostObservation' });
});
