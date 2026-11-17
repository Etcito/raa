import test from 'node:test';
import assert from 'node:assert/strict';
import { boxInView, assertSubjectsInView } from './view-check.mjs';

const camera = { eye: [0, 1.6, 5], target: [0, 1, 0], fovYDeg: 60, near: 0.1, far: 100 };

test('a floor larger than the frame stays in view', () => {
  const floor = { name: 'floor', min: [-50, -0.01, -50], max: [50, 0, 50] };
  assert.equal(boxInView(camera, floor.min, floor.max), true);
  assert.doesNotThrow(() => assertSubjectsInView({ camera, subjects: [floor] }, 'ctx'));
});

test('a box behind the camera is out and is refused by name', () => {
  assert.equal(boxInView(camera, [-1, 0, 8], [1, 1, 9]), false);
  assert.throws(
    () => assertSubjectsInView({ camera, subjects: [{ name: 'back', min: [-1, 0, 8], max: [1, 1, 9] }] }, 'ctx'),
    { message: 'ctx: subject back outside camera frustum' },
  );
});

test('a box beyond far is out', () => {
  assert.equal(boxInView(camera, [-1, 0, -200], [1, 1, -199]), false);
});

test('a box far to the right, beyond the horizontal half-angle, is out', () => {
  assert.equal(boxInView(camera, [40, 0, 0], [41, 1, 0]), false);
});

test('a box containing the eye is in view', () => {
  assert.equal(boxInView(camera, [-0.5, 1, 4.5], [0.5, 2.5, 5.5]), true);
});

test('an invalid camera is refused', () => {
  assert.throws(() => boxInView({ ...camera, near: 0 }, [0, 0, 0], [1, 1, 1]), { message: 'view-check: invalid camera' });
  assert.throws(
    () => assertSubjectsInView({ camera: { ...camera, near: 0 }, subjects: [] }, 'ctx'),
    { message: 'view-check: invalid camera' },
  );
});

test('an invalid subject is refused with its context', () => {
  const subject = patch => ({ name: 'cube', min: [-1, 0, -1], max: [1, 1, 1], ...patch });
  assert.throws(
    () => assertSubjectsInView({ camera, subjects: [subject({ name: ' ' })] }, 'ctx'),
    { message: 'ctx: invalid subject' },
  );
  assert.throws(
    () => assertSubjectsInView({ camera, subjects: [subject({ min: [2, 0, 0] })] }, 'ctx'),
    { message: 'ctx: invalid subject' },
  );
});
