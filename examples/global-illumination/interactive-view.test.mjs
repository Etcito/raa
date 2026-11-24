import assert from 'node:assert/strict';
import test from 'node:test';
import { createInteractiveView } from './lib/interactive-view.js';

function fixture() {
  const target = Object.assign(new EventTarget(), { document: new EventTarget() });
  const canvas = new EventTarget();
  const pending = new Map();
  const calls = [];
  const camera = { eye: [0, 2, -5], target: [0, 2, 0], fovYDeg: 39 };
  const variant = { selection: { camera }, engine: { msaa: 1 }, timeMs: 500 };
  const engine = {
    resize: (...args) => calls.push(['resize', ...args]),
    setCamera: desc => calls.push(['camera', desc]),
    setOrbit: (pivot, ...pose) => calls.push(['orbit', [...pivot], ...pose]),
    submittedWorkDone: () => Promise.resolve(),
    render: time => calls.push(['render', time]),
    dispose: () => calls.push(['dispose']),
  };
  const scene = {
    engine: { msaa: 4 },
    build: (instance, lib, selected) => calls.push(['build', instance, selected]),
    frame: (instance, time, selected, state, options) => {
      assert.equal(instance, engine);
      assert.equal(selected, variant);
      assert.equal(options.interactiveCamera, true);
      calls.push(['frame', time]);
    },
  };

  let frame = 0;
  let captured = null;
  target.requestAnimationFrame = callback => {
    pending.set(++frame, callback);
    return frame;
  };
  target.cancelAnimationFrame = id => pending.delete(id);
  canvas.focus = () => calls.push(['focus']);
  canvas.setPointerCapture = id => { captured = id; };
  canvas.hasPointerCapture = id => captured === id;
  canvas.releasePointerCapture = () => { captured = null; };
  const options = {
    target, canvas, scene, variant,
    lib: { raa: { createEngine: async (element, settings) => {
      assert.equal(element, canvas);
      assert.deepEqual(settings, { msaa: 1 });
      return engine;
    } } },
    onReady: () => calls.push(['ready']),
    onError: error => calls.push(['error', error]),
  };

  return {
    options, camera, engine, calls, pending,
    tick(time) {
      assert.equal(pending.size, 1);
      const [[id, callback]] = pending;
      pending.delete(id);
      return callback(time);
    },
    input(type, props) {
      return canvas.dispatchEvent(Object.assign(new Event(type, { cancelable: true }), props));
    },
  };
}

test('live frames use the variant camera, apply focused input and release the canvas on stop', async () => {
  const { options, camera, calls, pending, tick, input } = fixture();
  const view = createInteractiveView(options);
  await view.ready;
  assert.deepEqual(calls.at(-1), ['camera', camera]);
  await tick(100);
  assert.deepEqual(calls.slice(-3), [['frame', 500], ['render', 500], ['ready']]);

  assert.equal(input('keydown', { key: 'z', code: 'KeyW' }), false);
  await tick(116);
  const move = calls.at(-3);
  assert.equal(move[0], 'orbit');
  assert.ok(move[1][2] > 0);
  options.canvas.dispatchEvent(new Event('blur'));
  await tick(132);
  assert.equal(calls.filter(call => call[0] === 'orbit').length, 1);

  input('pointerdown', { button: 0, pointerId: 2, clientX: 0, clientY: 0 });
  input('pointermove', { pointerId: 2, clientX: 20, clientY: 12 });
  assert.equal(input('wheel', { deltaY: -100, deltaMode: 0 }), false);
  await tick(148);
  const orbit = calls.at(-3);
  assert.equal(orbit[0], 'orbit');
  assert.ok(orbit[2] < move[2]);
  assert.notEqual(orbit[3], move[3]);
  assert.notEqual(orbit[4], move[4]);

  input('keydown', { key: 'f' });
  await tick(164);
  assert.deepEqual(calls.at(-3)[1], camera.target);
  view.reset();
  await tick(180);
  assert.deepEqual(calls.at(-3), ['camera', camera]);

  assert.equal(options.canvas.hasPointerCapture(2), false);
  input('pointerdown', { button: 0, pointerId: 3, clientX: 0, clientY: 0 });
  assert.equal(options.canvas.hasPointerCapture(3), true);
  view.stop();
  view.stop();
  assert.equal(options.canvas.hasPointerCapture(3), false);
  assert.equal(input('keydown', { key: 'ArrowUp' }), true);
  assert.equal(pending.size, 0);
  assert.equal(calls.filter(call => call[0] === 'dispose').length, 1);
});

test('scene readiness advances alongside normal rendering until setup finishes', async () => {
  const { options, tick, calls } = fixture();
  let checks = 0;
  options.scene.ready = () => {
    assert.equal(calls.at(-1)[0], 'render');
    return { ready: ++checks === 2 };
  };
  const view = createInteractiveView(options);
  await view.ready;
  await tick(100);
  assert.deepEqual(calls.at(-1), ['ready']);
  await tick(116);
  await tick(132);
  assert.equal(checks, 2);
  view.stop();
});

test('stopping during a scene build disposes its engine when the build completes', async () => {
  const { options, engine, calls, pending } = fixture();
  const started = Promise.withResolvers();
  const build = Promise.withResolvers();
  options.scene.build = instance => {
    assert.equal(instance, engine);
    started.resolve();
    return build.promise;
  };
  const view = createInteractiveView(options);
  await started.promise;
  view.stop();
  assert.deepEqual(calls, [['resize', 1280, 720]]);
  build.resolve();
  await view.ready;
  assert.deepEqual(calls, [['resize', 1280, 720], ['dispose']]);
  assert.equal(pending.size, 0);
});

test('a pending GPU frame prevents further submissions and cannot restart a stopped view', async () => {
  const { options, engine, calls, pending, tick } = fixture();
  const fence = Promise.withResolvers();
  engine.submittedWorkDone = () => fence.promise;
  const view = createInteractiveView(options);
  await view.ready;
  const submitted = tick(100);
  assert.equal(pending.size, 0);
  assert.equal(calls.filter(call => call[0] === 'render').length, 1);
  view.stop();
  fence.resolve();
  await submitted;
  assert.equal(pending.size, 0);
});
