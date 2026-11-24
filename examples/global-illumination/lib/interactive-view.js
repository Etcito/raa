import { setCanvasSize, variantTimeMs } from '../../../web/common/page-runner.js';

const movement = new Map([
  ['w', [0, 1]], ['z', [0, 1]], ['arrowup', [0, 1]],
  ['a', [-1, 0]], ['q', [-1, 0]], ['arrowleft', [-1, 0]],
  ['s', [0, -1]], ['arrowdown', [0, -1]],
  ['d', [1, 0]], ['arrowright', [1, 0]],
]);

function attachCamera({ target, canvas, engine, camera, signal, pose }) {
  const offset = (camera.position ?? camera.eye).map((coordinate, axis) => coordinate - camera.target[axis]);
  const homeDistance = Math.hypot(...offset);
  const homeYaw = Math.atan2(offset[0], offset[2]);
  const homePitch = Math.asin(offset[1] / homeDistance);
  const pivot = new Float32Array(pose?.pivot ?? camera.target);
  const held = new Map();
  const options = { signal };
  let distance = pose?.distance ?? homeDistance;
  let yaw = pose?.yaw ?? homeYaw;
  let pitch = pose?.pitch ?? homePitch;
  let drag = null;
  let changed = !!pose;
  let restore = false;

  function clearInput() {
    held.clear();
    if (drag && canvas.hasPointerCapture(drag.id)) canvas.releasePointerCapture(drag.id);
    drag = null;
  }

  function reset() {
    clearInput();
    pivot.set(camera.target);
    distance = homeDistance;
    yaw = homeYaw;
    pitch = homePitch;
    restore = true;
    changed = false;
  }

  canvas.addEventListener('pointerdown', event => {
    if (event.button !== 0 || drag) return;
    canvas.focus({ preventScroll: true });
    canvas.setPointerCapture(event.pointerId);
    drag = { id: event.pointerId, x: event.clientX, y: event.clientY };
  }, options);
  canvas.addEventListener('pointermove', event => {
    if (drag?.id !== event.pointerId) return;
    yaw = (yaw - (event.clientX - drag.x) * 0.005) % (2 * Math.PI);
    pitch += (event.clientY - drag.y) * 0.005;
    const limit = 89 * Math.PI / 180;
    pitch = Math.max(-limit, Math.min(limit, pitch));
    drag.x = event.clientX;
    drag.y = event.clientY;
    changed = true;
  }, options);
  function stopDrag(event) {
    if (drag?.id === event.pointerId) drag = null;
  }
  for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    canvas.addEventListener(type, stopDrag, options);
  }
  canvas.addEventListener('wheel', event => {
    if (event.ctrlKey || event.metaKey || event.deltaY === 0) return;
    event.preventDefault();
    const pixels = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? canvas.clientHeight : 1);
    distance = Math.max(0.1, Math.min(1000, distance * Math.exp(pixels * 0.0015)));
    changed = true;
  }, { signal, passive: false });

  canvas.addEventListener('keydown', event => {
    if (event.ctrlKey || event.metaKey || event.altKey || event.isComposing) return;
    const key = event.key.toLowerCase();
    const direction = movement.get(key);
    if (direction) {
      event.preventDefault();
      held.set(event.code || key, direction);
    } else if (key === 'r' || key === 'f') {
      event.preventDefault();
      if (event.repeat) return;
      if (key === 'r') reset();
      else {
        pivot.set(camera.target);
        changed = true;
      }
    }
  }, options);
  canvas.addEventListener('keyup', event => held.delete(event.code || event.key.toLowerCase()), options);
  canvas.addEventListener('blur', clearInput, options);
  target.addEventListener('blur', clearInput, options);
  target.document.addEventListener('visibilitychange', clearInput, options);
  signal.addEventListener('abort', clearInput, { once: true });

  return {
    reset,
    snapshot: () => ({ pivot: Array.from(pivot), distance, yaw, pitch }),
    update(dt) {
      if (restore) {
        engine.setCamera(camera);
        restore = false;
      }
      let right = 0;
      let forward = 0;
      for (const [x, z] of held.values()) {
        right += x;
        forward += z;
      }
      right = Math.sign(right);
      forward = Math.sign(forward);
      if (dt > 0 && (right || forward)) {
        const step = dt * Math.max(0.5, distance * 0.4) / Math.hypot(right, forward);
        const sin = Math.sin(yaw);
        const cos = Math.cos(yaw);
        pivot[0] += (right * cos - forward * sin) * step;
        pivot[2] += (-right * sin - forward * cos) * step;
        changed = true;
      }
      if (changed) {
        engine.setOrbit(pivot, distance, yaw, pitch);
        changed = false;
      }
    },
  };
}

export function createInteractiveView({ target, canvas, lib, scene, variant, onReady, onError, settings, beforeFrame, onFrame, previous }) {
  const lifetime = new AbortController();
  const { signal } = lifetime;
  const camera = variant.selection.camera;
  const frameOptions = { interactiveCamera: true };
  const display = settings ?? { width: 1280, height: 720, maxFps: 0 };
  const pendingOptions = new Map();
  let request = null;
  let engine;
  let controls;
  let building = true;

  function dispose() {
    if (building) return;
    const current = engine;
    engine = null;
    current?.dispose();
  }

  function stop() {
    lifetime.abort();
    if (request !== null) target.cancelAnimationFrame(request);
    request = null;
    dispose();
  }

  function fail(error) {
    stop();
    onError(error);
  }

  async function start() {
    try {
      setCanvasSize(canvas, display.width, display.height);
      const engineOptions = { ...scene.engine, ...variant.engine };
      const aa = display.antiAliasing ?? 'preset';
      if (aa !== 'preset') engineOptions.msaa = aa === 'msaa' || aa === 'combined' ? 4 : 1;
      engine = await lib.raa.createEngine(canvas, engineOptions);
      if (signal.aborted) return;
      engine.resize(canvas.width, canvas.height);
      const state = await scene.build(engine, lib, variant);
      if (signal.aborted) return;
      await scene.prepare?.(engine, state);
      if (signal.aborted) return;
      engine.setCamera(camera);
      if (aa !== 'preset') engine.setPassEnabled('taa', aa === 'taa' || aa === 'combined');
      for (const [id, enabled] of Object.entries(previous?.options ?? {})) {
        if (!scene.options?.find(option => option.id === id)?.unavailable?.(state)) {
          scene.setOption(engine, state, id, enabled);
        }
      }
      controls = attachCamera({ target, canvas, engine, camera, signal, pose: previous?.pose });

      let appliedExposure;
      if (display.exposureEv !== undefined) engine.setAutoExposure({ enabled: false });
      let sceneReady = !scene.ready;
      let firstTime;
      let previousTime;
      let nextDue = 0;
      let previousCap;
      let framePending = false;

      async function frame(timestamp) {
        request = null;
        if (signal.aborted) return;
        const firstFrame = previousTime === undefined;
        try {
          // Request the next refresh before the fence resolves. Skip it if the GPU still
          // has a frame outstanding so input latency and queued resources stay bounded.
          if (!firstFrame) request = target.requestAnimationFrame(frame);
          if (framePending) return;
          const cap = display.maxFps;
          if (cap !== previousCap) nextDue = timestamp;
          previousCap = cap;
          if (cap && timestamp + 0.1 < nextDue) return;
          if (cap) {
            const period = 1000 / cap;
            nextDue += Math.max(1, Math.floor((timestamp + 0.1 - nextDue) / period) + 1) * period;
          }
          if (canvas.width !== display.width || canvas.height !== display.height) {
            setCanvasSize(canvas, display.width, display.height);
            engine.resize(display.width, display.height);
          }
          for (const [id, enabled] of pendingOptions) scene.setOption(engine, state, id, enabled);
          pendingOptions.clear();
          firstTime ??= timestamp;
          const dt = previousTime === undefined ? 0 : Math.min((timestamp - previousTime) / 1000, 0.05);
          const timeMs = variantTimeMs(variant) + timestamp - firstTime;
          if (display.exposureEv !== appliedExposure) {
            engine.setToneMapping({ exposureEv: display.exposureEv });
            appliedExposure = display.exposureEv;
          }
          controls.update(dt);
          scene.frame?.(engine, timeMs, variant, state, frameOptions);
          beforeFrame?.(engine, state);
          engine.render(timeMs);
          // Some scenes finish setup only after their first GPU frames.
          if (!sceneReady) sceneReady = scene.ready(engine, state, variant).ready;
          if (firstFrame) onReady(state);
          previousTime = timestamp;
          if (signal.aborted) return;
          framePending = true;
          try {
            await engine.submittedWorkDone();
          } finally {
            framePending = false;
          }
          if (signal.aborted) return;
          onFrame?.(engine, target.performance.now(), canvas.width, canvas.height);
          // Keep the first frame's completion boundary before starting steady-state pacing.
          if (firstFrame && !signal.aborted) request = target.requestAnimationFrame(frame);
        } catch (error) {
          if (!signal.aborted) fail(error);
        }
      }
      request = target.requestAnimationFrame(frame);
    } finally {
      building = false;
      if (signal.aborted) dispose();
    }
  }

  const ready = start().catch(error => {
    if (!signal.aborted) fail(error);
  });
  return { stop, ready, cameraPose: () => controls?.snapshot(), reset: () => controls?.reset(), setOption: (id, enabled) => pendingOptions.set(id, enabled) };
}
