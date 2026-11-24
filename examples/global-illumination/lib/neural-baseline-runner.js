import { browserFrameFeedback, observeBrowserFrameFeedback } from '../../../web/common/browser-frame-feedback.js';
import { setCanvasSize } from '../../../web/common/page-runner.js';
import {
  applyBaselineEvent, assertNeuralBaselineReady, BASELINE_EVENTS, BASELINE_FEATURES, BASELINE_FIXTURE,
  BASELINE_FRAME_MS, BASELINE_FRAMES, BASELINE_WARMUP_FRAMES, buildNeuralBaseline,
} from './neural-baseline.js';

const VIEW_ID = 'main';
const KEY = 'Space';
const EVENT_SPACING = 16;
const TIMEOUT = Symbol('baseline timeout');

function optionsFor(input, target) {
  if (input != null && (typeof input !== 'object' || Array.isArray(input))) {
    throw new Error('neural baseline: invalid run options');
  }
  const {
    captureId = target.crypto.randomUUID(), mode = 'scripted', loadClass = 'idle',
    repeat = mode === 'keyboard' ? 1 : 2, width = 1280, height = 720,
    remainingMs = 120000, refreshHz = null, powerMode = null,
  } = input ?? {};
  if (!['scripted', 'keyboard'].includes(mode) || typeof captureId !== 'string' || !captureId.trim()
      || typeof loadClass !== 'string' || !loadClass.trim()
      || !Number.isInteger(repeat) || repeat < 1 || repeat > 12
      || ![width, height].every(size => Number.isInteger(size) && size > 0 && size <= 4096)
      || !Number.isFinite(remainingMs) || remainingMs <= 0 || remainingMs > 300000
      || (refreshHz !== null && (!Number.isFinite(refreshHz) || refreshHz <= 0))
      || (powerMode !== null && (typeof powerMode !== 'string' || !powerMode.trim()))) {
    throw new Error('neural baseline: invalid run options');
  }
  return { captureId, mode, loadClass, repeat, width, height, remainingMs, refreshHz, powerMode };
}

function createFrameClock(target, remainingMs) {
  const now = () => target.performance.now();
  let request = null;
  let expired = false;
  const timer = target.setTimeout(() => {
    expired = true;
    request?.cancel();
  }, remainingMs);
  return {
    next() {
      if (expired) return Promise.reject(TIMEOUT);
      return new Promise((resolve, reject) => {
        const id = target.requestAnimationFrame(rafTimeMs => {
          request = null;
          if (target.document.visibilityState !== 'visible') {
            reject(new Error('neural baseline: the document must stay visible'));
            return;
          }
          resolve({ rafTimeMs, callbackStartMs: now() });
        });
        request = {
          cancel() {
            target.cancelAnimationFrame(id);
            request = null;
            reject(TIMEOUT);
          },
        };
      });
    },
    stop() {
      target.clearTimeout(timer);
      request?.cancel();
    },
  };
}

// One accepted key at a time. Every accepted input is either used by a single attempt or
// exported as a miss if the trial ends before that attempt.
function keyInput(canvas, now, events, identity, nextEvent, isArmed) {
  function onKey(event) {
    if (!event.isTrusted || event.code !== KEY || event.repeat || event.isComposing
        || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    event.preventDefault();
    const plan = nextEvent();
    if (!plan || !isArmed()) return;
    const acceptedMs = now();
    if (!Number.isFinite(event.timeStamp) || event.timeStamp < 0 || event.timeStamp > acceptedMs) return;
    events.push({
      eventId: plan.eventId, trigger: 'keyboard', keyCode: event.code, eventMs: event.timeStamp, acceptedMs,
      ...identity, viewId: VIEW_ID, presentId: null, appliedMs: null, presentation: [],
    });
  }
  canvas.addEventListener('keydown', onKey);
  return () => canvas.removeEventListener('keydown', onKey);
}

async function playTrial({ target, canvas, status, raa, clock, options, trialId }) {
  const { mode, loadClass, width, height, refreshHz, powerMode } = options;
  const now = () => target.performance.now();
  const timeOriginMs = target.performance.timeOrigin;
  const identity = { trialId, timeOriginMs };
  const events = [];
  const frames = [];
  const rafIntervalsMs = [];
  let trialStatus = 'complete';
  let engine;
  let stopKeys;
  let observer;
  let facts = null;
  let frame = 0;
  let eventIndex = 0;
  let lastEventFrame = -EVENT_SPACING;
  let lastRafTimeMs = null;
  let previousRenderEndMs = 0;
  let lastTrace = null;

  function observeRaf(tick) {
    if (lastRafTimeMs !== null) rafIntervalsMs.push(tick.rafTimeMs - lastRafTimeMs);
    lastRafTimeMs = tick.rafTimeMs;
    if (lastTrace !== null) {
      lastTrace.nextRafTimeMs = tick.rafTimeMs;
      lastTrace.nextCallbackStartMs = tick.callbackStartMs;
      lastTrace = null;
    }
  }

  try {
    status.textContent = `Preparing baseline trial ${trialId}`;
    setCanvasSize(canvas, width, height);
    engine = await raa.createEngine(canvas, { msaa: 1, powerPreference: 'high-performance' });
    if (typeof engine.renderInstrumented !== 'function') throw new Error('neural baseline: rebuild RAA with renderInstrumented');
    const state = await buildNeuralBaseline(engine);
    for (let index = 0; index < BASELINE_WARMUP_FRAMES; index++) {
      await clock.next();
      engine.render(index * BASELINE_FRAME_MS);
    }
    previousRenderEndMs = now();
    const areaShadows = assertNeuralBaselineReady(engine);
    const { adapter, hardware } = engine.stats();
    observer = observeBrowserFrameFeedback(target);
    facts = {
      fixture: BASELINE_FIXTURE, features: BASELINE_FEATURES, areaShadows, adapter, hardware,
      browser: target.navigator.userAgent, devicePixelRatio: target.devicePixelRatio,
      sourceSize: [width, height], displaySize: [width, height],
      canvasRect: canvas.getBoundingClientRect().toJSON(), refreshHz, powerMode,
      queuePressure: 'unmeasured', screenRefresh: 'unavailable', supportedFeedback: observer.types,
    };

    if (mode === 'keyboard') {
      status.textContent = `Press Space for event 1 of ${BASELINE_EVENTS.length}`;
      canvas.focus({ preventScroll: true });
      stopKeys = keyInput(canvas, now, events, identity, () => BASELINE_EVENTS[eventIndex],
        () => events.length === eventIndex && frame - lastEventFrame >= EVENT_SPACING);
    } else {
      status.textContent = 'Running scripted baseline';
    }

    while (mode === 'scripted' ? frame < BASELINE_FRAMES
      : eventIndex < BASELINE_EVENTS.length || frame - lastEventFrame <= EVENT_SPACING) {
      const tick = await clock.next();
      observeRaf(tick);
      const plan = BASELINE_EVENTS[eventIndex];
      let event = mode === 'keyboard' ? events[eventIndex] : null;
      if (mode === 'scripted' && plan?.frame === frame) {
        const eventMs = now();
        event = { eventId: plan.eventId, trigger: 'scripted', eventMs, acceptedMs: eventMs,
          ...identity, viewId: VIEW_ID, presentId: null, appliedMs: null, presentation: [] };
        events.push(event);
      }
      const presentId = String(frame);
      const nowMs = (BASELINE_WARMUP_FRAMES + frame) * BASELINE_FRAME_MS;
      if (event) {
        event.appliedMs = now();
        applyBaselineEvent(engine, state, event.eventId);
        event.presentId = presentId;
        lastEventFrame = frame;
        eventIndex++;
      }
      if (mode === 'scripted' || event) {
        const context = {
          viewId: VIEW_ID, trialId, presentId, inputCutoffMs: now(),
          eventId: event?.eventId ?? null, eventMs: event?.eventMs ?? null, acceptedMs: event?.acceptedMs ?? null,
        };
        const trace = engine.renderInstrumented(nowMs, context);
        if (trace.viewId !== VIEW_ID || trace.trialId !== trialId || trace.presentId !== presentId
            || trace.inputCutoffMs !== context.inputCutoffMs || trace.timeOriginMs !== timeOriginMs
            || trace.eventId !== context.eventId || trace.eventMs !== context.eventMs || trace.acceptedMs !== context.acceptedMs
            || trace.nowMs !== nowMs || (trace.error === null && trace.frameIndex !== BASELINE_WARMUP_FRAMES + frame)) {
          throw new Error('neural baseline: render telemetry does not match the requested frame');
        }
        lastTrace = {
          ...trace, ...tick, previousRenderEndMs, fixtureFrame: frame,
          nextRafTimeMs: null, nextCallbackStartMs: null,
        };
        frames.push(lastTrace);
        if (trace.error !== null) {
          trialStatus = 'render-failed';
          break;
        }
      } else {
        engine.render(nowMs);
      }
      previousRenderEndMs = now();
      frame++;
      if (mode === 'keyboard' && eventIndex < BASELINE_EVENTS.length && frame - lastEventFrame === EVENT_SPACING) {
        status.textContent = `Press Space for event ${eventIndex + 1} of ${BASELINE_EVENTS.length}`;
      }
    }
    stopKeys?.();
    observeRaf(await clock.next());
    // Event/frame observers may receive the document's presentation feedback after the next callback.
    await clock.next();
  } catch (error) {
    if (error !== TIMEOUT) throw error;
    if (trialStatus === 'complete') trialStatus = 'timed-out';
  } finally {
    stopKeys?.();
    const entries = observer?.stop() ?? [];
    engine?.dispose();
    const eventByPresent = new Map(events.filter(event => event.presentId !== null)
      .map(event => [event.presentId, event]));
    for (const trace of frames) {
      const event = eventByPresent.get(trace.presentId);
      trace.browserFeedback = browserFrameFeedback(trace, event, entries);
      if (event) {
        event.presentation = trace.browserFeedback.presentation.map(item => ({
          ...item, ...identity, viewId: trace.viewId, presentId: trace.presentId, eventId: event.eventId,
        }));
      }
    }
  }
  return {
    schema: 'raa.neural-baseline.trial.v1', trialId, loadClass, timeOriginMs, mode, facts,
    status: trialStatus, events, frames, rafIntervalsMs,
    missingEventIds: BASELINE_EVENTS.filter(plan => !events.some(event => event.eventId === plan.eventId))
      .map(plan => plan.eventId),
  };
}

export async function runNeuralBaseline({ target, canvas, status, raa }, input) {
  const options = optionsFor(input, target);
  const clock = createFrameClock(target, options.remainingMs);
  const trials = [];
  try {
    for (let index = 0; index < options.repeat; index++) {
      const trialId = `${options.captureId}:${index + 1}`;
      const trial = await playTrial({ target, canvas, status, raa, clock, options, trialId });
      trials.push(trial);
      if (trial.status !== 'complete') break;
    }
    const signatures = trials.map(trial => JSON.stringify(trial.frames.map(frame => [
      frame.viewId, frame.presentId, frame.eventId, frame.frameIndex,
    ])));
    if (options.mode === 'scripted' && trials.every(trial => trial.status === 'complete')
        && signatures.some(signature => signature !== signatures[0])) {
      throw new Error('neural baseline: scripted trials returned different presentation and event IDs');
    }
    status.textContent = trials.every(trial => trial.status === 'complete')
      ? 'Baseline complete'
      : `Baseline ${trials.at(-1).status}`;
    return trials;
  } finally {
    clock.stop();
  }
}
