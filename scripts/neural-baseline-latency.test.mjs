import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { analyze, main } from './neural-baseline-latency.mjs';

function trial(trialId = 'test-a', loadClass = 'cpu-bound') {
  const timeOriginMs = 1_800_000_000_000;
  const ids = { trialId, timeOriginMs, viewId: 'main', presentId: '2', eventId: 'camera-edge' };
  const frame = {
    ...ids, fixtureFrame: 2,
    inputCutoffMs: 110,
    renderStartMs: 111,
    acquireStartMs: 112,
    acquireEndMs: 113,
    submitStartMs: 114,
    submitEndMs: 116,
    presentSelectedMs: 117,
    presentReturnedMs: 118,
    error: null,
  };
  const event = {
    ...ids, trigger: 'keyboard', keyCode: 'Space', eventMs: 100, acceptedMs: 105, appliedMs: 109,
    presentation: [],
  };
  setFeedback(event, frame, [{ source: 'next-animation-frame', kind: 'estimate', atMs: 133 }]);
  return {
    schema: 'raa.neural-baseline.trial.v1', trialId, loadClass, timeOriginMs,
    mode: 'keyboard', status: 'complete', facts: {}, missingEventIds: [], events: [event], frames: [frame],
  };
}

function setFeedback(event, frame, feedback) {
  const { trialId, timeOriginMs, viewId, presentId, eventId } = event;
  event.presentation = feedback.map(item => ({ ...item, trialId, timeOriginMs, viewId, presentId, eventId }));
  frame.nextCallbackStartMs = feedback.find(item => item.source === 'next-animation-frame')?.atMs ?? null;
  frame.browserFeedback = {
    presentation: feedback,
    documentFrame: { presentationMs: feedback.find(item => item.source === 'long-animation-frame')?.atMs ?? null },
  };
}

test('joins key timestamps with affected frames and reports distributions per load', () => {
  const a = trial();
  const b = trial('test-b');
  b.events[0].eventMs = 99;
  setFeedback(b.events[0], b.frames[0], [
    { source: 'long-animation-frame', kind: 'browser-feedback', atMs: 127 },
    { source: 'next-animation-frame', kind: 'estimate', atMs: 133 },
  ]);
  const c = trial('test-c', 'gpu-bound');
  setFeedback(c.events[0], c.frames[0], []);
  const [cpu, gpu] = analyze([a, b, c]).loadClasses;
  assert.equal(cpu.trialCount, 2);
  assert.equal(cpu.keyboardEventCount, 2);
  assert.deepEqual(cpu.keyToAcceptanceMs, { count: 2, p50: 5, p95: 6, p99: 6 });
  assert.equal(cpu.keyToSubmitMs.p50, 16);
  assert.equal(cpu.keyToPresentSelectionMs.p50, 17);
  assert.equal(cpu.keyToPresentHandoffMs.p50, 18);
  assert.equal(cpu.noBrowserReportedPresentationCount, 1);
  assert.equal(cpu.trials[0].events[0].keyCode, 'Space');
  assert.deepEqual(cpu.keyToBrowserPresentationMs.map(row => [row.kind, row.latencyMs.p50]), [
    ['estimate', 33], ['browser-feedback', 28],
  ]);
  assert.equal(cpu.keyToBrowserPresentationMs[0].latencyMs.count, 2);
  assert.equal(cpu.keyToBrowserPresentationMs[1].latencyMs.count, 1);
  assert.equal(cpu.screenRefresh, 'unavailable');
  assert.equal(gpu.noPresentationSampleCount, 1);
  assert.deepEqual(gpu.keyToBrowserPresentationMs, []);
});

test('retains failed and missing frame attempts and excludes scripted actions from key statistics', () => {
  const trace = trial();
  const [frame] = trace.frames;
  frame.submitStartMs = frame.submitEndMs = frame.presentSelectedMs = frame.presentReturnedMs = null;
  frame.error = { code: 'test-failure', message: 'test failure' };
  const [event] = trace.events;
  trace.status = 'render-failed';
  trace.missingEventIds = ['flash-on'];
  setFeedback(event, frame, []);
  const missed = trial('test-b');
  missed.status = 'timed-out';
  missed.frames = [];
  Object.assign(missed.events[0], { presentId: null, appliedMs: null, presentation: [] });
  const scripted = trial('test-c');
  scripted.mode = scripted.events[0].trigger = 'scripted';
  const [report] = analyze([trace, missed, scripted]).loadClasses;
  assert.equal(report.keyboardEventCount, 2);
  assert.equal(report.scriptedEventCount, 1);
  assert.equal(report.missingEventCount, 1);
  assert.equal(report.failedAffectedFrameCount, 1);
  assert.equal(report.noAffectedFrameCount, 1);
  assert.deepEqual(report.keyToSubmitMs, { count: 0, p50: null, p95: null, p99: null });
});

test('rejects missing or inconsistent trial, event, view, present and clock identities', () => {
  const cases = [
    ['missing trial', t => delete t.trialId, /invalid trialId/],
    ['trial identity', t => t.frames[0].trialId = 'wrong', /trialId mismatch/],
    ['event trial identity', t => t.events[0].trialId = 'wrong', /trialId mismatch/],
    ['missing present', t => delete t.events[0].presentId, /invalid presentId/],
    ['view identity', t => t.events[0].viewId = 'wrong', /fixture viewId mismatch/],
    ['event identity', t => t.frames[0].eventId = 'wrong', /eventId mismatch/],
    ['presentation kind', t => t.events[0].presentation[0].kind = 'browser-feedback', /invalid presentation source or kind/],
    ['missing clock', t => delete t.frames[0].timeOriginMs, /invalid.*timeOriginMs/],
    ['wrong clock', t => t.frames[0].timeOriginMs++, /timeOriginMs mismatch/],
    ['wrong input order', t => t.events[0].acceptedMs = 112, /outside acceptance and affected frame input cutoff/],
    ['wrong browser feedback order', t => t.events[0].presentation[0].atMs = 102, /precedes present handoff/],
    ['feedback view identity', t => t.events[0].presentation[0].viewId = 'wrong', /presentation identity mismatch/],
    ['feedback clock', t => t.events[0].presentation[0].timeOriginMs++, /timeOriginMs mismatch/],
    ['feedback source mismatch', t => t.frames[0].nextCallbackStartMs = 132, /disagrees with affected frame/],
    ['duplicate frame', t => t.frames.push(t.frames[0]), /duplicate frame/],
    ['duplicate event', t => t.events.push(t.events[0]), /duplicate .*event/],
  ];
  for (const [label, corrupt, error] of cases) {
    const data = trial();
    corrupt(data);
    assert.throws(() => analyze([data]), error, label);
  }
  assert.throws(() => analyze([trial(), trial()]), /duplicate trialId/);
});

test('CLI entry reads trial files and exported arrays and rejects malformed inputs', (t) => {
  const dir = mkdtempSync(join(tmpdir(), 'raa-neural-latency-test-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const single = join(dir, 'trial.json');
  const bundle = join(dir, 'traces.json');
  writeFileSync(single, JSON.stringify(trial()));
  writeFileSync(bundle, JSON.stringify([trial('test-b', 'gpu-bound')]));
  assert.equal(JSON.parse(main([single, bundle])).loadClasses.length, 2);
  assert.match(main(['--help']), /Usage/);
  writeFileSync(bundle, JSON.stringify({ json: [trial()], files: [] }));
  assert.equal(JSON.parse(main([bundle])).loadClasses[0].trialCount, 1);
  writeFileSync(bundle, 'invalid');
  assert.throws(() => main([single, bundle]), /traces.json/);
});
