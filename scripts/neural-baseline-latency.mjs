#!/usr/bin/env node
import { readFileSync } from 'node:fs';
import { parseArgs } from 'node:util';

const trialSchema = 'raa.neural-baseline.trial.v1';
const usage = [
  'Usage: node scripts/neural-baseline-latency.mjs <trial.json> [<trial.json> ...]',
  '',
  'Read the baseline fixture download, a single trial, or its baseline job JSON result.',
  'Write per-load keyboard latency in milliseconds as JSON. Percentiles use nearest rank.',
  'Long-animation-frame timing is document presentation feedback. Next RAF is an estimate.',
  'Screen refresh confirmation is unavailable with these browser traces.',
].join('\n');

function requireValue(condition, message) {
  if (!condition) throw new Error(message);
}

function id(value, name) {
  requireValue(typeof value === 'string' && value.trim().length > 0, `missing or invalid ${name}`);
  return value;
}

function timestamp(value, name) {
  requireValue(Number.isFinite(value) && value >= 0, `missing or invalid ${name}`);
  return value;
}

function rows(value, name) {
  requireValue(Array.isArray(value), `missing or invalid ${name}`);
  return value;
}

function insert(map, key, value, name) {
  requireValue(!map.has(key), `duplicate ${name} ${key}`);
  map.set(key, value);
}

function frameKey({ viewId, presentId } = {}) {
  return JSON.stringify([id(viewId, 'viewId'), id(presentId, 'presentId')]);
}

function sameOrigin(value, expected, name) {
  requireValue(timestamp(value, `${name}.timeOriginMs`) === expected, `${name} timeOriginMs mismatch`);
}

function frameStages(frame, name) {
  let prior = timestamp(frame.inputCutoffMs, `${name}.inputCutoffMs`);
  let stopped = false;
  for (const field of [
    'renderStartMs', 'acquireStartMs', 'acquireEndMs', 'submitStartMs', 'submitEndMs',
    'presentSelectedMs', 'presentReturnedMs',
  ]) {
    const value = frame[field];
    if (value === null && field !== 'renderStartMs') {
      stopped = true;
      continue;
    }
    requireValue(!stopped && timestamp(value, `${name}.${field}`) >= prior, `${name} stage order at ${field}`);
    prior = value;
  }
  requireValue((frame.error !== null) === stopped, `${name} stage completion and error disagree`);
  if (stopped) id(frame.error?.code, `${name}.error.code`);
}

function joinTrial(trace) {
  requireValue(trace?.schema === trialSchema, `expected trial schema ${trialSchema}`);
  const trialId = id(trace.trialId, 'trialId');
  const { mode, status, facts } = trace;
  requireValue(mode === 'keyboard' || mode === 'scripted', `${trialId} invalid trial mode`);
  requireValue(['complete', 'render-failed', 'timed-out'].includes(status), `${trialId} invalid trial status`);
  const loadClass = id(trace.loadClass, `${trialId}.loadClass`);
  const timeOriginMs = timestamp(trace.timeOriginMs, `${trialId}.timeOriginMs`);
  const frames = new Map();
  for (const frame of rows(trace.frames, `${trialId}.frames`)) {
    const key = frameKey(frame);
    const name = `${trialId} frame ${key}`;
    requireValue(frame.viewId === 'main' && Number.isSafeInteger(frame.fixtureFrame) &&
      frame.fixtureFrame >= 0 && frame.presentId === String(frame.fixtureFrame), `${name} fixture view or presentId mismatch`);
    requireValue(frame.trialId === trialId, `${name} trialId mismatch`);
    sameOrigin(frame.timeOriginMs, timeOriginMs, name);
    frameStages(frame, name);
    insert(frames, key, frame, 'frame');
  }
  const events = new Map();
  for (const event of rows(trace.events, `${trialId}.events`)) {
    const eventId = id(event?.eventId, 'eventId');
    const name = `${trialId} event ${eventId}`;
    requireValue(event.trialId === trialId, `${name} trialId mismatch`);
    sameOrigin(event.timeOriginMs, timeOriginMs, name);
    requireValue(event.trigger === mode, `${name} trigger does not match trial mode`);
    insert(events, eventId, joinEvent(event, frames, name), `${trialId} event`);
  }
  for (const frame of frames.values()) {
    if (frame.eventId !== null) {
      requireValue(events.get(frame.eventId)?.frame === frame,
        `${trialId} frame ${frameKey(frame)} does not match its eventId ${frame.eventId}`);
    }
  }
  const missingEventIds = rows(trace.missingEventIds, `${trialId}.missingEventIds`).map(value => id(value, 'missingEventIds entry'));
  requireValue(new Set(missingEventIds).size === missingEventIds.length, `${trialId} duplicate missingEventIds entry`);
  requireValue(missingEventIds.every(eventId => !events.has(eventId)), `${trialId} recorded event listed as missing`);
  requireValue(status !== 'complete' || (missingEventIds.length === 0 && events.size > 0 && [...events.values()].every(event =>
    event.frame !== null && event.frameError === null)), `${trialId} complete trial contains an incomplete event`);
  return { trialId, loadClass, timeOriginMs, mode, status, facts, missingEventIds, events: [...events.values()] };
}

function joinEvent(event, frames, name) {
  const { eventId, trigger, viewId, presentId } = event;
  const eventMs = timestamp(event.eventMs, `${name}.eventMs`);
  const acceptedMs = timestamp(event.acceptedMs, `${name}.acceptedMs`);
  requireValue(acceptedMs >= eventMs, `${name} accepted before event timestamp`);
  id(viewId, `${name}.viewId`);
  requireValue(viewId === 'main', `${name} fixture viewId mismatch`);
  const frame = presentId === null ? null : frames.get(frameKey(event));
  requireValue(frame !== undefined, `${name} unknown affected frame ${viewId}/${presentId}`);
  const keyCode = trigger === 'keyboard' ? id(event.keyCode, `${name}.keyCode`) : null;
  const appliedMs = event.appliedMs === null ? null : timestamp(event.appliedMs, `${name}.appliedMs`);
  if (frame !== null) {
    requireValue(frame.eventId === eventId, `${name} affected frame eventId mismatch`);
    requireValue(appliedMs !== null && appliedMs >= acceptedMs && appliedMs <= frame.inputCutoffMs,
      `${name} event application falls outside acceptance and affected frame input cutoff`);
    if (frame.eventMs != null || frame.acceptedMs != null) {
      requireValue(frame.eventMs === eventMs && frame.acceptedMs === acceptedMs,
        `${name} affected frame event timing mismatch`);
    }
  } else {
    requireValue(appliedMs === null, `${name} applied event has no affected frame`);
  }
  const presentation = readPresentation(event, frame, name);
  return {
    eventId, trigger, keyCode, viewId, presentId, eventMs, acceptedMs, appliedMs, frame,
    frameError: frame?.error ?? null,
    acceptanceMs: acceptedMs - eventMs,
    submitMs: frame?.submitEndMs == null ? null : frame.submitEndMs - eventMs,
    selectionMs: frame?.presentSelectedMs == null ? null : frame.presentSelectedMs - eventMs,
    handoffMs: frame?.presentReturnedMs == null ? null : frame.presentReturnedMs - eventMs,
    presentation,
  };
}

function readPresentation(event, frame, name) {
  const feedback = rows(event.presentation, `${name}.presentation`);
  const frameFeedback = frame === null ? [] : rows(frame.browserFeedback?.presentation, `${name} affected frame feedback`);
  requireValue(feedback.length === frameFeedback.length, `${name} presentation count differs from affected frame feedback`);
  const sources = new Set();
  return feedback.map(item => {
    const { source, kind } = item;
    const expectedKind = source === 'long-animation-frame' ? 'browser-feedback' : 'estimate';
    requireValue((source === 'long-animation-frame' || source === 'next-animation-frame') && kind === expectedKind,
      `${name} invalid presentation source or kind`);
    requireValue(!sources.has(source), `${name} duplicate presentation source ${source}`);
    sources.add(source);
    requireValue(frame?.presentReturnedMs != null && frame.error === null &&
      item.trialId === event.trialId && item.eventId === event.eventId &&
      frameKey(item) === frameKey(event), `${name} presentation identity mismatch`);
    sameOrigin(item.timeOriginMs, event.timeOriginMs, `${name} presentation`);
    const atMs = timestamp(item.atMs, `${name} presentation atMs`);
    requireValue(atMs >= frame.presentReturnedMs, `${name} presentation precedes present handoff`);
    const sourceTime = source === 'long-animation-frame'
      ? frame.browserFeedback.documentFrame?.presentationMs : frame.nextCallbackStartMs;
    requireValue(atMs === sourceTime && frameFeedback.some(record =>
      record.source === source && record.kind === kind && record.atMs === atMs),
    `${name} presentation timestamp disagrees with affected frame feedback`);
    return { source, kind, atMs, latencyMs: atMs - event.eventMs };
  });
}

function summarize(values) {
  const sorted = values.filter(value => value !== null).sort((a, b) => a - b);
  const quantile = p => sorted[Math.ceil(p * sorted.length) - 1] ?? null;
  return { count: sorted.length, p50: quantile(0.5), p95: quantile(0.95), p99: quantile(0.99) };
}

function reportClass(trials) {
  const events = trials.flatMap(trial => trial.events);
  const keys = events.filter(event => event.trigger === 'keyboard');
  const feedback = new Map();
  for (const event of keys) {
    for (const { source, kind, latencyMs } of event.presentation) {
      if (!feedback.has(source)) feedback.set(source, { source, kind, samples: [] });
      feedback.get(source).samples.push(latencyMs);
    }
  }
  return {
    loadClass: trials[0].loadClass,
    trialCount: trials.length,
    keyboardEventCount: keys.length,
    scriptedEventCount: events.length - keys.length,
    missingEventCount: trials.reduce((count, trial) => count + trial.missingEventIds.length, 0),
    noAffectedFrameCount: keys.filter(event => event.presentId === null).length,
    failedAffectedFrameCount: keys.filter(event => event.frameError !== null).length,
    noPresentationSampleCount: keys.filter(event => event.presentation.length === 0).length,
    noBrowserReportedPresentationCount: keys.filter(event =>
      !event.presentation.some(row => row.kind === 'browser-feedback')).length,
    keyToAcceptanceMs: summarize(keys.map(event => event.acceptanceMs)),
    keyToSubmitMs: summarize(keys.map(event => event.submitMs)),
    keyToPresentSelectionMs: summarize(keys.map(event => event.selectionMs)),
    keyToPresentHandoffMs: summarize(keys.map(event => event.handoffMs)),
    keyToBrowserPresentationMs: [...feedback.values()].map(({ source, kind, samples }) => ({
      source, kind, latencyMs: summarize(samples),
    })),
    screenRefresh: 'unavailable',
    trials,
  };
}

export function analyze(traces) {
  requireValue(traces.length > 0, 'at least one trial is required');
  const classes = new Map();
  const ids = new Set();
  for (const trace of traces) {
    const trial = joinTrial(trace);
    requireValue(!ids.has(trial.trialId), `duplicate trialId ${trial.trialId}`);
    ids.add(trial.trialId);
    if (!classes.has(trial.loadClass)) classes.set(trial.loadClass, []);
    classes.get(trial.loadClass).push(trial);
  }
  return { schema: 'raa.neural-baseline.latency-report.v1', loadClasses: [...classes.values()].map(reportClass) };
}

function readTrials(path) {
  try {
    const data = JSON.parse(readFileSync(path, 'utf8'));
    const trials = Array.isArray(data) ? data : Array.isArray(data?.json) ? data.json : [data];
    requireValue(trials.length > 0, 'no trials in export');
    return trials;
  } catch (error) {
    throw new Error(`${path}: ${error.message}`);
  }
}

export function main(args) {
  const { values, positionals } = parseArgs({
    args,
    allowPositionals: true,
    options: { help: { type: 'boolean', short: 'h' } },
  });
  if (values.help) return usage;
  requireValue(positionals.length > 0, usage);
  return JSON.stringify(analyze(positionals.flatMap(readTrials)), null, 2);
}

if (import.meta.main) {
  try {
    process.stdout.write(main(process.argv.slice(2)) + '\n');
  } catch (error) {
    process.stderr.write(`neural-baseline-latency: ${error.message}\n`);
    process.exitCode = 1;
  }
}
