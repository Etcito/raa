#!/usr/bin/env node
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createPageJob, runCli } from './page-job.mjs';
import { validateFeatureMeasurement } from '../../examples/sky-weather/lib/measure.js';

const RESOLUTIONS = [64, 96];
const LEVELS = [3, 4, 5];
const PROBE_RAYS = [6, 8];

/** The validators of the built page; the page sources are never imported here. */
async function validators(page) {
  const file = path.join(path.resolve(page), 'lib/inventory.js');
  const namespace = await import(pathToFileURL(file).href);
  if (typeof namespace.validateInventory !== 'function'
    || typeof namespace.validateRequiredFacts !== 'function') {
    throw new Error(`${file}: validateInventory and validateRequiredFacts must be exported`);
  }
  return { validateInventory: namespace.validateInventory, validateRequiredFacts: namespace.validateRequiredFacts };
}

/** A GI_PAIR_RUN record: the PAIR_RUN fields plus the GI quality and allocatedBytes. */
export function validateGiMeasurement(value) {
  validateFeatureMeasurement(value);
  const quality = value.quality;
  if (quality === null || typeof quality !== 'object' || Array.isArray(quality)) {
    throw new Error('gi-job: measurement.quality must be an object');
  }
  const choices = { resolution: RESOLUTIONS, levels: LEVELS, probeRays: PROBE_RAYS };
  for (const [key, allowed] of Object.entries(choices)) {
    if (!allowed.includes(quality[key])) {
      throw new Error(`gi-job: measurement.quality.${key} must be one of ${allowed.join(', ')}`);
    }
  }
  if (!Number.isSafeInteger(value.allocatedBytes) || value.allocatedBytes < 0) {
    throw new Error('gi-job: measurement.allocatedBytes must be a safe non-negative integer');
  }
  return value;
}

/** The global-illumination output lines: the rows of check, the kept/rendered counts of render, then the summary. */
function report(action, stage, result, inventory) {
  const lines = [];
  if (action === 'check') {
    for (const row of inventory) lines.push(`${row.id} ${row.filename}`);
  } else if (action === 'render') {
    const kept = result.kept.length;
    lines.push(`gi-job render: kept ${kept}, rendered ${result.images.length - kept}`);
  }
  lines.push(`gi-job ${action}: ok (${stage}, ${inventory.length} rows)`);
  return lines;
}

const job = createPageJob({
  name: 'gi-job',
  topic: 'global-illumination',
  stages: ['volume', 'lighting', 'virtual'],
  envPrefix: 'GI',
  validators,
  validateMeasurement: validateGiMeasurement,
  keepSameBuildOnly: true,
  judgeHeldConstant: true,
  report,
});

export const {
  check, select, clean, render, verifyFresh, fresh, judge, parseArgs, main, STAGES, ACTIONS,
  RECORD_FIELDS, attemptRecords, repositoryRoot, readBuildReceipt, loadScenes, verifyPage, storeRoot, buildKey,
} = job;

runCli(job, import.meta.url);
