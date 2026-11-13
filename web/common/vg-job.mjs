#!/usr/bin/env node
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createPageJob, runCli } from './page-job.mjs';
import { validateFeatureMeasurement } from '../../examples/sky-weather/lib/measure.js';

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

/** The virtual-geometry output lines: the rows of check, the kept/rendered counts of render, then the summary. */
function report(action, stage, result, inventory) {
  const lines = [];
  if (action === 'check') {
    for (const row of inventory) lines.push(`${row.id} ${row.filename}`);
  } else if (action === 'render') {
    const kept = result.kept.length;
    lines.push(`vg-job render: kept ${kept}, rendered ${result.images.length - kept}`);
  }
  lines.push(`vg-job ${action}: ok (${stage}, ${inventory.length} rows)`);
  return lines;
}

const job = createPageJob({
  name: 'vg-job',
  topic: 'virtual-geometry',
  stages: ['bake', 'passes'],
  envPrefix: 'VG',
  validators,
  validateMeasurement: validateFeatureMeasurement,
  keepSameBuildOnly: true,
  judgeHeldConstant: true,
  report,
});

export const {
  check, select, clean, render, verifyFresh, fresh, judge, parseArgs, main, STAGES, ACTIONS,
  RECORD_FIELDS, attemptRecords, repositoryRoot, readBuildReceipt, loadScenes, verifyPage, storeRoot, buildKey,
} = job;

runCli(job, import.meta.url);
