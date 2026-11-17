#!/usr/bin/env node
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createPageJob, runCli } from './page-job.mjs';
import { validateFeatureMeasurement } from '../../examples/sky-weather/lib/measure.js';

/** The validators of the built page; the page sources are never imported here. */
export async function validators(page) {
  const file = path.join(path.resolve(page), 'lib/inventory.js');
  const namespace = await import(pathToFileURL(file).href);
  if (typeof namespace.validateInventory !== 'function'
    || typeof namespace.validateRequiredFacts !== 'function') {
    throw new Error(`${file}: validateInventory and validateRequiredFacts must be exported`);
  }
  return { validateInventory: namespace.validateInventory, validateRequiredFacts: namespace.validateRequiredFacts };
}

/** The post-page output lines: the rows of check, the kept/rendered counts of render, then the summary. */
export function report(action, stage, result, inventory) {
  const lines = [];
  if (action === 'check') {
    for (const row of inventory) lines.push(`${row.id} ${row.filename}`);
  } else if (action === 'render') {
    const kept = result.kept.length;
    lines.push(`post-job render: kept ${kept}, rendered ${result.images.length - kept}`);
  }
  lines.push(`post-job ${action}: ok (${stage}, ${inventory.length} rows)`);
  return lines;
}

/**
 * Store and delivery paths. The store is `$NMC_CACHE_DIR/post/<stage>`: `post` is this configuration's
 * topic, and `<stage>` is filled by the `--stage` argument value (exposure, sharpen or mips), a bare
 * stage name that carries no extension. Example: NMC_CACHE_DIR=/cache with `--stage exposure` stores under
 * /cache/post/exposure; with NMC_CACHE_DIR unset the store is the --out directory itself, i.e.
 * path.resolve('out/post/exposure'). Delivery `--out out/post/<stage>` takes the same stage value.
 */
const job = createPageJob({
  name: 'post-job',
  topic: 'post',
  stages: ['exposure', 'sharpen', 'mips'],
  envPrefix: 'POST',
  validators,
  validateMeasurement: validateFeatureMeasurement,
  keepSameBuildOnly: true,
  judgeHeldConstant: true,
  report,
});

export const {
  check, select, clean, render, verifyFresh, fresh, judge, parseArgs, main, STAGES, ACTIONS,
  RECORD_FIELDS, attemptRecords, readBuildReceipt, loadScenes, storeRoot, buildKey,
} = job;

runCli(job, import.meta.url);
