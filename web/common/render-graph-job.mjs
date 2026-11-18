#!/usr/bin/env node
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createPageJob, runCli } from './page-job.mjs';

/**
 * The rows of the scene delivered at `stage`.
 * The page's `validateInventory` walks the cumulative stage list, so it returns the rows of every
 * earlier stage too; a render-graph stage selects, renders and judges only its own scene's images.
 * `validate` is that page validator, called first so invalid scenes and stages still throw.
 */
export function stageInventory(namespaces, stage, validate) {
  const rows = validate(namespaces, stage);
  const own = new Set(namespaces
    .filter((namespace) => namespace?.default?.milestone === stage)
    .map((namespace) => namespace.default.id));
  return rows.filter((row) => own.has(row.scenario));
}

/** The validators of the built page; the page sources are never imported here. */
async function validators(page) {
  const file = path.join(path.resolve(page), 'lib/inventory.js');
  const namespace = await import(pathToFileURL(file).href);
  if (typeof namespace.validateInventory !== 'function'
    || typeof namespace.validateRequiredFacts !== 'function') {
    throw new Error(`${file}: validateInventory and validateRequiredFacts must be exported`);
  }
  return {
    validateInventory: (scenes, stage) => stageInventory(scenes, stage, namespace.validateInventory),
    validateRequiredFacts: namespace.validateRequiredFacts,
  };
}

/** The render-graph page takes no measurement: any record is refused. */
export function validateMeasurement() {
  throw new Error('render-graph-job: the render-graph page takes no measurement');
}

/** The rows of check, the kept/rendered counts of render, then the summary. */
export function report(action, stage, result, inventory) {
  const lines = [];
  if (action === 'check') {
    for (const row of inventory) lines.push(`${row.id} ${row.filename}`);
  } else if (action === 'render') {
    const kept = result.kept.length;
    lines.push(`render-graph-job render: kept ${kept}, rendered ${result.images.length - kept}`);
  }
  lines.push(`render-graph-job ${action}: ok (${stage}, ${inventory.length} rows)`);
  return lines;
}

const job = createPageJob({
  name: 'render-graph-job',
  topic: 'render-graph',
  stages: ['interpreter', 'views', 'parameters', 'wiring', 'slots', 'gi-stages', 'post-stages', 'compute', 'frozen'],
  envPrefix: 'RG',
  validators,
  validateMeasurement,
  keepSameBuildOnly: true,
  judgeHeldConstant: true,
  report,
});

export const {
  check, select, clean, render, verifyFresh, fresh, judge, parseArgs, main, STAGES, ACTIONS,
  RECORD_FIELDS, attemptRecords, repositoryRoot, readBuildReceipt, loadScenes, verifyPage, storeRoot, buildKey,
} = job;

runCli(job, import.meta.url);
