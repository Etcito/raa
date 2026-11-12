#!/usr/bin/env node
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  createPageJob, RECORD_FIELDS, ACTIONS, attemptRecords, repositoryRoot, readBuildReceipt, buildKey,
  blankCapture, judgeCapture, JUDGE_GRACE_MS, JUDGE_CONCURRENCY, JUDGE_BUDGET_MS,
} from './page-job.mjs';
import { validateInventory, validateRequiredFacts } from '../../examples/sky-weather/lib/inventory.js';
import { validateFeatureMeasurement } from '../../examples/sky-weather/lib/measure.js';

const job = createPageJob({
  name: 'sky-weather-job',
  topic: 'sky-weather',
  stages: ['foundation', 'atmosphere', 'weather'],
  envPrefix: 'SKY_WEATHER',
  validators: () => ({ validateInventory, validateRequiredFacts }),
  validateMeasurement: validateFeatureMeasurement,
});

export const {
  check, select, clean, render, verifyFresh, fresh, judge, parseArgs, main, STAGES,
  loadScenes, verifyPage, storeRoot,
} = job;

export {
  RECORD_FIELDS, ACTIONS, attemptRecords, repositoryRoot, readBuildReceipt, buildKey,
  blankCapture, judgeCapture, JUDGE_GRACE_MS, JUDGE_CONCURRENCY, JUDGE_BUDGET_MS,
};

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch((error) => {
    process.stderr.write(`sky-weather-job: ${error?.message ?? error}\n`);
    process.exit(1);
  });
}
