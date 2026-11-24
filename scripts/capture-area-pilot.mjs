#!/usr/bin/env node
import { resolve } from 'node:path';
import { captureAreaPilot } from '../web/common/render-graph-job.mjs';

const [page, output, extra] = process.argv.slice(2);
if (!page || !output || extra || page === '--help') {
  console.log('Usage: node scripts/capture-area-pilot.mjs <built GI page> <new output directory>');
  if (page !== '--help') process.exitCode = 2;
} else {
  try {
    console.log(await captureAreaPilot(resolve(page), resolve(output)));
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
