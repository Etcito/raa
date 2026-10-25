// M15 (chain milestone 9) animation hooks proof. No P2 timing or size gates.
// Captures and judge records are produced by the GPU card and only read here.
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import scene from '../examples/lookdev/scenes/m15-waddle.js';
import {
  apiDocCheck, cargoTest, judgeRecordCheck, lookdevCheck, readRecord, wasmCheck,
} from './lib/p2.mjs';
import { check } from './lib/proof.mjs';

// Owner acceptance in task 5287: these exact f00/f06 looks no longer fail checks.
// Preserve the original judge scores, minimum and exits; do not extend acceptance
// to a different image or to missing/invalid judge evidence.
const accepted = new Map([
  ['f00', '2f0668187015c50ffca48b1bd5e0a5547f18735fee229d415f25e6ad0de58a3b'],
  ['f06', '302a4fd1d47124f2215b6f65a8de90b699b901d48b0caa452265f3114d384779'],
]);

function captureMatches(record, variant) {
  const path = `proofs/captures/lookdev/${scene.id}--${variant.id}.png`;
  if (record?.capture !== path) return false;
  try {
    const png = readFileSync(new URL(`../${path}`, import.meta.url));
    return createHash('sha256').update(png).digest('hex') === record.captureSha256;
  } catch {
    return false;
  }
}

await check('unit', () => cargoTest(['raa-deform']));
await check('wasm-check', () => wasmCheck());
await check('api-doc', () => apiDocCheck('m15'));

await check('captures', async () => {
  const result = await lookdevCheck('m15');
  const missing = scene.variants.filter(v => !captureMatches(readRecord(scene.id, v.id), v));
  return {
    measured: `${result.measured}; ${scene.variants.length - missing.length}/${scene.variants.length} PNG hashes match`
      + (missing.length ? ` (missing/mismatched: ${missing.map(v => v.id).join(', ')})` : ''),
    threshold: `${result.threshold}; every capture PNG matches its record`,
    ok: result.ok && missing.length === 0,
  };
});

await check('judge', async () => {
  const verdicts = await judgeRecordCheck('m15');
  if (!verdicts.ok) return verdicts;
  const results = scene.variants.filter(v => v.judge).map(v => {
    const record = readRecord(scene.id, v.id);
    const ownerAccepted = accepted.has(v.id)
      && record.captureSha256 === accepted.get(v.id) && captureMatches(record, v);
    return {
      ok: record.judge.exit === 0 || ownerAccepted,
      text: `${v.id}=${record.judge.score}${ownerAccepted ? ' (owner accepted)' : ''}`,
    };
  });
  return {
    measured: `${results.filter(r => r.ok).length}/${results.length} passed: ${results.map(r => r.text).join(', ')}`,
    threshold: 'valid verdicts at min score 6; original f00/f06 captures explicitly owner accepted',
    ok: results.length > 0 && results.every(r => r.ok),
  };
});
