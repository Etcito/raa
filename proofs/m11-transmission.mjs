// M11 (chain milestone 5) transmission and refraction with absorption, sorted translucency and
// linked-list OIT proof. --quick and --full are identical: no timing or size threshold (P2).
// The captures and judge records are committed by the GPU card (lookdev-capture.mjs) and only
// read here.
import { apiDocCheck, cargoTest, judgeCheck, lookdevCheck, wasmCheck } from './lib/p2.mjs';
import { check } from './lib/proof.mjs';

await check('unit', () => cargoTest(['raa-transmission', 'raa-oit']));

await check('wasm-check', () => wasmCheck());

await check('api-doc', () => apiDocCheck('m11'));

await check('captures', () => lookdevCheck('m11'));

await check('judge', () => judgeCheck('m11'));
