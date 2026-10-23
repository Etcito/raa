// M8 (chain milestone 2) procedural sky with clouds and image-based lighting proof. --quick and
// --full are identical: no timing or size threshold (P2). The captures and judge records are
// committed by the GPU card (lookdev-capture.mjs) and only read here.
import { apiDocCheck, cargoTest, judgeCheck, lookdevCheck, wasmCheck } from './lib/p2.mjs';
import { check } from './lib/proof.mjs';

await check('unit', () => cargoTest(['raa-sky', 'raa-ibl']));

await check('wasm-check', () => wasmCheck());

await check('api-doc', () => apiDocCheck('m08'));

await check('captures', () => lookdevCheck('m08'));

await check('judge', () => judgeCheck('m08'));
