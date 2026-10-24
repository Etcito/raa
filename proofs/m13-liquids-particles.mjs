// M13 (chain milestone 6) animated liquid surfaces and GPU particles proof. --quick and --full
// are identical: no timing or size threshold (P2). The captures and judge records are committed
// by the GPU card (lookdev-capture.mjs) and only read here.
import { apiDocCheck, cargoTest, judgeCheck, lookdevCheck, wasmCheck } from './lib/p2.mjs';
import { check } from './lib/proof.mjs';

await check('unit', () => cargoTest(['raa-liquid', 'raa-particles']));

await check('wasm-check', () => wasmCheck());

await check('api-doc', () => apiDocCheck('m13'));

await check('captures', () => lookdevCheck('m13'));

await check('judge', () => judgeCheck('m13'));
