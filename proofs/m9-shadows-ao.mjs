// M9 (chain milestone 3) soft cascaded sun shadows, GTAO and contact shadows proof. --quick and
// --full are identical: no timing or size threshold (P2). The captures and judge records are
// committed by the GPU card (lookdev-capture.mjs) and only read here.
import { apiDocCheck, cargoTest, judgeCheck, lookdevCheck, wasmCheck } from './lib/p2.mjs';
import { check } from './lib/proof.mjs';

await check('unit', () => cargoTest(['raa-shadow', 'raa-gtao', 'raa-contact']));

await check('wasm-check', () => wasmCheck());

await check('api-doc', () => apiDocCheck('m09'));

await check('captures', () => lookdevCheck('m09'));

await check('judge', () => judgeCheck('m09'));
