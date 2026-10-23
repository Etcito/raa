// M10 (chain milestone 4) WGSL material functions, layered BRDF (clear coat, sheen), fur shells
// and emissive bloom proof. --quick and --full are identical: no timing or size threshold (P2).
// The captures and judge records are committed by the GPU card (lookdev-capture.mjs) and only
// read here.
import { apiDocCheck, cargoTest, judgeCheck, lookdevCheck, wasmCheck } from './lib/p2.mjs';
import { check } from './lib/proof.mjs';

await check('unit', () => cargoTest(['raa-matfn', 'raa-brdf', 'raa-fur', 'raa-bloom']));

await check('wasm-check', () => wasmCheck());

await check('api-doc', () => apiDocCheck('m10'));

await check('captures', () => lookdevCheck('m10'));

await check('judge', () => judgeCheck('m10'));
