// M14 (chain milestone 7) camera effects proof: SSR, depth of field, fog, sun shafts and colour
// grading. --quick and --full are identical: no timing or size threshold (P2). The captures and
// judge records are committed by the GPU card (lookdev-capture.mjs) and only read here.
import { apiDocCheck, cargoTest, lookdevCheck, wasmCheck } from './lib/p2.mjs';
import { check } from './lib/proof.mjs';
import { m14WorldJudgeCheck } from './m14-world-judge.mjs';

await check('unit', () => cargoTest(['raa-ssr', 'raa-dof', 'raa-fog', 'raa-shafts', 'raa-grading']));

await check('wasm-check', () => wasmCheck());

await check('api-doc', () => apiDocCheck('m14'));

await check('captures', () => lookdevCheck('m14'));

await check('judge', () => m14WorldJudgeCheck());
