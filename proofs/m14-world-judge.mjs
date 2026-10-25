import scene from '../examples/lookdev/scenes/m14-camera-effects.js';
import { readJson } from './lib/proof.mjs';

// Owner acceptance, task 5228: these looks are accepted below the judge minimum.
// Keep the original records, scores and judge exits intact. This exception is
// confined to M14 and does not waive hardware/nonEmpty checks or neon's min 6.
const ACCEPTED = new Set(['world-candy', 'world-ice', 'world-jungle']);

export function m14WorldJudgeCheck(read = readJson) {
  const results = scene.variants.filter(v => v.judge).map(v => {
    let record;
    try { record = read(`proofs/results/lookdev/${scene.id}--${v.id}.json`); }
    catch { record = null; }
    const j = record?.judge;
    const accepted = ACCEPTED.has(v.id) && j != null;
    const judged = j?.exit === 0 && Number.isFinite(j.score) && j.score >= 6
      && j.minScore === 6;
    return { pass: judged || accepted,
      text: `${v.id}=${j?.score ?? 'missing'}${accepted ? ' (owner accepted)' : judged ? '' : ' FAIL'}` };
  });
  return {
    measured: `${results.filter(r => r.pass).length}/${results.length} passed: ${results.map(r => r.text).join(', ')}`,
    threshold: 'neon judge exit 0, score >= 6; candy/ice/jungle explicitly owner accepted',
    ok: results.length > 0 && results.every(r => r.pass),
  };
}
