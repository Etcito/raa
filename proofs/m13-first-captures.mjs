// M13 first GPU captures (the 'GPU card: first m13 captures, judge and proof' contract): every
// judged m13 variant has a judge verdict recorded (Q14). The verdict may fail here: judge
// failures feed the tuning tasks, whose recapture cards and the full m13 proof (its judge
// check) require judge.exit == 0 at min score 6.
import { judgeRecordCheck } from './lib/p2.mjs';
import { check } from './lib/proof.mjs';

await check('judge-records', () => judgeRecordCheck('m13'));
