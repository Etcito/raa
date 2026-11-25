// GPU time of the ray-budget arms at a chosen resolution (no reference, timing only).
// Same graphs and timed passes as gallery-ray-budget.js; the denoisers' work is timed inside
// their consumer pass (area.mask, ssr.replace), like the learned heads.
import { jointTrajectory } from './gallery-joint-trajectories.js';
import { loadGalleryNeural } from './gallery-neural.js';
import { reflectionGraph } from './gallery-reflection-teacher.js';
import { route, timedPasses } from './gallery-ray-budget.js';
import { createGalleryReplay, warmGalleryReplay } from './gallery-sequence.js';

const percentile = (values, p) => values.toSorted((a, b) => a - b)[Math.max(0, Math.ceil(values.length * p) - 1)];

async function timeArm(context, graph, tape, head, size, arm) {
  const { target, raa, canvas } = context;
  const student = await createGalleryReplay(target, raa, canvas, size, tape.heldout, false,
    route(graph, head, arm.budget, undefined, arm.denoise), head === 'area');
  const { engine } = student;
  try {
    if (arm.learned) {
      const model = await loadGalleryNeural(engine);
      if (!model?.manifest.trainedHeads.includes(head)) throw new Error('Missing trained ' + head + ' head');
      const selected = engine.setRenderGraphSelection(route(graph, head, 0, model.id));
      if (!selected.ok) return { ...arm, refused: selected.issues };
    }
    const timed = [...timedPasses[head],
      ...(head === 'area' && arm.denoise?.mode === 'analytic' ? ['area.temporal#areaTemporal', 'area.spatial#areaSpatial'] : [])];
    student.apply(tape.replay[0]);
    await warmGalleryReplay(engine, tape.replay[0].timeMs);
    const samples = new Map();
    for (let round = 0; round < 3; round++) {
      for (const [i, pose] of tape.replay.entries()) {
        student.apply(pose, tape.replay[i - 1]);
        const timing = engine.renderInstrumented(pose.timeMs, { viewId: 'primary', trialId: tape.id,
          presentId: head + '-timing-' + arm.name + '-' + round + '-' + i, inputCutoffMs: target.performance.now() });
        if (timing.error) throw new Error('Timing frame refused');
        await engine.submittedWorkDone();
        const gpu = engine.passGpuTimings();
        if (gpu.passes) {
          const passes = Object.fromEntries(timed.map(name => [name, gpu.passes.find(pass => pass.pass === name)?.ms]));
          if (Object.values(passes).every(Number.isFinite)) {
            samples.set(gpu.frame, { passes, ms: Object.values(passes).reduce((a, b) => a + b, 0) });
          }
        }
      }
    }
    const rows = [...samples.values()];
    if (rows.length < 8) throw new Error('Insufficient timestamp readbacks for ' + arm.name);
    const ms = rows.map(row => row.ms);
    const perPass = Object.fromEntries(timed.map(name => [name, percentile(rows.map(row => row.passes[name]), 0.5)]));
    return { ...arm, n: rows.length, p50: percentile(ms, 0.5), p95: percentile(ms, 0.95), perPass };
  } finally {
    engine.dispose();
  }
}

export async function measureGalleryDenoiseTiming(context, { buildId, size = [1920, 1080], trajectory = 'joint-final-a', denoiseParams = {} } = {}) {
  if (!buildId || !/Windows.+ Chrome\//.test(context.target.navigator.userAgent)) {
    throw new Error('Denoise timing requires a pinned Windows Chrome build');
  }
  const tape = jointTrajectory(trajectory);
  const graph = await reflectionGraph();
  const nrd = { mode: 'nrd', params: denoiseParams }, analytic = { mode: 'analytic', params: denoiseParams };
  const arms = {
    area: [{ name: 'baseline' }, { name: 'learned', learned: true }, { name: 'denoised', denoise: nrd },
      { name: 'analytic', denoise: analytic }, ...[2, 3, 5].map(budget => ({ name: 'extra-rays', budget })),
      { name: 'denoised-rays', budget: 2, denoise: nrd }],
    reflection: [{ name: 'baseline' }, { name: 'learned', learned: true }, { name: 'denoised', denoise: nrd },
      { name: 'analytic', denoise: analytic }, ...[1, 2].map(budget => ({ name: 'extra-rays', budget }))],
  };
  const results = [];
  for (const head of ['area', 'reflection']) {
    for (const arm of arms[head]) {
      results.push({ head, ...await timeArm(context, graph, tape, head, size, arm) });
    }
  }
  return { json: { source: { buildId }, size, trajectory, results }, files: [] };
}
