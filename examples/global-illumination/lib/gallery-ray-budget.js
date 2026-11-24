import { digest } from './gallery-path-model.js';
import { jointTrajectory } from './gallery-joint-trajectories.js';
import { loadGalleryNeural } from './gallery-neural.js';
import { reflectionGraph } from './gallery-reflection-teacher.js';
import { createGalleryReplay, decodeGalleryTap, warmGalleryReplay } from './gallery-sequence.js';

const size = [160, 90];
const pixels = size[0] * size[1];
const budgetSteps = { reflection: [1, 2, 4, 8], area: [2, 3, 5, 9, 16] };
const timedPasses = {
  reflection: ['ssr.hiz#hiz', 'ssr.trace#ssrTrace', 'ssr.incident#incident', 'ssr.replace#replace'],
  area: ['area.trace#trace', 'area.mask#mask', 'forward', 'ssr.replace#replace'],
};

function route(base, head, budget, model) {
  const graph = structuredClone(base);
  if (budget) {
    const target = graph.nodes.find(node => node.id === (head === 'area' ? 'trace' : 'incident'));
    target.params = head === 'area' ? { samplesPerPixel: budget } : { extraRays: budget };
  }
  if (model) {
    const [id, type, key, src, dst] = head === 'area'
      ? ['areaModel', 'neural.area-reconstruction', 'pairs', 'trace.raw', 'mask.src']
      : ['reflectionModel', 'neural.reflection-reconstruction', 'incident', 'incident.incident', 'replace.incident'];
    graph.nodes.push({ id, type, params: { model } });
    graph.links = graph.links.filter(link => link.to !== dst);
    graph.links.push({ from: src, to: id + '.' + key }, { from: id + '.' + key, to: dst });
  }
  return graph;
}

async function fetchPlane(root, record, count) {
  const response = await fetch(new URL(record.file, root));
  if (!response.ok) throw new Error('Missing independent ray teacher ' + record.file);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.byteLength !== record.byteLength || await digest(bytes) !== record.sha256) {
    throw new Error('Independent ray teacher hash mismatch ' + record.file);
  }
  return decodeGalleryTap(bytes, count, false);
}

async function reflectionLabels(id, replay) {
  const root = new URL('../shared/ray-teachers/' + id + '/', import.meta.url);
  const response = await fetch(new URL('manifest.json', root));
  if (!response.ok) throw new Error('Stage the independent ray labels for ' + id);
  const manifest = await response.json(), [clip] = manifest.sequences;
  const replayHash = await digest(new TextEncoder().encode(JSON.stringify(replay)));
  if (manifest.format !== 'raa.gallery-temporal-sequence/1' || clip.id !== id
      || clip.replaySha256 !== replayHash || clip.sceneSize.join() !== size.join()) {
    throw new Error('Independent ray teacher does not match the replay');
  }
  const frames = await Promise.all(clip.frames.map(async ({ planes, counts, jitterPx }) => ({
    target: await fetchPlane(root, planes.reflectionTarget, pixels * 3),
    trace: await fetchPlane(root, planes.reflectionHit, pixels * 4),
    meta: await fetchPlane(root, planes.reflectionMeta, pixels * 4),
    counts, jitterPx,
  })));
  return { source: manifest.source.buildId, replayHash, frames };
}

function reflectionScore(frame, baseline, label) {
  let lanes = 0, offscreen = 0, changed = 0, squared = 0, logSquared = 0;
  const { trace, meta, incident } = frame;
  for (let p = 0; p < pixels; p++) {
    const k = p * 4, match = (a, b) => a.subarray(k, k + 4).every((value, i) => value === b[k + i]);
    if (!match(trace, baseline.trace) || !match(meta, baseline.meta)
        || incident[k + 3] !== baseline.incident[k + 3]) {
      throw new Error('First-ray hit identity or confidence changed at pixel ' + p);
    }
    const accepted = meta[k] === 1 && meta[k + 1] > 0.5 && incident[k + 3] > 0;
    if (!accepted) {
      offscreen += Number(meta[k + 1] === 0 && meta[k + 2] === -2);
      if (!match(incident, baseline.incident)) throw new Error('Incident radiance changed outside accepted screen hits at pixel ' + p);
      continue;
    }
    if (!match(trace, label.trace) || !match(meta, label.meta)) {
      throw new Error('Current first ray does not match the held-out teacher at pixel ' + p);
    }
    for (let c = 0; c < 3; c++) {
      const actual = incident[k + c], expected = label.target[3 * p + c];
      changed += Number(actual !== baseline.incident[k + c]);
      squared += (actual - expected) ** 2;
      logSquared += (Math.log1p(Math.max(0, actual)) - Math.log1p(Math.max(0, expected))) ** 2;
      lanes++;
    }
  }
  return { lanes, offscreen, changed, squared, logSquared };
}

function areaScore(pairs, reference) {
  let lanes = 0, squared = 0, weightSquared = 0;
  for (let k = 0; k < pairs.length; k += 2) {
    if (reference[k + 1] <= 0) continue;
    const visibility = pairs[k + 1] > 1e-8 ? pairs[k] / pairs[k + 1] : 1;
    const target = reference[k] / reference[k + 1];
    squared += (visibility - target) ** 2;
    weightSquared += (pairs[k + 1] - reference[k + 1]) ** 2;
    lanes++;
  }
  return { lanes, squared, weightSquared };
}

const percentile = (values, p) => values.toSorted((a, b) => a - b)[Math.ceil(values.length * p) - 1];
function summarize(frames, samples, head) {
  const sum = (rows, key) => rows.reduce((s, row) => s + (row[key] ?? 0), 0);
  const combine = rows => {
    const lanes = sum(rows, 'lanes');
    if (!lanes) throw new Error('Replay has no valid comparison lanes');
    return { lanes, rmse: Math.sqrt(sum(rows, 'squared') / lanes),
      ...(head === 'reflection' ? { logRmse: Math.sqrt(sum(rows, 'logSquared') / lanes),
        offscreen: sum(rows, 'offscreen'), changed: sum(rows, 'changed') }
        : { weightRmse: Math.sqrt(sum(rows, 'weightSquared') / lanes) }) };
  };
  const ms = samples.map(sample => sample.ms);
  if (ms.length < 8) throw new Error('Insufficient timestamp readbacks');
  return { overall: combine(frames),
    motion: combine(frames.filter(frame => !frame.event)),
    lighting: combine(frames.filter(frame => frame.event)),
    timing: { n: ms.length, p50: percentile(ms, 0.50), p95: percentile(ms, 0.95), samples },
  };
}

async function areaLabels(context, graph, tape) {
  const { target } = context;
  const student = await createGalleryReplay(target, context.raa, context.canvas, size, tape.heldout, false,
    route(graph, 'area', 16), true);
  try {
    student.apply(tape.replay[0]);
    await warmGalleryReplay(student.engine, tape.replay[0].timeMs);
    // Skip the low-budget sequence phases. Each teacher batch visits two complete groups of
    // four strata at disjoint phases and observes the exact same pose and panel state.
    await warmGalleryReplay(student.engine, tape.replay[0].timeMs);
    const frames = [];
    for (const [i, pose] of tape.replay.entries()) {
      student.apply(pose, tape.replay[i - 1]);
      const batches = [];
      for (let batch = 0; batch < 2; batch++) {
        const accumulated = new Float64Array(pixels * 4);
        for (let j = 0; j < 8; j++) {
          student.engine.render(pose.timeMs);
          await student.engine.submittedWorkDone();
          const taps = await student.engine.captureTrainingTaps(student.engine.stats().frames - 1, false, true);
          const pairs = decodeGalleryTap(taps.areaPairs, accumulated.length, true);
          pairs.forEach((value, k) => { accumulated[k] += value / 8; });
        }
        batches.push(accumulated);
      }
      const convergence = areaScore(batches[0], batches[1]);
      frames.push({
        target: Float64Array.from(batches[0], (value, k) => (value + batches[1][k]) / 2),
        batchRmse: Math.sqrt(convergence.squared / convergence.lanes),
      });
    }
    return { frames, samplesPerBatch: 128 };
  } finally {
    student.engine.dispose();
  }
}

async function measureCase(context, graph, tape, head, settings) {
  const { target, raa, canvas } = context;
  const student = await createGalleryReplay(target, raa, canvas, size, tape.heldout, false,
    route(graph, head, settings.budget), head === 'area');
  const { engine } = student;
  try {
    let model;
    if (settings.learned) {
      model = await loadGalleryNeural(engine);
      if (!model?.manifest.trainedHeads.includes(head)) throw new Error('Missing trained ' + head + ' head');
      const selected = engine.setRenderGraphSelection(route(graph, head, 0, model.id));
      if (!selected.ok) throw new Error('Learned route refused ' + JSON.stringify(selected.issues));
    }
    student.apply(tape.replay[0]);
    await warmGalleryReplay(engine, tape.replay[0].timeMs);
    const frames = [], samples = new Map();
    for (const [i, pose] of tape.replay.entries()) {
      student.apply(pose, tape.replay[i - 1]);
      const timing = engine.renderInstrumented(pose.timeMs, { viewId: 'primary', trialId: tape.id,
        presentId: head + '-' + settings.name + '-' + i, inputCutoffMs: target.performance.now() });
      if (timing.error || !Number.isInteger(timing.frameIndex)) throw new Error('Ray replay frame refused');
      await engine.submittedWorkDone();
      const taps = await engine.captureTrainingTaps(timing.frameIndex, false, head === 'area');
      const gpu = engine.passGpuTimings();
      const first = frames[0]?.frameId ?? timing.frameIndex;
      if (gpu.frame >= first && gpu.frame <= timing.frameIndex && gpu.passes) {
        const passes = Object.fromEntries(timedPasses[head].map(name => [name,
          gpu.passes.find(pass => pass.pass === name)?.ms]));
        if (Object.values(passes).every(Number.isFinite)) {
          samples.set(gpu.frame, { frame: gpu.frame, passes,
            ms: Object.values(passes).reduce((total, ms) => total + ms, 0) });
        }
      }
      if (settings.learned && !engine.neuralReconstructionStats()?.ran[head]) {
        throw new Error('Learned ' + head + ' head did not publish');
      }
      frames.push({ frameId: timing.frameIndex, event: pose.event,
        ...(head === 'area' ? { pairs: decodeGalleryTap(
          settings.learned ? taps.areaReconstruction : taps.areaPairs, pixels * 4, true) }
          : { trace: decodeGalleryTap(taps.reflectionTrace, pixels * 4, true),
            meta: decodeGalleryTap(taps.reflectionMeta, pixels * 4, false),
            incident: decodeGalleryTap(settings.learned ? taps.reflectionReconstruction : taps.reflectionIncident, pixels * 4, false) }),
      });
    }
    return { frames, samples: [...samples.values()], facts: student.facts, weights: model?.manifest.weightsSha256 };
  } finally {
    engine.dispose();
  }
}

export async function measureGalleryRayBudget(context, { buildId, head, trajectory, budget } = {}) {
  if (!buildId || !/Windows.+ Chrome\//.test(context.target.navigator.userAgent) || !Object.hasOwn(budgetSteps, head)) {
    throw new Error('Ray-budget replay requires a pinned Windows Chrome build and a known head');
  }
  const tape = jointTrajectory(trajectory);
  const development = tape.partition === 'development';
  if ((!development && tape.partition !== 'test') || (!development && !budgetSteps[head].includes(budget))) {
    throw new Error('Final replay needs an extra-ray count frozen from development');
  }
  const graph = await reflectionGraph();
  const label = head === 'reflection' ? await reflectionLabels(trajectory, tape.replay)
    : await areaLabels(context, graph, tape);
  const baseline = await measureCase(context, graph, tape, head, { name: 'baseline', budget: 0 });
  const results = [];
  let weights;
  for (const settings of [
    { name: 'baseline', budget: 0 }, { name: 'learned', learned: true },
    ...(development ? budgetSteps[head] : [budget]).map(budget => ({ name: 'extra-rays', budget })),
  ]) {
    const run = settings.name === 'baseline' ? baseline : await measureCase(context, graph, tape, head, settings);
    weights ??= run.weights;
    const frames = run.frames.map((frame, i) => ({ frame: frame.frameId, event: frame.event,
      ...(head === 'area' ? areaScore(frame.pairs, label.frames[i].target)
        : reflectionScore(frame, baseline.frames[i], label.frames[i])) }));
    results.push({ ...settings, ...summarize(frames, run.samples, head), frames });
  }
  const learned = results[1];
  const byDistance = (a, b) => Math.abs(a.timing.p50 - learned.timing.p50) - Math.abs(b.timing.p50 - learned.timing.p50);
  const closest = results.slice(2).toSorted(byDistance)[0];
  const relativeTimeDifference = (closest.timing.p50 - learned.timing.p50) / learned.timing.p50;
  if (head === 'reflection' && !(learned.overall.changed > 0 && learned.overall.offscreen > 0)) {
    throw new Error('Replay did not exercise learned radiance and offscreen rejection');
  }
  return { json: { source: { buildId, facts: baseline.facts }, trajectory, split: tape.partition, head, size,
    weightsSha256: weights, selectedBudget: closest.budget, relativeTimeDifference,
    teacher: head === 'area' ? { samplesPerBatch: label.samplesPerBatch,
      batchRmse: label.frames.map(frame => frame.batchRmse) }
      : { buildId: label.source, replaySha256: label.replayHash },
    results }, files: [] };
}
