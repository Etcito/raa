import { interactiveInterior } from './interactive-interior.js';
import { loadGalleryNeural } from './gallery-neural.js';
import { jointTrajectory } from './gallery-joint-trajectories.js';

const display = [1280, 720], source = [640, 360], cadenceMs = 1000 / 60;
const warmFrames = 36, measuredFrames = 60;
const runOrder = ['analytic', 'learned', 'learned', 'analytic', 'learned', 'analytic', 'analytic', 'learned'];
const poseTape = jointTrajectory('joint-development').replay;
const unitExposureEv = 0;

function percentiles(values) {
  const sorted = values.filter(Number.isFinite).toSorted((a, b) => a - b);
  const at = p => sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * p) - 1)] ?? null;
  return { n: sorted.length, p50: at(.50), p95: at(.95), p99: at(.99) };
}

function medianPassCost(sample) {
  return sample.passes.reduce((sum, pass) => sum + (pass.ms ?? 0), 0);
}

function passSummary(samples) {
  const names = new Set(samples.flatMap(row => row.passes.map(pass => pass.pass)));
  return Object.fromEntries([...names].map(name => [name,
    percentiles(samples.map(row => row.passes.find(pass => pass.pass === name)?.ms))]));
}

function summarize(gpu, real, longAnimationFrames) {
  return {
    rendererGpu: { summedTimedPassMs: percentiles(gpu.map(medianPassCost)), passes: passSummary(gpu) },
    chrome: {
      rafGapMs: percentiles(real.slice(1).map((row, i) => row.rafMs - real[i].rafMs)),
      missedCadenceSlots: real.reduce((sum, row) => sum + row.missed, 0),
      submitToCanvasHandoffMs: percentiles(real.map(row => row.presentReturnedMs - row.submitEndMs)),
      rafToCanvasHandoffMs: percentiles(real.map(row => row.presentReturnedMs - row.rafMs)),
      longAnimationFrames: longAnimationFrames.length,
    },
  };
}

function waitForFrame(target) {
  return new Promise(resolve => target.requestAnimationFrame(resolve));
}

export async function measureGallerySrTiming({ target, canvas, raa }, { buildId, modelHeads = 'sr' } = {}) {
  if (!buildId || !/Windows.+Chrome\//.test(target.navigator.userAgent) || !['sr', 'sr-reflection', 'joint'].includes(modelHeads)) {
    throw new Error('SR timing requires a pinned Windows Chrome build and a recognized model configuration');
  }
  [canvas.width, canvas.height] = display;
  const engine = await raa.createEngine(canvas, { msaa: 1, allowSoftware: false, powerPreference: 'high-performance' });
  const longFrames = [];
  let observer;
  try {
    const { hardware, adapter } = engine.stats();
    if (!hardware || adapter.isFallbackAdapter !== false) throw new Error('SR timing requires hardware WebGPU');
    const scene = interactiveInterior.default;
    const preset = scene.variants.find(item => item.id === 'performance');
    const effects = { ...preset.effects, 'moving-objects': false, ssr: false, bloom: false, dof: false };
    const state = scene.build(engine, { raa }, { ...preset, effects });
    engine.setAutoExposure({ enabled: false });
    engine.setToneMapping({ exposureEv: unitExposureEv });
    const model = await loadGalleryNeural(engine);
    if (!model) throw new Error('No fitted SR model in the Windows timing stage');
    const heads = ['display'];
    if (modelHeads !== 'sr') heads.push('reflection');
    if (modelHeads === 'joint') heads.push('area');
    if (!heads.every(head => model.manifest.trainedHeads.includes(head))) {
      throw new Error('SR timing model lacks the requested fitted heads');
    }
    const rotation = [0, 0, 0, 1], scale = [1, 1, 1];
    function apply(tick) {
      // Replay a smooth closed phase curve with the same logical ticks in every block.
      const phase = tick % (2 * (poseTape.length - 1));
      const pose = poseTape[Math.min(phase, 2 * (poseTape.length - 1) - phase)];
      engine.setCamera({ ...preset.selection.camera, ...pose.camera });
      engine.setTransform(state.sphereNode, pose.sphere, rotation, scale);
      engine.setTransform(state.entryNode, pose.reflector, rotation, scale);
    }
    const records = [];
    if (target.PerformanceObserver?.supportedEntryTypes.includes('long-animation-frame')) {
      observer = new target.PerformanceObserver(list => {
        for (const entry of list.getEntries()) longFrames.push({
          startMs: entry.startTime, durationMs: entry.duration, renderStartMs: entry.renderStart,
          presentationTimeMs: Number.isFinite(entry.presentationTime) ? entry.presentationTime : null,
        });
      });
      observer.observe({ type: 'long-animation-frame' });
    }
    const frozenGraph = JSON.stringify(model.graph);
    for (const [block, mode] of runOrder.entries()) {
      const switchGraph = engine.setRenderGraphSelection(model.graph, undefined, ...source);
      if (!switchGraph.ok) throw new Error('SR timing graph refused: ' + JSON.stringify(switchGraph.issues));
      const selection = { model: model.id, sr: true, area: modelHeads === 'joint', reflection: modelHeads !== 'sr' };
      engine.setNeuralReconstruction(mode === 'learned' ? selection : null);
      engine.setCameraJitter(true);
      await engine.submittedWorkDone();
      let nextRaf = await waitForFrame(target);
      const samples = new Map(), real = [], windows = new Set();
      let begin, end;
      for (let tick = 0; tick < warmFrames + measuredFrames; tick++) {
        let raf;
        do { raf = await waitForFrame(target); } while (raf + .5 < nextRaf);
        const missed = Math.max(0, Math.floor((raf - nextRaf + .5) / cadenceMs));
        nextRaf += (missed + 1) * cadenceMs;
        const timed = tick >= warmFrames;
        if (tick === warmFrames) begin = raf;
        apply(tick);
        const timing = engine.renderInstrumented(2000 + tick * cadenceMs, {
          viewId: 'primary', trialId: 'sr-paired-' + block, presentId: 'sr-' + block + '-' + tick,
          inputCutoffMs: target.performance.now(),
        });
        if (timing.error || !Number.isFinite(timing.presentReturnedMs)) throw new Error('SR timing frame failed: ' + JSON.stringify(timing));
        if (timed) {
          windows.add(timing.frameIndex);
          real.push({ ...timing, tick, rafMs: raf, missed });
        }
        // Query once per opportunity. The returned timestamp sample is asynchronous.
        const gpu = engine.passGpuTimings();
        if (gpu.status === 'unavailable') throw new Error('SR GPU timestamps unavailable: ' + gpu.reason);
        if (gpu.status === 'sample' && windows.has(gpu.frame)) samples.set(gpu.frame, gpu);
      }
      await engine.submittedWorkDone();
      for (let i = 0; i < 5; i++) {
        await waitForFrame(target);
        const gpu = engine.passGpuTimings();
        if (gpu.status === 'sample' && windows.has(gpu.frame)) samples.set(gpu.frame, gpu);
      }
      end = real.at(-1).presentReturnedMs;
      const gpu = [...samples.values()];
      if (gpu.length < 20) throw new Error('Insufficient adjacent timestamp samples in SR block ' + block);
      if (gpu.some(row => !row.passes.some(pass => /super-resolution|resolve|taa/.test(pass.pass) && pass.ms > 0))) {
        throw new Error('SR resolve is missing from GPU timestamp evidence');
      }
      const graph = engine.renderGraphStats();
      const extent = (id, port) => {
        const r = graph.nodes.find(node => node.id === id)?.resources.find(resource => resource.port === port);
        return r && [r.width, r.height];
      };
      if (String(extent('prepass', 'depth')) !== String(source) || String(extent('resolve', 'color')) !== String(display)) {
        throw new Error('SR timing source/display extents differ from the paired contract');
      }
      const dispatch = engine.neuralReconstructionStats();
      if (mode === 'learned'
        ? !dispatch?.ran.sr || dispatch.ran.area !== selection.area || dispatch.ran.reflection !== selection.reflection
          || dispatch.weightsSha256 !== model.manifest.weightsSha256
        : dispatch !== null) {
        throw new Error('SR model dispatch does not match the timing arm');
      }
      const loa = longFrames.filter(row => row.startMs >= begin && row.startMs <= end);
      records.push({ block, mode, begin, end, actualExtents: { source, display }, dispatch,
        gpu, real, longAnimationFrames: loa, summary: summarize(gpu, real, loa) });
    }
    const pairs = [];
    for (let i = 0; i < records.length; i += 2) {
      const [learned, analytic] = records.slice(i, i + 2).sort((a, b) => b.mode.localeCompare(a.mode));
      const delta = (plane, metric) => Object.fromEntries(['p50','p95','p99'].map(p => [p,
        learned.summary[plane][metric][p] - analytic.summary[plane][metric][p]]));
      pairs.push({ analyticBlock: analytic.block, learnedBlock: learned.block,
        summedTimedPassMs: delta('rendererGpu', 'summedTimedPassMs'), rafGapMs: delta('chrome', 'rafGapMs'),
        rafToCanvasHandoffMs: delta('chrome', 'rafToCanvasHandoffMs') });
    }
    return { json: { source: { buildId, facts: { hardware, adapter, browser: target.navigator.userAgent,
      devicePixelRatio: target.devicePixelRatio } }, size: { source, display }, cadenceMs, exposureEv: unitExposureEv,
      warmFrames, measuredFrames, effects, modelHeads, dispatchGraph: JSON.parse(frozenGraph),
      weightsSha256: model.manifest.weightsSha256, clock: {
        rendererGpu: 'WebGPU timestamp-query intervals around RAA feature passes, summed over the timed passes',
        chrome: 'RAF callbacks, canvas handoff and document long animation frame signals',
      }, order: runOrder, pairs, records }, files: [] };
  } finally {
    observer?.disconnect();
    engine.dispose();
  }
}
