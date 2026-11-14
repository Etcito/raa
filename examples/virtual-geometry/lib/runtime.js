// Contract 'vg-runtime'. The constants and standalone helpers are re-exported from
// web/common/page-runner.js (contract 'page-runner'); createVgPageRunner is createPageRunner with topic 'vg'.
import {
  MAX_REMAINING_MS, WARMUP_FRAMES, DEFAULT_TIME_MS, createOperationBudget, browserClock, describeError,
  variantTimeMs, renderAt, setCanvasSize, validateInput, resolveSelection, viewSizesFor, adapterFacts,
  isHardware, browserFacts, createErrorCollectors, createPageRunner,
} from '../../../web/common/page-runner.js';
import { validateRequiredFacts } from './inventory.js';
import { collectVgFacts } from './facts.js';

export {
  MAX_REMAINING_MS, WARMUP_FRAMES, DEFAULT_TIME_MS, createOperationBudget, browserClock, describeError,
  variantTimeMs, renderAt, setCanvasSize, validateInput, resolveSelection, viewSizesFor, adapterFacts,
  isHardware, browserFacts, createErrorCollectors,
};

// The VG facts of an observation, in the order settings, stats, meshes, instances, assertions, edits.
function collectFacts(engine, observation) {
  const c = collectVgFacts(engine, observation.meshes, observation.instances);
  return {
    settings: c.settings,
    stats: c.stats,
    meshes: c.meshes,
    instances: c.instances,
    assertions: observation.assertions,
    edits: observation.edits,
  };
}

// The PAIR_RUN extras of a measured variant: the on state reads the cluster counts of virtualGeometryStats(),
// the off state reads 0 clusters and the triangles of stats().
function measurementExtras(engine, variant) {
  const vg = variant.measurement.state === 'on' && typeof engine.virtualGeometryStats === 'function'
    ? engine.virtualGeometryStats()
    : null;
  return {
    drawnClusters: vg ? vg.drawnClusters : 0,
    drawnTriangles: vg ? vg.drawnTriangles : engine.stats().triangles,
    hardwareClusters: vg ? vg.hardwareClusters : 0,
    softwareClusters: vg ? vg.softwareClusters : 0,
  };
}

// Share 2 of contract 'vg-runtime'. Form: examples/sky-weather/lib/runtime.js lines 159-460.
// Handoff: createVgPageRunner(options) is rewritten by card vg-page: createVgPageRunner on createPageRunner,
// which builds it on createPageRunner (web/common/page-runner.js:256 createPageRunner). Today it is this
// standalone runner, calling the page-kit helpers re-exported above.
export function createVgPageRunner(options) {
  return createPageRunner({ ...options, topic: 'vg', collectFacts, validateRequiredFacts, measurementExtras });
}
