// Page runtime of the global-illumination page (S40, module gi-page). createGiPageRunner wraps createPageRunner
// of page-kit with the GI facts, the GI inventory check, the GI measurement extras and the GI readiness gate.
import { createPageRunner, describeError } from '../../../web/common/page-runner.js';
// Consumed upstream, not re-implemented here: createPageRunner (web/common/page-runner.js:265, its Handoff doc
// comment is at :256) and describeError (web/common/page-runner.js:113), and validateRequiredFacts
// (examples/global-illumination/lib/inventory.js:137, its Handoff doc comment is at :132).
import { collectGiFacts } from './facts.js';
import { validateRequiredFacts } from './inventory.js';

const GI_SETTLE_FRAMES = 240;

// A failed gate verdict; every message carries the 'gi readiness: ' prefix.
const failed = (message, stats) => ({ status: 'failed', stats, error: `gi readiness: ${message}` });

// The stats fields that the deadline message names, or null before any giStats() result.
const summary = stats => (stats === null ? null : {
  active: stats.active,
  inactiveReason: stats.inactiveReason,
  pendingRebuild: stats.pendingRebuild,
  framesSinceRebuild: stats.framesSinceRebuild,
});

/**
 * Handoff: gi-page: contract.test.mjs takes `waitGiSettled(engine, render, budget, frames = 240) -> { status, stats, error }`
 * exported by examples/global-illumination/lib/runtime.js:33 waitGiSettled; the card also calls `createGiPageRunner(options)`,
 * exported by examples/global-illumination/lib/runtime.js:80 createGiPageRunner. State/keys: the verdict
 * { status: 'ready' | 'failed', stats: object | null, error: string | null }, where stats is the last giStats() result or null
 * when none was read.
 * Today `waitGiSettled` renders and reads giStats() until the stats are active, not pending a rebuild and have
 * framesSinceRebuild >= frames: ready. A non-null inactiveReason, a thrown error or the budget's deadline fails it.
 * It never reports ready otherwise.
 */
export async function waitGiSettled(engine, render, budget, frames = GI_SETTLE_FRAMES) {
  let last = null;
  for (;;) {
    if (budget.remainingMs() <= 0) {
      return failed(`not settled before the operation deadline (last stats: ${JSON.stringify(summary(last))})`, last);
    }
    try {
      await render();
      last = engine.giStats() ?? null;
    } catch (error) {
      return failed(describeError(error), last);
    }
    if (last?.inactiveReason != null) return failed(`inactive (${last.inactiveReason})`, last);
    if (last?.active && !last.pendingRebuild && last.framesSinceRebuild >= frames) {
      return { status: 'ready', stats: last, error: null };
    }
  }
}

// The settle phase of createPageRunner. A 'disabled' variant renders once and must read GI off with no allocation;
// any other variant (the default 'active') waits for GI to settle over GI_SETTLE_FRAMES frames.
async function settle(engine, { render, budget, variant }) {
  if (variant?.giExpected !== 'disabled') return waitGiSettled(engine, render, budget, GI_SETTLE_FRAMES);
  try {
    await render();
    const stats = engine.giStats() ?? null;
    const seen = {
      active: stats?.active ?? null,
      inactiveReason: stats?.inactiveReason ?? null,
      allocatedBytes: stats?.allocatedBytes ?? null,
    };
    if (seen.active === false && seen.inactiveReason === 'disabled' && seen.allocatedBytes === 0) {
      return { status: 'ready', stats, error: null };
    }
    return failed(`expected GI disabled, got ${JSON.stringify(seen)}`, stats);
  } catch (error) {
    return failed(describeError(error), null);
  }
}

/**
 * Handoff: gi-page: page entry, build script and p4-gi commands takes `createGiPageRunner(options: object) -> { run, engine, dispose }`
 * exported by examples/global-illumination/lib/runtime.js:80 createGiPageRunner; attached by the page entry importing it and
 * calling it with its canvas, status, load, scenes and lib options. State/keys: those options, plus topic 'gi', collectGiFacts
 * (lib/facts.js), validateRequiredFacts (lib/inventory.js), the GI measurement extras and the settle gate.
 * Today it returns createPageRunner's runner unchanged: every refusal of that runner carries the 'gi runtime: ' prefix.
 */
export function createGiPageRunner(options) {
  return createPageRunner({
    ...options,
    topic: 'gi',
    collectFacts: collectGiFacts,
    validateRequiredFacts,
    measurementExtras: engine => ({ quality: engine.giQuality(), allocatedBytes: engine.giStats().allocatedBytes }),
    settle,
  });
}
