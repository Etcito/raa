// Facts of the global-illumination page (contract 'gi-facts'). Browser-free: it reads only the engine's public
// getters and the scene observation, so node can import it. The page runner appends passes, viewSizes and readiness.

// Calls one getter. A missing method or a throw is recorded as null, so one unavailable value never hides the others.
function safe(fn) {
  try {
    return fn() ?? null;
  } catch {
    return null;
  }
}

/**
 * `collectGiFacts` returns the GI facts of one observed frame, in the key order of contract 'gi-facts':
 * settings (`globalIllumination()`), quality (`giQuality()`), stats (`giStats()`), instances (one per
 * `observation.instances` entry through `instanceGlobalIllumination(instance)`), assertions and edits
 * (the observation's own lists, `[]` when absent). Unavailable values are null. The variant is not read.
 */
export function collectGiFacts(engine, observation, variant) {
  const instances = Array.isArray(observation?.instances) ? observation.instances : [];
  return {
    settings: safe(() => engine.globalIllumination()),
    quality: safe(() => engine.giQuality()),
    stats: safe(() => engine.giStats()),
    instances: instances.map(({ name, instance }) => ({
      name,
      instance,
      globalIllumination: safe(() => engine.instanceGlobalIllumination(instance)),
    })),
    assertions: observation?.assertions ?? [],
    edits: observation?.edits ?? [],
  };
}
