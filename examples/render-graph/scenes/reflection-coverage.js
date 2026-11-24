import { buildRoom } from '../lib/scene.js';
import { showInView } from './post-stages.js';

export const reflectionPorts = Object.freeze({
  hitMeta: 'ssrTrace.hitMeta',
  incident: 'ssrIncident.incident',
  fallbackMeta: 'ssrReplace.fallbackMeta',
  fallback: 'ssrReplace.fallback',
  previousLit: 'ssgiTrace.raw',
});

export function reflectionGraph(starter) {
  const doc = JSON.parse(starter);
  const legacy = doc.nodes.find(node => node.type === 'pass.ssr');
  const color = doc.links.find(link => link.to === `${legacy.id}.color`).from;
  doc.name = 'Reflection coverage';
  doc.nodes.splice(doc.nodes.indexOf(legacy), 1,
    { id: 'ssrHiz', type: 'ssr.hiz' },
    { id: 'ssrTrace', type: 'ssr.trace' },
    { id: 'ssrIncident', type: 'ssr.incident' },
    { id: 'ssrReplace', type: 'ssr.replace' });
  doc.links = doc.links.filter(link => link.to !== `${legacy.id}.color`).map(link =>
    link.from === `${legacy.id}.color` ? { ...link, from: 'ssrReplace.color' } : link);
  doc.links.push(
    { from: 'ssrHiz.hiz', to: 'ssrTrace.hiz' },
    { from: 'ssrTrace.trace', to: 'ssrIncident.trace' },
    { from: reflectionPorts.incident, to: 'ssrReplace.incident' },
    { from: color, to: 'ssrIncident.color' },
    { from: color, to: 'ssrReplace.color' });
  return doc;
}

// Browser previews for source class and age. Use linear readbacks for metrics.
export function reflectionPreview(starter, plane, channel) {
  const graph = showInView(reflectionGraph(starter), reflectionPorts[plane], channel, plane);
  const params = graph.nodes.find(node => node.id === 'showPostStage').params;
  if (channel === 'r' && (plane === 'hitMeta' || plane === 'fallbackMeta')) params.rangeMax = 3;
  if (channel === 'a' && plane === 'fallbackMeta') params.rangeMin = -1;
  return graph;
}

// Providers are separate layers because edge confidence blends screen and fallback.
// The volume and sky share the fallback layer. Age -1 denotes untimestamped volume data.
export const sourceExclusions = Object.freeze({
  currentHdr: 'Outside the current opaque depth buffer, occluded surfaces, and transparent/particle geometry.',
  previousLit: 'Outside current opaque depth or the previous projection. Previous visibility and shading of moving objects can disagree.',
  volume: 'Unsupported or excluded voxelizers, geometry outside the clipmap, and thin or moving features lost to voxel resolution or update cadence.',
  sky: 'All local scene geometry, including the actor.',
});

export const reflectionFrames = Object.freeze([
  { id: 'baseline', actorX: 1.5, flash: 0 },
  { id: 'flash-on', actorX: 1.5, flash: 80 },
  { id: 'flash-off', actorX: 1.5, flash: 0, phase: 'after-event' },
  { id: 'screen-edge', actorX: 3.2, flash: 0 },
  { id: 'offscreen-first', actorX: 4.5, flash: 0, phase: 'after-event' },
  { id: 'offscreen-held', actorX: 4.5, flash: 0, phase: 'after-event' },
  { id: 'reentry', actorX: 1.5, flash: 0 },
]);

const actorPosition = x => [x, 1.25, -1.4];
const rotation = [0, 0, 0, 1], scale = [1, 1, 1];

export async function buildReflectionCoverage(engine) {
  await buildRoom(engine);
  engine.setCamera({ eye: [0, 2.3, 5], target: [0, 0.9, -0.8], fovYDeg: 36, near: 0.1, far: 200 });
  const actor = engine.addNode(null, actorPosition(reflectionFrames[0].actorX), rotation, scale);
  const mesh = engine.primitiveMesh('box', { halfExtents: [0.045, 1.25, 0.075] });
  engine.addInstance(actor, mesh, engine.addMaterial({ color: [1, 0.18, 0.05], roughness: 0.55 }));
  const flash = engine.addLight({
    kind: 'point', position: [1.5, 2, -0.7], color: [1, 0.7, 0.35], range: 5, intensity: 0,
  });
  return frame => {
    engine.setTransform(actor, actorPosition(frame.actorX), rotation, scale);
    engine.updateLight(flash, { intensity: frame.flash });
  };
}

// The capture callback must read the named raw planes of the just-submitted
// frame, before stepping the scene again. Warm up on the baseline pose.
export async function runReflectionCoverage(engine, starter, { settle, renderFrame, capture }) {
  const apply = await buildReflectionCoverage(engine);
  engine.setGlobalIllumination({ enabled: true, reflections: true, screenSpace: true });
  const result = engine.setRenderGraph(reflectionGraph(starter));
  if (!result.ok) throw new Error(JSON.stringify(result.issues));
  await settle();
  for (const [index, frame] of reflectionFrames.entries()) {
    apply(frame);
    await renderFrame(index);
    const { statsFrameIndex, litHistoryAge } = engine.giStats();
    await capture({ ...frame, statsFrameIndex, litHistoryAge, ports: reflectionPorts });
  }
}

const luminance = ([r, g, b]) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

// Each sample is one receiver pixel in one real frame, in the union of the
// reference silhouette and potential ghost region. References are source-isolated
// linear RGB, in the same units and with the same weighting as rgb.
// Include zero-weight samples for reference misses to account for missing energy.
export function reflectionSourceMetrics(samples) {
  const groups = new Map();
  for (const { source, age, weight, rgb, reference, phase } of samples) {
    if (!groups.has(source)) groups.set(source, {
      source, pixels: 0, coveredPixels: 0, coverageWeight: 0, ageCounts: {},
      radianceBias: 0, falseEnergy: 0, trailingEnergy: 0,
    });
    const group = groups.get(source);
    const expected = luminance(reference);
    const observed = weight * luminance(rgb);
    const error = observed - expected;
    group.pixels++;
    if (weight > 0) {
      group.coveredPixels++;
      const ageKey = age ?? 'unknown';
      group.ageCounts[ageKey] = (group.ageCounts[ageKey] ?? 0) + 1;
    }
    group.coverageWeight += weight;
    group.radianceBias += error;
    if (expected === 0) group.falseEnergy += observed;
    if (phase === 'after-event') group.trailingEnergy += Math.max(0, error);
  }
  return [...groups.values()].map(group => ({
    ...group,
    coverage: group.coveredPixels / group.pixels,
    meanWeight: group.coverageWeight / group.pixels,
    radianceBias: group.radianceBias / group.pixels,
  }));
}

// Decode the existing split SSR graph outputs at the same receiver texel.
// The caller supplies per-source references and keeps the sky reference in split-sum units.
export function reflectionSamples({ hitMeta, incident, fallbackMeta, fallback, references, phase }) {
  return [
    ['currentHdr', hitMeta, incident, 1],
    ['volume', fallbackMeta, fallback, 2],
    ['sky', fallbackMeta, fallback, 3],
  ].map(([source, meta, radiance, sourceId]) => {
    const valid = meta[0] === sourceId && meta[1] > 0;
    return {
      source, age: valid && meta[3] >= 0 ? meta[3] : null,
      weight: valid ? radiance[3] : 0, rgb: radiance.slice(0, 3), reference: references[source], phase,
    };
  });
}

// The SSGI raw RGB is already integrated over accepted sectors (its alpha is the
// unclaimed fraction). Keep its diffuse irradiance reference separate from SSR.
export function previousLitSample(raw, litHistoryAge, reference, phase) {
  const coverage = 1 - raw[3];
  return {
    source: 'previousLit', age: coverage > 0 ? litHistoryAge : null,
    weight: coverage, rgb: coverage > 0 ? raw.slice(0, 3).map(value => value / coverage) : [0, 0, 0],
    reference, phase,
  };
}
