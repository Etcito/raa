// Functional image acceptance shared by produced metadata, GPU transport and judge.
// Numeric/API/prepass claims remain in scene notes and deterministic tests.
export const attachedStages = [
  ['initial', 'source-at-initial-parent', 'Initial attachment: the source marker is visible near the first parent block.'],
  ['translated', 'source-after-parent-translation', 'After parent translation: the source marker is visible near the moved parent block.'],
  ['rotated', 'source-after-parent-rotation', 'After parent rotation: the source marker occupies its rotated offset from the parent.'],
  ['reattached', 'source-at-second-parent', 'Reattached: the source marker is visible near the second parent block.'],
  ['detached', 'source-stays-while-parent-moves', 'Detached: the source marker is visible separated from the moved second parent block.'],
  ['reattached-before-delete', 'source-before-parent-deletion', 'Before deletion: the source marker is visible near the second parent at its new position.'],
  ['parent-deleted', 'source-survives-parent-deletion', 'After parent deletion: the second parent is absent and the source marker remains visible.'],
  ['light-removed', 'source-and-light-removed', 'After light removal: the source marker is absent and its local illumination is gone.'],
  ['attached-final', 'source-restored-at-first-parent', 'Final attachment: the source marker is visible near the first parent block.'],
];
const effects = {
  point: ['distance-attenuation', 'Bounded point illumination on a surface, diminishing with distance.'],
  spot: ['smooth-cone-transition', 'Spot footprints with a smooth inner-to-outer transition.'],
  directional: ['consistent-incident-direction', 'Directional illumination with consistent incident direction across separated receivers.'],
  area: ['small-and-wide-illuminated-regions', 'Two illuminated regions on the floor: the right region is wider than the left, showing the effect of rectangular emitter extent.'],
  'many-lights': ['multiple-light-pools', 'A field of instances with multiple distinct coloured local light pools.'],
  'attached-light': ['source-and-parent-illumination', 'A source marker near the parent block, with bounded local illumination around it. This is the final attachment state.'],
  motion: ['signed-motion-display', 'Moving boxes show signed motion colours (red, cyan, green, magenta); stationary and newly introduced geometry show zero motion.'],
};
export function captureEvidence(scene, variant, stage = null) {
  const effect = effects[scene];
  if (!effect) return null; // Other pages and later milestones own their metadata.
  let [slug, label] = effect;
  if (stage !== null) {
    const entry = attachedStages.find(([name]) => name === stage);
    if (scene !== 'attached-light' || !entry) throw new Error(`unknown capture stage: ${scene}/${stage}`);
    [, slug, label] = entry;
  }
  if (variant === 'off') {
    label = stage !== null ? `${label} Local illumination is disabled.`
      : scene === 'motion' ? 'A valid ordinarily shaded scene; the signed-motion diagnostic is disabled.'
        : `${scene}: a valid scene with the demonstrated local illumination disabled; no local light pools. Source markers and base shading may remain visible.`;
  } else if (stage !== null && stage !== 'light-removed') {
    label += ' Local illumination is enabled near the source.';
  }
  return { filename: `${scene}-${variant}-${slug}.png`, label: `${label} Local lights cast no shadow yet.` };
}
