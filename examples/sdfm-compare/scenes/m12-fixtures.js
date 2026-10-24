// Compare scene m12-fixtures (M7, R12): the three fixture assets shown by the MAMA reference viewport
// (left) and by RAA through @etcito/raa-sdfm (right), with the same camera. Not judged: the
// owner reviews each saved capture against R12.
// Camera positions frame each fixture's bounds (target at the bounds centre, eye 1.1 bounding
// spheres away along the page's front-right direction, fovY 45 deg).
// Pure module (S14): no top-level side effects, node reads id, milestone and variants.

export default {
  id: 'm12-fixtures',
  milestone: 'm12',
  title: 'M7 fixtures: one sdfm.game-asset in MAMA and in RAA, side by side',
  variants: [
    {
      id: 'compare-crate',
      description:
        'Rounded crate: the same shape, parts, materials and colours on both sides (R12): wood-like body, metal bands, painted panel.',
      fixture: 'fixtures/crate.asset.json',
      references: [],
      judge: false,
      settings: {
        camera: { position: [1.322, 1.341, 2.403], target: [0, 0.5, 0], fovYDeg: 45 },
      },
    },
    {
      id: 'compare-blend',
      description:
        'Smooth-blended shapes (sphere, box and capsule smooth-unioned, 2 materials): the same blended silhouette, parts, materials and colours on both sides (R12).',
      fixture: 'fixtures/blend.asset.json',
      references: [],
      judge: false,
      settings: {
        camera: { position: [2.043, 2.138, 3.706], target: [0.005, 0.841, 0], fovYDeg: 45 },
      },
    },
    {
      id: 'compare-figure',
      description:
        'Small figure with rigid parts (body, head and two arms as rigid sub-groups with pivots, 3+ materials): the same shape, parts, materials and colours on both sides (R12).',
      fixture: 'fixtures/figure.asset.json',
      references: [],
      judge: false,
      settings: {
        camera: { position: [1.571, 1.694, 2.856], target: [0, 0.695, 0], fovYDeg: 45 },
      },
    },
  ],
};
