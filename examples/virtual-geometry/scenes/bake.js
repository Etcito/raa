// Milestone-1 bake scene of the virtual-geometry page (contract vg-scene-module): the four `bake`
// rows of S33. The inventory below is the only source of selection, filenames and judge
// descriptions. No imports and no top-level side effect: the page passes its helpers as `lib`
// (raa, content, cutPreviewMesh, bakeSection) and the Engine methods are the m29 ones.

const HELD = ['dark neutral background', 'one directional light with ambient light', 'white matte material', 'no ground and no other object'];
const WHITE = [1, 1, 1];
const BEHAVIOUR = {
  finest: 'One bumpy sphere fills the centre of the view. Its whole surface is covered by many small compact patches of clearly different flat colours, one per cluster. The silhouette is continuous and no hole, gap or background colour shows through the surface. The status line under the view reads ready.',
  'three-cuts': 'Three copies of the same bumpy sphere stand side by side at the same size. From left to right the colour patches become fewer and larger. All three keep the same round silhouette and none shows a crack, a hole or a missing patch.',
  foliage: 'Two square patches of many small upright blades lie side by side. The right patch has clearly fewer and larger blades than the left one and still covers the same square footprint about as densely; neither patch has an empty area.',
  sliced: 'One lit bumpy sphere is drawn complete, without holes. The status line under the view shows the bake going from none through baking with rising percentages to ready, with its cluster and level counts.',
};
const ASSERTIONS = {
  finest: ['finest-cut-has-source-triangles', 'finest-cut-has-many-clusters'],
  'three-cuts': ['cuts-have-fewer-triangles-left-to-right', 'cuts-have-fewer-clusters-left-to-right'],
  foliage: ['foliage-coarse-cut-has-fewer-triangles', 'foliage-area-kept'],
  sliced: ['bake-state-sequence', 'bake-result-has-clusters'],
};
const SLUGS = {
  finest: 'bake-finest-cut-cluster-colours',
  'three-cuts': 'bake-three-cuts-coarser-left-to-right',
  foliage: 'bake-foliage-thinned-keeps-coverage',
  sliced: 'bake-sliced-progress-then-complete',
};
const CRITERIA = {
  finest: 'Finest cut with one colour per cluster',
  'three-cuts': 'Coarser cuts left to right',
  foliage: 'Foliage thinned keeps coverage',
  sliced: 'Sliced bake progress then complete',
};

const camera = (eye, target, fovYDeg = 40) => ({ eye, target, fovYDeg, near: .1, far: 60 });
const box = (min, max) => ({ min, max });
const SPHERE = [1.06, 1.06, 1.06];
const copy = (name, x) => ({ name, ...box([x - 1.06, -1.06, -1.06], [x + 1.06, 1.06, 1.06]) });
const variant = (id, selection) => ({ id, description: BEHAVIOUR[id], references: [], judge: true, selection });
const variants = [
  variant('finest', { camera: camera([0, 0, 4], [0, 0, 0]), subjects: [{ name: 'sphere', ...box([-1.06, -1.06, -1.06], SPHERE) }] }),
  variant('three-cuts', { camera: camera([0, 0, 6.5], [0, 0, 0]), subjects: [copy('left-copy', -2.6), copy('middle-copy', 0), copy('right-copy', 2.6)] }),
  variant('foliage', {
    camera: camera([0, 3.4, 9.5], [0, .15, 0], 50),
    subjects: [{ name: 'left-patch', ...box([-4.5, 0, -2], [-.5, .25, 2]) }, { name: 'right-patch', ...box([.5, 0, -2], [4.5, .25, 2]) }],
  }),
  variant('sliced', { camera: camera([0, 0, 4], [0, 0, 0]), subjects: [{ name: 'sphere', ...box([-1.06, -1.06, -1.06], SPHERE) }] }),
];
const row = (mode, assertions) => ({
  id: SLUGS[mode], criterionName: CRITERIA[mode], scenario: 'bake', mode, filename: `${SLUGS[mode]}.png`,
  expectedVisibleBehavior: BEHAVIOUR[mode], heldConstant: HELD,
  requiredFacts: ['snapshot:meshes', 'runtime:passes', 'runtime:viewSizes', 'runtime:ready', ...assertions.map(name => `assertion:${name}`)],
  pair: null,
});

export const inventory = Object.keys(SLUGS).map(mode => row(mode, ASSERTIONS[mode]));
export const factNames = Object.values(ASSERTIONS).flat();

// The smallest lod_error of the clusters at one DAG level (S30 clusters: 48 B records, level in
// bits 16-23 of the u32 at 44, lod_error f32 at 32, little-endian).
function smallestErrorAtLevel(section, level) {
  const bytes = section.clusters;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let best = Infinity;
  for (let at = 0; at + 48 <= bytes.byteLength; at += 48) {
    if (((view.getUint32(at + 44, true) >>> 16) & 255) === level) best = Math.min(best, view.getFloat32(at + 32, true));
  }
  if (best === Infinity) throw new Error(`no cluster at DAG level ${level}`);
  return best;
}

// One Engine mesh with one sub-mesh over all its indices, one node at x, one instance.
function draw(engine, name, desc, material, x) {
  const mesh = engine.addMesh({ ...desc, subMeshes: [{ first: 0, count: desc.indices.length, material }] });
  const instance = engine.addMeshInstance(engine.addNode(null, [x, 0, 0], [0, 0, 0, 1], [1, 1, 1]), mesh);
  return { name, mesh, instance };
}

// The cut copies of each cut-mode variant: name, error threshold (metres) and x position, left to right.
const COPIES = {
  finest: () => [['finest', 0, 0]],
  'three-cuts': (section) => [
    ['finest', 0, -2.6],
    ['coarse-middle', smallestErrorAtLevel(section, Math.ceil(section.levels / 3)), 0],
    ['coarse-right', smallestErrorAtLevel(section, Math.ceil((2 * section.levels) / 3)), 2.6],
  ],
  // The thinned copy is the level a third of the way up the DAG: a quarter of the blades, each
  // widened about its base and still upright, so the right patch keeps its footprint covered.
  foliage: (section) => [['finest', 0, -2.5], ['thinned', smallestErrorAtLevel(section, Math.ceil(section.levels / 3)), 2.5]],
};

function buildSliced(engine, lib, material, state) {
  const drawn = draw(engine, 'high-poly-runtime-bake', lib.content.highPolyMesh(), material, 0);
  engine.setClusterBakeBudget(4);
  state.mesh = drawn.mesh;
  state.instance = drawn.instance;
  state.name = drawn.name;
  state.samples = [engine.meshClusters(drawn.mesh)];
  state.trace = ['none'];
  engine.bakeMeshClusters(drawn.mesh);
  return state;
}

const check = (name, passed, detail) => ({ name, passed: passed === true, detail });

// Summed area of the triangles of a cut, in square metres (object space, copies only translate).
function triangleArea({ positions, indices }) {
  const at = v => [positions[3 * v], positions[3 * v + 1], positions[3 * v + 2]];
  let total = 0;
  for (let i = 0; i < indices.length; i += 3) {
    const [a, b, c] = [indices[i], indices[i + 1], indices[i + 2]].map(at);
    const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    total += .5 * Math.hypot(u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]);
  }
  return total;
}

function observeSliced(state) {
  const samples = state.samples, first = samples[0], last = samples.at(-1);
  const baking = samples.filter(sample => sample.state === 'baking');
  const rising = baking.every((sample, i) => i === 0 || sample.progress >= baking[i - 1].progress) && baking.every(sample => sample.progress < 1);
  return {
    meshes: [{ name: state.name, mesh: state.mesh, clusters: null, cuts: [], states: samples.map(sample => ({ state: sample.state, progress: sample.progress })) }],
    instances: [{ name: state.name, instance: state.instance, virtualGeometry: null }],
    assertions: [
      check('bake-state-sequence', first.state === 'none' && baking.length >= 2 && rising && last.state === 'ready' && last.source === 'runtime',
        `states ${state.trace.join(' -> ')}; baking samples ${baking.length}; last source ${last.source}`),
      check('bake-result-has-clusters', last.clusters > 0 && last.levels > 0, `clusters ${last.clusters}, levels ${last.levels}`),
    ],
    edits: [],
  };
}

// Facts of a cut-mode variant: one mesh entry per drawn copy, each with its own uniform cut.
export async function observe(engine, selected, state) {
  if (state.mode === 'sliced') return observeSliced(state);
  const cuts = state.meshes.map(copy => ({ copy, cut: state.raa.clusterCutMesh(state.section, copy.errorMetres) }));
  const [first, second, third] = cuts.map(({ cut }) => cut);
  let assertions;
  if (state.mode === 'finest') {
    assertions = [
      check('finest-cut-has-source-triangles', first.triangles === state.section.sourceTriangles, `cut triangles ${first.triangles}, source triangles ${state.section.sourceTriangles}`),
      check('finest-cut-has-many-clusters', first.clusters >= 640, `cut clusters ${first.clusters}, at least 640`),
    ];
  } else if (state.mode === 'three-cuts') {
    assertions = [
      check('cuts-have-fewer-triangles-left-to-right', first.triangles > second.triangles && second.triangles > third.triangles, `triangles ${first.triangles}, ${second.triangles}, ${third.triangles}`),
      check('cuts-have-fewer-clusters-left-to-right', first.clusters > second.clusters && second.clusters > third.clusters, `clusters ${first.clusters}, ${second.clusters}, ${third.clusters}`),
    ];
  } else {
    const finestArea = triangleArea(first), coarseArea = triangleArea(second);
    assertions = [
      check('foliage-coarse-cut-has-fewer-triangles', second.triangles < first.triangles, `triangles ${second.triangles} coarse, ${first.triangles} finest`),
      check('foliage-area-kept', Math.abs(coarseArea - finestArea) <= .15 * finestArea, `area ${coarseArea.toFixed(4)} coarse, ${finestArea.toFixed(4)} finest m2`),
    ];
  }
  return {
    meshes: cuts.map(({ copy, cut }) => ({
      name: copy.name, mesh: copy.mesh, clusters: null,
      cuts: [{ errorMetres: copy.errorMetres, clusters: cut.clusters, triangles: cut.triangles }], states: [],
    })),
    instances: state.meshes.map(copy => ({ name: copy.name, instance: copy.instance, virtualGeometry: null })),
    assertions,
    edits: [],
  };
}

export default {
  id: 'bake', milestone: 'bake', title: 'Cluster DAG bake: cuts, foliage and sliced progress',
  engine: { msaa: 4 },
  variants,
  async build(engine, lib, selected) {
    engine.setBackground([.05, .05, .06]);
    engine.setLight({ direction: [-.35, -.6, -.55], color: [1, 1, 1], intensity: 3 });
    engine.setAmbient({ color: [1, 1, 1], intensity: .25 });
    engine.setCamera(selected.selection.camera);
    const material = engine.addSurfaceMaterial({ color: WHITE, roughness: .9 });
    const state = { mode: selected.id, raa: lib.raa, section: null, meshes: [], samples: null, trace: [], status: undefined };
    if (selected.id === 'sliced') return buildSliced(engine, lib, material, state);
    const foliage = selected.id === 'foliage';
    state.section = lib.bakeSection(foliage ? 'foliage-patch' : 'high-poly', foliage ? lib.content.foliagePatch() : lib.content.highPolyMesh());
    // Each copy is a cut preview: the module's uniform cut with one colour per cluster (S33).
    // The foliage patches are drawn in the white matte material the inventory holds constant, so
    // their preview colours are dropped; every other row shows the clusters as colour patches.
    const preview = (errorMetres) => {
      const { colors, ...plain } = lib.cutPreviewMesh(lib.raa, state.section, errorMetres);
      return foliage ? plain : { ...plain, colors };
    };
    state.meshes = COPIES[selected.id](state.section).map(([name, errorMetres, x]) => ({
      ...draw(engine, name, preview(errorMetres), material, x), errorMetres,
    }));
    return state;
  },
  // Samples the bake each frame of a sliced variant; every other variant is ready at once.
  ready(engine, state) {
    if (state.mode !== 'sliced') return { ready: true, stage: 'ready' };
    const clusters = engine.meshClusters(state.mesh);
    state.samples.push(clusters);
    const label = clusters.state === 'baking' ? `baking ${Math.round(clusters.progress * 100)}%`
      : clusters.state === 'ready' ? `ready (${clusters.clusters} clusters, ${clusters.levels} levels)` : clusters.state;
    if (state.trace.at(-1) !== label) state.trace.push(label);
    if (clusters.state === 'ready') state.status = `bake: ${state.trace.join(' -> ')}`;
    return { ready: clusters.state === 'ready', stage: clusters.state };
  },
};
