// The cluster-bake module functions run without an Engine or WebGPU, so the built package is
// loaded in Node with the wasm bytes read from packages/raa/pkg/ (scripts/build-package.sh).
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import test from 'node:test';

const here = dirname(fileURLToPath(import.meta.url));
const wasmPath = join(here, '..', 'pkg', 'raa_js_bg.wasm');
const sizesPath = join(here, '..', '..', '..', 'dist', 'raa-sizes.json');

// Q2 is growth over the milestone-1 starting revision (aa2c1bf), not an absolute figure:
// that revision already ships a 978602 B gzip wasm, below which no change can measure.
// The cap is the recorded baseline plus WASM_GZIP_MAX (300000 B) of headroom.
const WASM_GZIP_BASELINE = 978602;
const WASM_GZIP_MAX = 300000;

const SECTION_KEYS = [
  'version',
  'algorithm',
  'vertices',
  'triangles',
  'clusters',
  'nodes',
  'rootNode',
  'hierarchyDepth',
  'levels',
  'quantOrigin',
  'quantStep',
  'uvOrigin',
  'uvStep',
  'boundsCenter',
  'boundsRadius',
  'sourceTriangles',
];

// An icosphere of 1280 triangles (20 * 4^3), unit radius, shared vertices.
function icosphere(subdivisions) {
  const t = (1 + Math.sqrt(5)) / 2;
  const vertices = [
    [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
    [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
    [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
  ].map((p) => {
    const length = Math.hypot(p[0], p[1], p[2]);
    return [p[0] / length, p[1] / length, p[2] / length];
  });
  let faces = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
  ];
  const midpoints = new Map();
  const midpoint = (a, b) => {
    const key = a < b ? `${a}:${b}` : `${b}:${a}`;
    const hit = midpoints.get(key);
    if (hit !== undefined) return hit;
    const pa = vertices[a];
    const pb = vertices[b];
    const p = [pa[0] + pb[0], pa[1] + pb[1], pa[2] + pb[2]];
    const length = Math.hypot(p[0], p[1], p[2]);
    vertices.push([p[0] / length, p[1] / length, p[2] / length]);
    const index = vertices.length - 1;
    midpoints.set(key, index);
    return index;
  };
  for (let level = 0; level < subdivisions; level++) {
    const next = [];
    for (const [a, b, c] of faces) {
      const ab = midpoint(a, b);
      const bc = midpoint(b, c);
      const ca = midpoint(c, a);
      next.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]);
    }
    faces = next;
  }
  const positions = new Float32Array(vertices.length * 3);
  const normals = new Float32Array(vertices.length * 3);
  vertices.forEach((p, i) => {
    positions.set(p, i * 3);
    normals.set(p, i * 3);
  });
  const indices = new Uint32Array(faces.length * 3);
  faces.forEach((face, i) => indices.set(face, i * 3));
  return { positions, normals, indices };
}

const MESH = icosphere(3);

let loaded;
function raa() {
  if (!existsSync(wasmPath)) {
    throw new Error(`missing ${wasmPath}: build the package first with bash scripts/build-package.sh`);
  }
  loaded ??= import(pathToFileURL(join(here, '..', 'index.js'))).then((module) =>
    module.loadRaa({ wasm: readFileSync(wasmPath) }),
  );
  return loaded;
}

test('the packaged wasm stays within the baseline-relative gzip cap (Q2)', () => {
  if (!existsSync(sizesPath)) {
    throw new Error(`missing ${sizesPath}: build the package first with bash scripts/build-package.sh`);
  }
  const sizes = JSON.parse(readFileSync(sizesPath, 'utf8'));
  const cap = WASM_GZIP_BASELINE + WASM_GZIP_MAX;
  assert.ok(
    sizes.gzip <= cap,
    `gzip ${sizes.gzip} B exceeds the ${WASM_GZIP_BASELINE} B baseline + ${WASM_GZIP_MAX} B headroom (cap ${cap} B; raw ${sizes.raw} B)`,
  );
});

test('the one-call bake returns the sixteen-key section and it validates', async () => {
  const module = await raa();
  const section = module.bakeClusters(MESH);
  assert.deepEqual(Object.keys(section), SECTION_KEYS);
  assert.equal(section.version, 1);
  assert.equal(section.algorithm, 1);
  assert.ok(section.vertices instanceof Uint16Array);
  assert.ok(section.triangles instanceof Uint32Array);
  assert.ok(section.clusters instanceof Uint8Array);
  assert.ok(section.nodes instanceof Uint8Array);
  assert.equal(section.vertices.length % 8, 0);
  assert.equal(section.clusters.length % 48, 0);
  assert.equal(section.nodes.length % 48, 0);
  assert.ok(section.clusters.length > 0);
  assert.ok(section.nodes.length > 0);
  assert.equal(typeof section.rootNode, 'number');
  assert.equal(typeof section.hierarchyDepth, 'number');
  assert.equal(typeof section.levels, 'number');
  assert.ok(Array.isArray(section.quantOrigin) && section.quantOrigin.length === 3);
  assert.equal(typeof section.quantStep, 'number');
  assert.ok(Array.isArray(section.uvOrigin) && section.uvOrigin.length === 2);
  assert.ok(Array.isArray(section.uvStep) && section.uvStep.length === 2);
  assert.ok(Array.isArray(section.boundsCenter) && section.boundsCenter.length === 3);
  assert.equal(typeof section.boundsRadius, 'number');
  assert.equal(section.sourceTriangles, 1280);
  module.validateClusters(section);
});

test('the uniform cut at 0 metres is the source mesh and a huge error is coarser', async () => {
  const module = await raa();
  const section = module.bakeClusters(MESH);
  const finest = module.clusterCutMesh(section, 0);
  assert.equal(finest.triangles, 1280);
  assert.equal(finest.indices.length, finest.triangles * 3);
  assert.equal(finest.clusterOfTriangle.length, finest.triangles);
  const coarse = module.clusterCutMesh(section, 1e29);
  assert.ok(coarse.triangles < 1280);
  assert.equal(coarse.clusterOfTriangle.length, coarse.triangles);
});

test('the sliced bake stepped at 1 ms matches the one-call bake', async () => {
  const module = await raa();
  const section = module.bakeClusters(MESH);
  const bake = module.createClusterBake(MESH);
  try {
    let fraction = -1;
    let progress;
    do {
      progress = bake.step(1);
      assert.ok(progress.fraction >= fraction);
      fraction = progress.fraction;
    } while (!progress.done);
    assert.equal(progress.fraction, 1);
    assert.equal(progress.stage, 'done');
    assert.deepEqual(bake.result(), section);
  } finally {
    bake.free();
  }
});

test('the result before done and a step after free are vg.invalid', async () => {
  const module = await raa();
  const bake = module.createClusterBake(MESH);
  assert.throws(
    () => bake.result(),
    (error) => error.code === 'vg.invalid',
  );
  bake.free();
  bake.free();
  assert.throws(
    () => bake.step(1),
    (error) => error.code === 'vg.invalid' && /freed/.test(error.message),
  );
});

test('a normals length mismatch is refused naming normals', async () => {
  const module = await raa();
  const bad = { ...MESH, normals: MESH.normals.slice(0, MESH.normals.length - 1) };
  assert.throws(
    () => module.bakeClusters(bad),
    // The JS message carries the engine's `<code>: ` prefix before the field path.
    (error) => error.code === 'vg.invalid' && error.message.includes('normals:'),
  );
});

test('an out-of-range option is refused naming options.targetRatio', async () => {
  const module = await raa();
  assert.throws(
    () => module.bakeClusters({ ...MESH, options: { targetRatio: 2 } }),
    (error) => error.code === 'vg.invalid' && error.message.includes('options.targetRatio'),
  );
});

test('a corrupted clusters byte fails the validation with vg.invalid', async () => {
  const module = await raa();
  const section = module.bakeClusters(MESH);
  const bad = { ...section, clusters: section.clusters.slice() };
  // Offset 44 of a 48 B cluster record is the low byte of its counts word (vertex count, 1..255).
  bad.clusters[44] = 0;
  assert.throws(
    () => module.validateClusters(bad),
    (error) => error.code === 'vg.invalid',
  );
});

test('an empty index array is vg.mesh_empty', async () => {
  const module = await raa();
  assert.throws(
    () =>
      module.bakeClusters({
        positions: new Float32Array(0),
        normals: new Float32Array(0),
        indices: new Uint32Array(0),
      }),
    (error) => error.code === 'vg.mesh_empty',
  );
});
