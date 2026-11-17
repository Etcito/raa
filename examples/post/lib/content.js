// Generated content of the post page: geometry and RGBA textures. Pure ES module (no DOM,
// no engine, no imports, importable by node). Every mesh builder returns a fresh MeshDesc
// (contract 'post-content-api'): Float32Array positions (3n), unit outward normals (3n),
// Uint32Array indices (3m, counter-clockwise seen from outside), plus `subMeshes` when the
// mesh is split. Metres, +Y up. Determinism: no Math.random and no clock.

// Appends one quad (corners counter-clockwise seen from outside) to a plain accumulator.
function pushQuad(out, corners, normal) {
  const base = out.positions.length / 3;
  for (const c of corners) out.positions.push(c[0], c[1], c[2]);
  for (let i = 0; i < 4; i++) out.normals.push(normal[0], normal[1], normal[2]);
  out.indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
}

// Appends a closed axis-aligned box with hard edges: 24 vertices, 12 triangles.
function pushBox(out, center, half) {
  for (let a = 0; a < 3; a++) {
    for (const s of [1, -1]) {
      const u = s > 0 ? (a + 1) % 3 : (a + 2) % 3;
      const v = s > 0 ? (a + 2) % 3 : (a + 1) % 3;
      const normal = [0, 0, 0];
      normal[a] = s;
      const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([su, sv]) => {
        const p = [0, 0, 0];
        p[a] = center[a] + s * half[a];
        p[u] = center[u] + su * half[u];
        p[v] = center[v] + sv * half[v];
        return p;
      });
      pushQuad(out, corners, normal);
    }
  }
}

// Appends the box spanning [min, max]; an empty or inverted extent adds nothing.
function pushSpan(out, min, max) {
  if (!(max[0] > min[0] && max[1] > min[1] && max[2] > min[2])) return;
  const center = min.map((m, i) => (m + max[i]) / 2);
  const half = min.map((m, i) => (max[i] - m) / 2);
  pushBox(out, center, half);
}

// Appends an XZ quad at y 0 with normal +Y.
function pushFlat(out, x0, z0, x1, z1) {
  pushQuad(out, [[x0, 0, z0], [x0, 0, z1], [x1, 0, z1], [x1, 0, z0]], [0, 1, 0]);
}

// Packs plain accumulators (one per sub-mesh, in order) into one MeshDesc.
function finish(parts) {
  const positions = [];
  const normals = [];
  const indices = [];
  const subMeshes = [];
  for (const part of parts) {
    const base = positions.length / 3;
    positions.push(...part.positions);
    normals.push(...part.normals);
    const first = indices.length;
    for (const i of part.indices) indices.push(base + i);
    subMeshes.push({ first, count: indices.length - first });
  }
  return {
    positions: Float32Array.from(positions),
    normals: Float32Array.from(normals),
    indices: Uint32Array.from(indices),
    subMeshes,
  };
}

const accumulator = () => ({ positions: [], normals: [], indices: [] });

/** Closed axis-aligned box: 24 vertices, 36 indices. `center` and `half` are [x, y, z]. */
export function box(center, half) {
  const out = accumulator();
  pushBox(out, center, half);
  const mesh = finish([out]);
  delete mesh.subMeshes;
  return mesh;
}

/** XZ plane at y 0 centred at the origin, normal +Y: 4 vertices, 6 indices. */
export function plane(width, depth) {
  const out = accumulator();
  pushFlat(out, -width / 2, -depth / 2, width / 2, depth / 2);
  const mesh = finish([out]);
  delete mesh.subMeshes;
  return mesh;
}

/**
 * Handoff: card 'post-page: exposure-room scene' calls these builders, all exported by
 * this file: roomWithOpening (below) returns a MeshDesc with subMeshes [floor, ceiling,
 * walls]; tiledGround({ x0, z0, columns, rows, size }) returns a MeshDesc with subMeshes
 * [even, odd] of flat +Y quads; facadeCourses({ x, zMin, zMax, yMin, yMax, course, depth,
 * skip }) returns a MeshDesc with subMeshes [even, odd] of horizontal boxes that leave the
 * skip rect free; box(center, half) returns a 24-vertex MeshDesc; plane(width, depth)
 * returns a 4-vertex MeshDesc. The scene passes each MeshDesc to engine.addMesh with one
 * material per subMesh. stripes and checker are read by later milestones' scenes only.
 *
 * Room: interior x in [-width/2, width/2], y in [0, height], z in [-depth, 0]; floor and
 * ceiling are slabs of thickness `wall`; the east wall is split into four boxes around
 * the opening.
 */
export function roomWithOpening({ width, height, depth, wall, opening }) {
  const hw = width / 2;
  const xo = hw + wall;
  const floor = accumulator();
  const ceiling = accumulator();
  const walls = accumulator();
  pushSpan(floor, [-xo, -wall, -depth - wall], [xo, 0, wall]);
  pushSpan(ceiling, [-xo, height, -depth - wall], [xo, height + wall, wall]);
  pushSpan(walls, [-xo, 0, -depth - wall], [-hw, height, wall]);
  pushSpan(walls, [-hw, 0, -depth - wall], [hw, height, -depth]);
  pushSpan(walls, [-hw, 0, 0], [hw, height, wall]);
  const { zMin, zMax, yMin, yMax } = opening;
  pushSpan(walls, [hw, 0, -depth], [xo, yMin, 0]);
  pushSpan(walls, [hw, yMax, -depth], [xo, height, 0]);
  pushSpan(walls, [hw, yMin, -depth], [xo, yMax, zMin]);
  pushSpan(walls, [hw, yMin, zMax], [xo, yMax, 0]);
  return finish([floor, ceiling, walls]);
}

/** Flat ground of columns x rows tiles; sub-meshes [even, odd] by (i + j) % 2. */
export function tiledGround({ x0, z0, columns, rows, size }) {
  const parts = [accumulator(), accumulator()];
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < columns; i++) {
      const x = x0 + i * size;
      const z = z0 + j * size;
      pushFlat(parts[(i + j) % 2], x, z, x + size, z + size);
    }
  }
  return finish(parts);
}

/**
 * Horizontal courses (boxes from x to x + depth, `course` high) stacked from yMin to yMax;
 * sub-meshes [even, odd] by course index. A course overlapping `skip` in y is split into
 * the parts outside skip.zMin..skip.zMax.
 */
export function facadeCourses({ x, zMin, zMax, yMin, yMax, course, depth, skip }) {
  const parts = [accumulator(), accumulator()];
  for (let k = 0; yMin + k * course < yMax; k++) {
    const y0 = yMin + k * course;
    const y1 = Math.min(y0 + course, yMax);
    const out = parts[k % 2];
    if (y1 > skip.yMin && y0 < skip.yMax) {
      pushSpan(out, [x, y0, zMin], [x + depth, y1, Math.min(skip.zMin, zMax)]);
      pushSpan(out, [x, y0, Math.max(skip.zMax, zMin)], [x + depth, y1, zMax]);
    } else {
      pushSpan(out, [x, y0, zMin], [x + depth, y1, zMax]);
    }
  }
  return finish(parts);
}

function texture(name, size, period, on) {
  if (!Number.isInteger(size) || size <= 0 || !Number.isInteger(period) || period < 2) {
    throw new Error(`content: ${name} needs an integer size > 0 and an integer period >= 2`);
  }
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const v = on(Math.floor((2 * x) / period), Math.floor((2 * y) / period)) ? 255 : 0;
      data.set([v, v, v, 255], (y * size + x) * 4);
    }
  }
  return { width: size, height: size, data };
}

/** size x size RGBA8 vertical stripes, half-period `period / 2` texels wide. */
export function stripes(size, period) {
  return texture('stripes', size, period, (a) => a % 2 === 0);
}

/** size x size RGBA8 checker, cells `period / 2` texels wide. */
export function checker(size, period) {
  return texture('checker', size, period, (a, b) => (a + b) % 2 === 0);
}
