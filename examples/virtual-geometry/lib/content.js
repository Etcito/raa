// Preset meshes of the virtual-geometry page. Pure ES module (no DOM, no engine, importable
// by node). Every builder returns a fresh MeshDesc: Float32Array positions (3n), normals
// (3n, unit), uvs (2n) and Uint32Array indices (3m). Metres, +Y up, counter-clockwise front
// faces, outward normals (the form of examples/lookdev/lib/geometry.js).
//
// Determinism: no Math.random and no clock. Foliage draws from a 32-bit LCG with a fixed seed.

const TAU = Math.PI * 2;
const FOLIAGE_SEED = 0x2f6e2b1;
const FOLIAGE_GRID = 64;
const FOLIAGE_EXTENT = 4;
const BLADE_WIDTH = 0.03;
const BLADE_HEIGHT = 0.25;

const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const normalize = (a) => scale(a, 1 / Math.hypot(a[0], a[1], a[2]));

// Cross product of two 3-vectors.
function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

// Packs plain arrays into the MeshDesc typed arrays.
function pack(positions, normals, uvs, indices) {
  return {
    positions: Float32Array.from(positions),
    normals: Float32Array.from(normals),
    uvs: Float32Array.from(uvs),
    indices: Uint32Array.from(indices),
  };
}

// Icosphere of the given subdivision: the 12 icosahedron vertices and 20 faces, every edge
// split at its midpoint through a Map keyed by the ordered vertex pair. Returns unit
// directions and faces (outward winding, fixed by the signed volume of each face).
function icosphere(subdivisions) {
  const t = (1 + Math.sqrt(5)) / 2;
  let dirs = [
    [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
    [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
  ].map(normalize);
  let faces = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
  ];
  faces = faces.map(([a, b, c]) => {
    const n = cross(sub(dirs[b], dirs[a]), sub(dirs[c], dirs[a]));
    const centroid = add(add(dirs[a], dirs[b]), dirs[c]);
    return dot(n, centroid) < 0 ? [a, c, b] : [a, b, c];
  });
  for (let s = 0; s < subdivisions; s++) {
    const mid = new Map();
    const midpoint = (a, b) => {
      const key = Math.min(a, b) * 65536 + Math.max(a, b);
      let m = mid.get(key);
      if (m === undefined) {
        m = dirs.length;
        dirs.push(normalize(add(dirs[a], dirs[b])));
        mid.set(key, m);
      }
      return m;
    };
    const next = [];
    for (const [a, b, c] of faces) {
      const ab = midpoint(a, b);
      const bc = midpoint(b, c);
      const ca = midpoint(c, a);
      next.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]);
    }
    faces = next;
  }
  return { dirs, faces };
}

// VG_HIGH_POLY: displaced icosphere, subdivision 6 (81920 triangles, 40962 shared vertices).
// Each unit direction d is placed at d * (1 + 0.06 sin(9 x) sin(7 y) sin(8 z)); normals are the
// normalised area-weighted sums of the incident face normals after displacement.
export function highPolyMesh() {
  const { dirs, faces } = icosphere(6);
  const positions = [];
  const uvs = [];
  for (const d of dirs) {
    const k = 1 + 0.06 * Math.sin(9 * d[0]) * Math.sin(7 * d[1]) * Math.sin(8 * d[2]);
    positions.push(d[0] * k, d[1] * k, d[2] * k);
    uvs.push(0.5 + Math.atan2(d[2], d[0]) / TAU, 0.5 - Math.asin(d[1]) / Math.PI);
  }
  const vertex = (i) => positions.slice(3 * i, 3 * i + 3);
  const sums = new Float64Array(positions.length);
  const indices = [];
  for (const [a, b, c] of faces) {
    indices.push(a, b, c);
    const n = cross(sub(vertex(b), vertex(a)), sub(vertex(c), vertex(a)));
    for (const i of [a, b, c]) {
      for (let k = 0; k < 3; k++) sums[3 * i + k] += n[k];
    }
  }
  const normals = [];
  for (let i = 0; i < sums.length; i += 3) {
    normals.push(...normalize(sums.slice(i, i + 3)));
  }
  return pack(positions, normals, uvs, indices);
}

// Foliage: 4096 upright blades, one per cell of a 64 x 64 grid over a 4 m square centred at
// the origin on y = 0. Each blade is a quad of two triangles with four own vertices, 0.03 m
// wide and 0.25 m high; its in-cell offset and yaw come from the LCG (x, z, yaw per cell in
// row-major order). A blade's normal is its face normal.
export function foliagePatch() {
  let state = FOLIAGE_SEED >>> 0;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  const cell = FOLIAGE_EXTENT / FOLIAGE_GRID;
  const x0 = -FOLIAGE_EXTENT / 2;
  const positions = [];
  const normals = [];
  const uvs = [];
  const indices = [];
  const hw = BLADE_WIDTH / 2;
  for (let j = 0; j < FOLIAGE_GRID; j++) {
    for (let i = 0; i < FOLIAGE_GRID; i++) {
      const bx = x0 + (i + next()) * cell;
      const bz = x0 + (j + next()) * cell;
      const yaw = next() * TAU;
      const right = [Math.cos(yaw), 0, -Math.sin(yaw)];
      const normal = normalize(cross(right, [0, 1, 0]));
      const base = positions.length / 3;
      const corners = [[-hw, 0, 0, 0], [hw, 0, 1, 0], [hw, BLADE_HEIGHT, 1, 1], [-hw, BLADE_HEIGHT, 0, 1]];
      for (const [lx, ly, u, v] of corners) {
        positions.push(bx + lx * right[0], ly, bz + lx * right[2]);
        normals.push(normal[0], normal[1], normal[2]);
        uvs.push(u, v);
      }
      indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }
  }
  return pack(positions, normals, uvs, indices);
}

// Appends one quad (corners in counter-clockwise order seen from outside) to the arrays.
function pushQuad(out, corners, normal, uv) {
  const base = out.positions.length / 3;
  corners.forEach((c, i) => {
    out.positions.push(c[0], c[1], c[2]);
    out.normals.push(normal[0], normal[1], normal[2]);
    out.uvs.push(uv[i][0], uv[i][1]);
  });
  out.indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
}

// Closed axis-aligned box with hard edges: 6 faces, 24 vertices, 12 triangles.
function box(center, half) {
  const out = { positions: [], normals: [], uvs: [], indices: [] };
  const uv = [[0, 0], [1, 0], [1, 1], [0, 1]];
  for (let a = 0; a < 3; a++) {
    for (const s of [1, -1]) {
      // For +axis the in-face axes are (a+1, a+2); for -axis they swap, so cross(u, v) = normal.
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
      pushQuad(out, corners, normal, uv);
    }
  }
  return pack(out.positions, out.normals, out.uvs, out.indices);
}

// A 6 m x 3 m x 0.3 m occluder wall, centred on x and z with its base on y = 0.
export function occluderWall() {
  return box([0, 1.5, 0], [3, 1.5, 0.15]);
}

// A 40 m x 40 m square on y = 0 centred at the origin, normal +Y, uvs 0..1.
export function groundPlane() {
  const h = 20;
  const corners = [[-h, 0, -h], [-h, 0, h], [h, 0, h], [h, 0, -h]];
  const uv = [[0, 0], [0, 1], [1, 1], [1, 0]];
  const out = { positions: [], normals: [], uvs: [], indices: [] };
  pushQuad(out, corners, [0, 1, 0], uv);
  return pack(out.positions, out.normals, out.uvs, out.indices);
}

// Axis-aligned bounds of a MeshDesc's positions.
export function meshBounds(mesh) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  const p = mesh.positions;
  for (let i = 0; i < p.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      if (p[i + k] < min[k]) min[k] = p[i + k];
      if (p[i + k] > max[k]) max[k] = p[i + k];
    }
  }
  return { min, max };
}
