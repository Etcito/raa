// The six lookdev objects (R7): inflatable hammer, ice block, plush sphere, neon tube,
// slime pool and floor tiles. Pure ES module (no DOM, importable by node), built only on
// ./geometry.js. Every function returns an S13 MeshDesc (positions, normals, uvs,
// tangents, indices; colours where noted) in metres, right-handed, Y up. Each object
// stands on y = 0 centred on the Y axis (the floor tiles have their top faces at y = 0),
// so a scene only places it. Sub-meshes carry no `material`: a scene sets
// `desc.subMeshes[i].material` to its own handles before `addMesh`.
//
// Everything is deterministic: noise and colour variation come from a seeded integer
// hash, never from Math.random.

import { lathe, sweepTube, roundedBox, uvSphere, merge, withTangents } from './geometry.js';

const TAU = Math.PI * 2;

function fail(message) {
  throw new Error(`objects: ${message}`);
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const mix = (a, b, t) => a.map((x, i) => x + (b[i] - x) * t);

function smoothstep(e0, e1, x) {
  const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1);
  return t * t * (3 - 2 * t);
}

// Integer hash of three lattice coordinates and a seed -> [0, 1).
function hash3(a, b, c, seed) {
  let h = Math.imul(a | 0, 0x8da6b343) ^ Math.imul(b | 0, 0xd8163841)
    ^ Math.imul(c | 0, 0xcb1ab31f) ^ Math.imul(seed | 0, 0x165667b1);
  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d);
  h = Math.imul(h ^ (h >>> 12), 0x297a2d39);
  h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
}

// Smooth value noise in [0, 1) on the integer lattice (quintic fade, trilinear blend).
function valueNoise3(p, seed) {
  const i = p.map(Math.floor);
  const f = p.map((x, k) => x - i[k]);
  const w = f.map((t) => t * t * t * (t * (t * 6 - 15) + 10));
  let sum = 0;
  for (let c = 0; c < 8; c++) {
    const o = [c & 1, (c >> 1) & 1, (c >> 2) & 1];
    const weight = o.reduce((m, ok, k) => m * (ok ? w[k] : 1 - w[k]), 1);
    sum += weight * hash3(i[0] + o[0], i[1] + o[1], i[2] + o[2], seed);
  }
  return sum;
}

// Two-octave fractal value noise in [0, 1).
function fbm3(p, seed) {
  const a = valueNoise3(p, seed);
  const b = valueNoise3(p.map((x) => x * 2.03 + 17.1), seed + 1);
  return (2 * a + b) / 3;
}

// Returns `desc` with positions mapped by fn(p, vertex) -> p'. Normals and tangents are
// left as they are: call `refresh` after a deformation, not after a rigid move.
function mapPositions(desc, fn) {
  const src = desc.positions;
  const out = new Float32Array(src.length);
  for (let v = 0; v < src.length / 3; v++) {
    const p = fn([src[3 * v], src[3 * v + 1], src[3 * v + 2]], v);
    out[3 * v] = p[0];
    out[3 * v + 1] = p[1];
    out[3 * v + 2] = p[2];
  }
  return { ...desc, positions: out };
}

function translate(desc, t) {
  return mapPositions(desc, (p) => [p[0] + t[0], p[1] + t[1], p[2] + t[2]]);
}

// Moves `desc` so that its bounding box is centred on the Y axis with its bottom at y = 0.
function standOnFloor(desc) {
  const p = desc.positions;
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let k = 0; k < p.length; k++) {
    lo[k % 3] = Math.min(lo[k % 3], p[k]);
    hi[k % 3] = Math.max(hi[k % 3], p[k]);
  }
  return translate(desc, [-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2]);
}

// Recomputes smooth normals after a deformation: the area-weighted sum of the face
// normals around each welded position (seam and pole duplicates share one normal), then
// the tangents (withTangents, as raa-mesh). A vertex without a usable sum keeps its old
// normal.
function refresh(desc) {
  const p = desc.positions;
  const n = p.length / 3;
  const key = (v) => `${Math.round(p[3 * v] * 1e6)},${Math.round(p[3 * v + 1] * 1e6)},${Math.round(p[3 * v + 2] * 1e6)}`;
  const groups = new Map();
  const group = new Uint32Array(n);
  for (let v = 0; v < n; v++) {
    const k = key(v);
    if (!groups.has(k)) groups.set(k, groups.size);
    group[v] = groups.get(k);
  }
  const acc = new Float64Array(3 * groups.size);
  const pos = (v) => [p[3 * v], p[3 * v + 1], p[3 * v + 2]];
  const idx = desc.indices;
  for (let t = 0; t + 2 < idx.length; t += 3) {
    const a = pos(idx[t]);
    const fn = cross(sub(pos(idx[t + 1]), a), sub(pos(idx[t + 2]), a));
    for (let k = 0; k < 3; k++) {
      const g = group[idx[t + k]];
      acc[3 * g] += fn[0];
      acc[3 * g + 1] += fn[1];
      acc[3 * g + 2] += fn[2];
    }
  }
  const normals = new Float32Array(desc.normals);
  for (let v = 0; v < n; v++) {
    const g = group[v];
    const s = [acc[3 * g], acc[3 * g + 1], acc[3 * g + 2]];
    const len = Math.hypot(s[0], s[1], s[2]);
    if (len > 1e-12) normals.set(s.map((x) => x / len), 3 * v);
  }
  const { tangents: _stale, ...rest } = desc;
  return withTangents({ ...rest, normals });
}

// Rotates positions, normals and tangent xyz by the 3x3 matrix `m` (rows), a rotation.
function rotate(desc, m) {
  const apply = (src, stride) => {
    const out = new Float32Array(src);
    for (let k = 0; k < src.length; k += stride) {
      const v = [src[k], src[k + 1], src[k + 2]];
      for (let r = 0; r < 3; r++) out[k + r] = dot(m[r], v);
    }
    return out;
  };
  return {
    ...desc,
    positions: apply(desc.positions, 3),
    normals: apply(desc.normals, 3),
    tangents: apply(desc.tangents, 4),
  };
}

// Linear colours of the inflatable (vertex colour stripes).
const HAMMER_RED = [0.75, 0.04, 0.03, 1];
const HAMMER_YELLOW = [0.95, 0.6, 0.04, 1];
const HAMMER_BLUE = [0.03, 0.12, 0.62, 1];
const HAMMER_WHITE = [0.85, 0.85, 0.82, 1];

// Handle: a lathe with a round knob at the bottom and an inflated shaft whose bands are
// pinched where they meet; bands alternate white and blue, the knob is yellow.
function hammerHandle() {
  const radius = 0.045;
  const shaftStart = 0.13;
  const shaftEnd = 0.95;
  const band = 0.1;
  const profile = [];
  for (let i = 0; i <= 10; i++) {
    const t = -Math.PI / 2 + ((3 * Math.PI) / 4) * (i / 10);
    profile.push([i === 0 ? 0 : 0.065 * Math.cos(t), 0.06 + 0.06 * Math.sin(t)]);
  }
  for (let y = shaftStart; y <= shaftEnd + 1e-9; y += 0.01) {
    const u = (y - shaftStart) / band;
    const d = Math.abs(u - Math.round(u)) * band;
    profile.push([radius * (1 - 0.14 * Math.exp(-((d / 0.012) ** 2))), y]);
  }
  profile.push([0.035, 0.965], [0.02, 0.972], [0, 0.975]);
  const color = (p) => {
    if (p[1] < shaftStart - 0.01) return HAMMER_YELLOW;
    const k = Math.floor((p[1] - shaftStart) / band);
    return k % 2 ? HAMMER_BLUE : HAMMER_WHITE;
  };
  return lathe(profile, 32, { color });
}

// Head: an inflated pillow cylinder (superellipse profile, p = 4) about its own axis,
// with a pinched groove ring near each end and six pinched seams along it. Built about
// Y, then laid along X on top of the handle. Panels alternate red and yellow, the end
// caps are blue and the seam welds are white.
function hammerHead() {
  const halfLength = 0.36;
  const radius = 0.19;
  const panels = 6;
  const seamWidth = 0.07;
  const capAt = 0.78;
  const profile = [];
  const steps = 48;
  for (let i = 0; i <= steps; i++) {
    const t = -Math.PI / 2 + (Math.PI * i) / steps;
    const s = Math.sin(t);
    const y = halfLength * Math.sign(s) * Math.sqrt(Math.abs(s));
    const groove = 1 - 0.07 * Math.exp(-(((Math.abs(y) / halfLength - capAt) / 0.035) ** 2));
    const r = i === 0 || i === steps ? 0 : radius * Math.sqrt(Math.abs(Math.cos(t))) * groove;
    profile.push([r, y]);
  }
  const phiOf = (p) => (Math.atan2(-p[2], p[0]) + TAU) % TAU;
  // 0 on a panel centre, 1 on a seam (seams at phi = k * TAU / panels).
  const seam = (p) => {
    const step = TAU / panels;
    const phi = phiOf(p);
    const d = Math.abs(phi - step * Math.round(phi / step));
    return Math.exp(-((d / seamWidth) ** 2));
  };
  const color = (p) => {
    const step = TAU / panels;
    const panel = Math.floor(phiOf(p) / step) % panels;
    const body = mix(panel % 2 ? HAMMER_YELLOW : HAMMER_RED, HAMMER_WHITE, seam(p));
    return mix(body, HAMMER_BLUE, smoothstep(capAt - 0.02, capAt + 0.02, Math.abs(p[1]) / halfLength));
  };
  let head = lathe(profile, 72, { color });
  head = mapPositions(head, (p) => {
    const k = 1 - 0.08 * seam(p);
    return [p[0] * k, p[1], p[2] * k];
  });
  head = refresh(head);
  // Axis Y -> X (rotation of -90 degrees about Z): (x, y, z) -> (y, -x, z). Panel
  // centres then face +Z and -Z, the seams run along the top and bottom.
  head = rotate(head, [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]);
  return translate(head, [0, 1.08, 0]);
}

/**
 * Inflatable toy hammer, about 0.72 m long and 1.27 m tall: sub-mesh 0 is the striped
 * handle (lathe, knob at y = 0), sub-mesh 1 the inflated head with pinched seams laid
 * along X. Both carry linear vertex colours (stripes, panels, caps, seam welds).
 */
export function inflatableHammer() {
  return merge([hammerHandle(), hammerHead()]);
}

/**
 * Ice block: a rounded box (0.8 x 0.7 x 0.8 m, edge radius 0.08 m) whose rounded
 * bevels are chipped inwards by deterministic fractal noise (seeded hash, up to 15 mm);
 * the flat faces stay planar. opts: { seed (integer, default 7) }. No colours.
 */
export function iceBlock(opts = {}) {
  const seed = opts.seed ?? 7;
  if (!Number.isInteger(seed)) fail(`iceBlock seed must be an integer (got ${seed})`);
  const half = [0.4, 0.35, 0.4];
  let box = roundedBox(half, 0.08, 12);
  const n = box.normals;
  box = mapPositions(box, (p, v) => {
    const nv = [n[3 * v], n[3 * v + 1], n[3 * v + 2]];
    const flatness = Math.max(Math.abs(nv[0]), Math.abs(nv[1]), Math.abs(nv[2]));
    const bevel = Math.min(1, (1 - flatness) * 20);
    const depth = 0.015 * bevel * fbm3(p.map((x) => x * 11), seed);
    return [p[0] - depth * nv[0], p[1] - depth * nv[1], p[2] - depth * nv[2]];
  });
  return translate(refresh(box), [0, half[1], 0]);
}

/**
 * Plush sphere: radius 0.45 m resting on y = 0, 64 x 32 segments. uv: u = 0..1 around
 * Y (one seam column), v = 0..1 pole to pole, so fur shells and strand density can be
 * driven from uv later; tangents follow u. No colours.
 */
export function plushSphere() {
  return translate(uvSphere(0.45, 64, 32), [0, 0.45, 0]);
}

// Uniform Catmull-Rom spline through `points` (ends extrapolated), `per` samples per
// span, the last control point included.
function catmullRom(points, per) {
  const n = points.length;
  const at = (i) => {
    if (i < 0) return sub(points[0], sub(points[1], points[0]));
    if (i >= n) return sub(points[n - 1], sub(points[n - 2], points[n - 1]));
    return points[i];
  };
  const out = [];
  for (let i = 0; i + 1 < n; i++) {
    const [p0, p1, p2, p3] = [at(i - 1), at(i), at(i + 1), at(i + 2)];
    for (let s = 0; s < per; s++) {
      const t = s / per;
      const t2 = t * t;
      const t3 = t2 * t;
      out.push([0, 1, 2].map((k) => 0.5 * (2 * p1[k] + (p2[k] - p0[k]) * t
        + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2
        + (3 * p1[k] - p0[k] - 3 * p2[k] + p3[k]) * t3)));
    }
  }
  out.push(points[n - 1].slice());
  return out;
}

// Control points of a script capital S in the XY plane (metres), drawn top to bottom.
const NEON_LETTER = [
  [0.27, 0.64], [0.16, 0.77], [-0.04, 0.8], [-0.21, 0.71], [-0.23, 0.54],
  [-0.06, 0.44], [0.14, 0.37], [0.25, 0.23], [0.21, 0.07], [0.01, 0.0],
  [-0.19, 0.03], [-0.29, 0.14],
].map(([x, y]) => [x, y, 0]);

/**
 * Neon tube: a glass tube of radius 25 mm swept along a script letter S (about 0.6 m
 * wide, 0.85 m tall) in the XY plane facing +Z, round caps, 16 sides. u runs around the
 * tube, v along it (arc length / tube circumference). No colours.
 */
export function neonTube() {
  return standOnFloor(sweepTube(catmullRom(NEON_LETTER, 16), 0.025, 16, { caps: 'round' }));
}

// Quarter (or other) arc of profile points around centre c from angle a0 to a1.
function arc(c, r, a0, a1, steps) {
  return Array.from({ length: steps + 1 }, (_, i) => {
    const a = a0 + ((a1 - a0) * i) / steps;
    return [c[0] + r * Math.cos(a), c[1] + r * Math.sin(a)];
  });
}

/**
 * Slime pool, 1.2 m across: sub-mesh 0 is the closed basin (a lathe: flat underside,
 * outer wall, a round rim 0.16 m high, a sloped inner wall and a floor at 0.04 m),
 * sub-mesh 1 the liquid surface, an open disc facing +Y at y = 0.1 m whose edge sits
 * inside the inner wall (no gap). The surface has planar uvs (u along +X, v towards -Z,
 * like grid) and 16 rings for wave displacement later. No colours.
 */
export function slimePool() {
  const outer = 0.6;
  const rim = 0.05;
  const top = 0.16;
  const floorY = 0.04;
  const level = 0.1;
  const wallTop = [outer - 2 * rim, top - rim];
  const wallBottom = [0.42, floorY + 0.03];
  const basinProfile = [
    [0, 0],
    ...arc([outer - 0.02, 0.02], 0.02, -Math.PI / 2, 0, 4),
    ...arc([outer - rim, top - rim], rim, 0, Math.PI, 16),
    [wallTop[0] + (wallBottom[0] - wallTop[0]) * 0.5, (wallTop[1] + wallBottom[1]) / 2],
    ...arc([wallBottom[0] - 0.03, wallBottom[1]], 0.03, 0, -Math.PI / 2, 6),
    [0.2, floorY],
    [0, floorY],
  ];
  const basin = lathe(basinProfile, 64);
  // Inner wall radius at the liquid level, plus 1 cm into the wall.
  const t = (wallTop[1] - level) / (wallTop[1] - wallBottom[1]);
  const edge = wallTop[0] + (wallBottom[0] - wallTop[0]) * t + 0.01;
  const surfaceProfile = Array.from({ length: 17 }, (_, i) => [edge * (1 - i / 16), level]);
  let surface = lathe(surfaceProfile, 64);
  const uvs = new Float32Array(surface.uvs.length);
  for (let v = 0; v < uvs.length / 2; v++) {
    uvs[2 * v] = surface.positions[3 * v] / (2 * edge) + 0.5;
    uvs[2 * v + 1] = -surface.positions[3 * v + 2] / (2 * edge) + 0.5;
  }
  surface = withTangents({ ...surface, uvs });
  return merge([basin, surface]);
}

// Linear tile colours: warm cream, sage, terracotta, sky blue.
const TILE_COLORS = [
  [0.78, 0.66, 0.48, 1],
  [0.33, 0.48, 0.3, 1],
  [0.62, 0.22, 0.12, 1],
  [0.25, 0.45, 0.7, 1],
];

/**
 * n x n floor tiles (n integer in [1, 64]) on a 0.5 m pitch, centred at the origin,
 * each a 0.48 x 0.06 x 0.48 m rounded box (radius 15 mm) with its top face at y = 0 and
 * a 20 mm grout gap. Each tile has one linear vertex colour, chosen from a four-colour
 * palette by a seeded hash of its (i, j) and varied +-10% in brightness. One sub-mesh.
 */
export function floorTiles(n) {
  if (!Number.isInteger(n) || n < 1 || n > 64) fail(`floorTiles n must be an integer in [1, 64] (got ${n})`);
  const pitch = 0.5;
  const half = [0.24, 0.03, 0.24];
  const tiles = [];
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const base = TILE_COLORS[Math.floor(hash3(i, j, 0, 11) * TILE_COLORS.length)];
      const k = 0.9 + 0.2 * hash3(i, j, 1, 11);
      const color = [base[0] * k, base[1] * k, base[2] * k, 1];
      const x = (i - (n - 1) / 2) * pitch;
      const z = (j - (n - 1) / 2) * pitch;
      tiles.push(translate(roundedBox(half, 0.015, 2, { color }), [x, -half[1], z]));
    }
  }
  const all = merge(tiles);
  return { ...all, subMeshes: [{ first: 0, count: all.indices.length }] };
}

// Signed volume of the triangles indices[first .. first + count) (> 0 on a closed mesh
// with faces counter-clockwise from outside).
function signedVolume(d, first, count) {
  const p = d.positions;
  const idx = d.indices;
  const pos = (v) => [p[3 * v], p[3 * v + 1], p[3 * v + 2]];
  let v = 0;
  for (let t = first; t + 2 < first + count; t += 3) {
    v += dot(pos(idx[t]), cross(pos(idx[t + 1]), pos(idx[t + 2])));
  }
  return v / 6;
}

// Asserts the MeshDesc invariants of `d` (the geometry selfCheck rules); returns
// { vertices, triangles, bboxMin, bboxMax }.
function checkMesh(name, d) {
  const must = (ok, what) => {
    if (!ok) throw new Error(`objects selfCheck ${name}: ${what}`);
  };
  must(d.positions instanceof Float32Array && d.positions.length % 3 === 0, 'positions Float32Array 3n');
  const n = d.positions.length / 3;
  must(n > 0, 'no vertex');
  must(d.normals instanceof Float32Array && d.normals.length === 3 * n, 'normals length 3n');
  must(d.uvs instanceof Float32Array && d.uvs.length === 2 * n, 'uvs length 2n');
  must(d.tangents instanceof Float32Array && d.tangents.length === 4 * n, 'tangents length 4n');
  if (d.colors) must(d.colors instanceof Float32Array && d.colors.length === 4 * n, 'colors length 4n');
  must(d.indices instanceof Uint32Array && d.indices.length > 0, 'indices Uint32Array, not empty');
  must(d.indices.length % 3 === 0, 'indices length % 3');
  for (const key of ['positions', 'normals', 'uvs', 'tangents', 'colors']) {
    const a = d[key];
    if (!a) continue;
    const bad = a.findIndex((x) => !Number.isFinite(x));
    must(bad < 0, `${key}[${bad}] is not finite`);
  }
  const badIndex = d.indices.findIndex((i) => i >= n);
  must(badIndex < 0, `indices[${badIndex}] >= ${n}`);
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let v = 0; v < n; v++) {
    const nv = [d.normals[3 * v], d.normals[3 * v + 1], d.normals[3 * v + 2]];
    must(Math.abs(Math.hypot(...nv) - 1) <= 1e-4, `normal ${v} length ${Math.hypot(...nv)}`);
    const t = [d.tangents[4 * v], d.tangents[4 * v + 1], d.tangents[4 * v + 2]];
    must(Math.abs(Math.hypot(...t) - 1) <= 1e-4, `tangent ${v} length ${Math.hypot(...t)}`);
    must(Math.abs(dot(t, nv)) <= 1e-3, `tangent ${v} not orthogonal to its normal`);
    const w = d.tangents[4 * v + 3];
    must(w === 1 || w === -1, `tangent ${v} w = ${w}`);
    for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k], d.positions[3 * v + k]);
      hi[k] = Math.max(hi[k], d.positions[3 * v + k]);
    }
  }
  if (d.subMeshes) {
    let end = 0;
    for (const [i, s] of d.subMeshes.entries()) {
      must(s.first % 3 === 0 && s.count % 3 === 0 && s.count > 0, `subMeshes[${i}] first/count`);
      must(s.first >= end && s.first + s.count <= d.indices.length, `subMeshes[${i}] range`);
      end = s.first + s.count;
    }
  }
  return { vertices: n, triangles: d.indices.length / 3, bboxMin: lo, bboxMax: hi };
}

/**
 * Builds the six objects and asserts, per object: the MeshDesc invariants of the
 * geometry selfCheck (array lengths 3n/3n/2n/4n(/4n), indices % 3 and < n, finite
 * values, unit normals and tangents within 1e-4, tangent w = +-1, sub-mesh ranges), the
 * sub-mesh count (hammer 2, pool 2, others 1), colours present on the hammer and the
 * tiles, outward winding (signed volume > 0) on every closed part, the size (largest
 * extent 0.4..1.6 m; tiles 0.5 m pitch) and the resting height (min y = 0; tiles: top
 * y = 0); plus iceBlock determinism (same seed equal, other seed different). Throws on
 * the first failure; prints one line per object and returns the measured values.
 * Run: node --input-type=module -e "import {selfCheck} from './examples/lookdev/lib/objects.js'; selfCheck()"
 */
export function selfCheck(log = console.log) {
  const cases = [
    { name: 'inflatableHammer', desc: inflatableHammer(), subs: 2, colors: true, closed: [0, 1] },
    { name: 'iceBlock', desc: iceBlock(), subs: 1, colors: false, closed: [0] },
    { name: 'plushSphere', desc: plushSphere(), subs: 1, colors: false, closed: [0] },
    { name: 'neonTube', desc: neonTube(), subs: 1, colors: false, closed: [0] },
    { name: 'slimePool', desc: slimePool(), subs: 2, colors: false, closed: [0] },
    { name: 'floorTiles(4)', desc: floorTiles(4), subs: 1, colors: true, closed: [0], tiles: 4 },
  ];
  const results = [];
  for (const c of cases) {
    const must = (ok, what) => {
      if (!ok) throw new Error(`objects selfCheck ${c.name}: ${what}`);
    };
    const d = c.desc;
    const m = checkMesh(c.name, d);
    const subs = d.subMeshes && d.subMeshes.length ? d.subMeshes : [{ first: 0, count: d.indices.length }];
    must(subs.length === c.subs, `${subs.length} sub-meshes, expected ${c.subs}`);
    must(subs.every((s) => s.material === undefined), 'sub-meshes must leave material to the scene');
    must(Boolean(d.colors) === c.colors, c.colors ? 'colours missing' : 'unexpected colours');
    const volumes = c.closed.map((i) => signedVolume(d, subs[i].first, subs[i].count));
    volumes.forEach((v, i) => must(v > 0, `sub-mesh ${c.closed[i]} signed volume ${v} is not > 0`));
    const size = m.bboxMax.map((x, k) => x - m.bboxMin[k]);
    if (c.tiles) {
      const span = c.tiles * 0.5 - 0.02;
      must(Math.abs(size[0] - span) < 1e-4 && Math.abs(size[2] - span) < 1e-4, `tiles span ${size}`);
      must(Math.abs(m.bboxMax[1]) < 1e-5, `tiles top y ${m.bboxMax[1]} != 0`);
    } else {
      const largest = Math.max(...size);
      must(largest >= 0.4 && largest <= 1.6, `largest extent ${largest} m outside 0.4..1.6`);
      must(Math.abs(m.bboxMin[1]) < 1e-5, `min y ${m.bboxMin[1]} != 0`);
    }
    results.push({ name: c.name, ...m, subMeshes: subs.length, volumes });
    log(`selfCheck ${c.name}: vertices ${m.vertices}, triangles ${m.triangles}, sub-meshes ${subs.length}, colours ${c.colors ? 'yes' : 'no'}, size ${size.map((x) => x.toFixed(3)).join(' x ')} m | PASS`);
  }
  const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
  if (!same(iceBlock().positions, iceBlock().positions)) throw new Error('objects selfCheck iceBlock: not deterministic');
  if (same(iceBlock().positions, iceBlock({ seed: 8 }).positions)) throw new Error('objects selfCheck iceBlock: seed has no effect');
  log('selfCheck iceBlock determinism | PASS');
  log(`selfCheck: ${results.length} objects PASS`);
  return results;
}
