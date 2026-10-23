// Procedural mesh builders of the lookdev page. Pure ES module (no DOM, importable by
// node). Every builder returns an S13 MeshDesc: Float32Array positions (3n), normals (3n,
// unit), uvs (2n), tangents (4n, xyz unit, w = +1|-1) and a Uint32Array of indices (3m);
// `colors` (Float32Array 4n, linear rgba) when its opts carry a `color`. Right-handed,
// Y up, metres; triangles are counter-clockwise seen from outside, normals point outward.
//
// `opts.color` is either a constant [r, g, b, a?] (linear) or a function
// `(position, normal, uv, info) -> [r, g, b, a?]` evaluated once per vertex; `info` names
// the vertex in the builder's own grid ({ ring, side } for lathe, uvSphere and sweepTube,
// { face, i, j } for roundedBox, { i, j } for grid). A missing alpha is 1.
//
// uv convention: u runs around (or along +X), v runs up (or along the path, or towards -Z
// on grid), so every builder yields tangent handedness w = +1 on its smooth surfaces.

const TAU = Math.PI * 2;

function fail(message) {
  throw new Error(`geometry: ${message}`);
}

function checkPositive(name, v) {
  if (!Number.isFinite(v) || v <= 0) fail(`${name} must be a finite number > 0 (got ${v})`);
}

function checkCount(name, v, min) {
  if (!Number.isInteger(v) || v < min) fail(`${name} must be an integer >= ${min} (got ${v})`);
}

function checkVec3(name, v) {
  if (!Array.isArray(v) && !ArrayBuffer.isView(v)) fail(`${name} must be [x, y, z]`);
  if (v.length !== 3 || ![0, 1, 2].every((i) => Number.isFinite(v[i]))) {
    fail(`${name} must be 3 finite numbers`);
  }
}

const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const length = (a) => Math.hypot(a[0], a[1], a[2]);

function normalize(a) {
  const len = length(a);
  return len > 0 ? scale(a, 1 / len) : null;
}

// Branchless orthonormal basis (Duff et al. 2017): the S10 default tangent of a unit
// normal, w = 1.
function defaultTangent(n) {
  const sign = n[2] >= 0 ? 1 : -1;
  const a = -1 / (sign + n[2]);
  const b = n[0] * n[1] * a;
  return [1 + sign * n[0] * n[0] * a, sign * b, -sign * n[0]];
}

function colorFn(color) {
  if (color === undefined || color === null) return null;
  if (typeof color === 'function') return color;
  if (Array.isArray(color) || ArrayBuffer.isView(color)) return () => color;
  fail('opts.color must be [r, g, b, a?] or a function');
}

// Accumulates vertices and triangles, then emits the MeshDesc.
class Builder {
  constructor(opts) {
    this.positions = [];
    this.normals = [];
    this.uvs = [];
    this.indices = [];
    this.color = colorFn(opts && opts.color);
    this.colors = this.color ? [] : null;
  }

  get count() {
    return this.positions.length / 3;
  }

  vertex(p, n, uv, info) {
    const nn = normalize(n);
    if (!nn || !nn.every(Number.isFinite)) fail(`zero or non-finite normal at vertex ${this.count}`);
    this.positions.push(p[0], p[1], p[2]);
    this.normals.push(nn[0], nn[1], nn[2]);
    this.uvs.push(uv[0], uv[1]);
    if (this.colors) {
      const c = this.color(p, nn, uv, info);
      if (!c || c.length < 3 || ![0, 1, 2].every((i) => Number.isFinite(c[i]))) {
        fail(`opts.color returned an invalid colour at vertex ${this.count}`);
      }
      const alpha = c.length > 3 ? c[3] : 1;
      if (!Number.isFinite(alpha)) fail(`opts.color returned a non-finite alpha`);
      this.colors.push(c[0], c[1], c[2], alpha);
    }
    return this.count - 1;
  }

  tri(a, b, c) {
    this.indices.push(a, b, c);
  }

  finish() {
    const desc = {
      positions: new Float32Array(this.positions),
      normals: new Float32Array(this.normals),
      uvs: new Float32Array(this.uvs),
      indices: new Uint32Array(this.indices),
    };
    if (this.colors) desc.colors = new Float32Array(this.colors);
    return withTangents(desc);
  }
}

// Emits rings of `sides + 1` vertices (the seam column duplicated for uv) and stitches
// consecutive rings. A ring is { c, nAxis, bAxis, aAxis, r, nr, na, v, pole, crease }:
// vertex k sits at c + r * radial(k) with radial(k) = cos(phi) nAxis + sin(phi) bAxis,
// its normal is nr * radial(k) + na * aAxis. bAxis = aAxis x nAxis and rings advance
// along aAxis, which makes the quads counter-clockwise from outside. A pole ring (r = 0)
// drops the degenerate triangle of each quad; a crease ring (same positions as the ring
// before it, other normals) is not stitched to it.
function emitRings(b, rings, sides) {
  const starts = [];
  rings.forEach((ring, ri) => {
    starts.push(b.count);
    for (let k = 0; k <= sides; k++) {
      const phi = (TAU * (k % sides)) / sides;
      const radial = add(scale(ring.nAxis, Math.cos(phi)), scale(ring.bAxis, Math.sin(phi)));
      const p = add(ring.c, scale(radial, ring.r));
      const n = add(scale(radial, ring.nr), scale(ring.aAxis, ring.na));
      b.vertex(p, n, [k / sides, ring.v], { ring: ri, side: k });
    }
  });
  for (let i = 0; i + 1 < rings.length; i++) {
    const lo = rings[i];
    const hi = rings[i + 1];
    if (hi.crease || (lo.pole && hi.pole)) continue;
    for (let k = 0; k < sides; k++) {
      const a = starts[i] + k;
      const bb = a + 1;
      const c = starts[i + 1] + k;
      const d = c + 1;
      if (!lo.pole) b.tri(a, bb, c);
      if (!hi.pole) b.tri(bb, d, c);
    }
  }
}

const LATHE_N = [1, 0, 0];
const LATHE_B = [0, 0, -1];
const LATHE_A = [0, 1, 0];

/**
 * Surface of revolution about +Y. `profile` is [[r, y], ...] (r >= 0) travelled with the
 * outside on its right in the (r, y) half-plane, i.e. bottom to top for a convex body.
 * Points with r = 0 close the surface at the axis (poles). Repeating a point makes a hard
 * crease there (each copy takes the normal of its own side), e.g. the rim of a flat cap:
 * [[0, -1], [0.5, -1], [0.5, -1], [0.5, 1], [0.5, 1], [0, 1]] is a closed cylinder.
 * u = segment / segments around (from +X towards -Z), v = arc length / total along the
 * profile. opts: { color }; colour info = { ring: profile index, side }.
 */
export function lathe(profile, segments, opts = {}) {
  if (!Array.isArray(profile) || profile.length < 2) fail('lathe profile needs >= 2 points');
  checkCount('lathe segments', segments, 3);
  profile.forEach((pt, i) => {
    if (!pt || pt.length !== 2 || !Number.isFinite(pt[0]) || !Number.isFinite(pt[1])) {
      fail(`lathe profile[${i}] must be [r, y] finite`);
    }
    if (pt[0] < 0) fail(`lathe profile[${i}] has r < 0`);
  });
  const n = profile.length;
  const extent = profile.reduce((m, [r, y]) => Math.max(m, Math.abs(r), Math.abs(y)), 1);
  const eps = 1e-9 * extent;
  // Outward normal of each profile segment: the tangent (dr, dy) turned clockwise.
  const segN = [];
  const arc = [0];
  for (let i = 0; i + 1 < n; i++) {
    const dr = profile[i + 1][0] - profile[i][0];
    const dy = profile[i + 1][1] - profile[i][1];
    const len = Math.hypot(dr, dy);
    segN.push(len > eps ? [dy / len, -dr / len] : null);
    arc.push(arc[i] + (len > eps ? len : 0));
  }
  const total = arc[n - 1];
  if (!(total > 0)) fail('lathe profile has zero length');
  const rings = profile.map(([r, y], i) => {
    const prev = i > 0 ? segN[i - 1] : undefined;
    const next = i < n - 1 ? segN[i] : undefined;
    let m;
    if (prev === null || prev === undefined) m = next;
    else if (next === null || next === undefined) m = prev;
    else m = [prev[0] + next[0], prev[1] + next[1]];
    const len = m ? Math.hypot(m[0], m[1]) : 0;
    if (!(len > 1e-6)) fail(`lathe profile[${i}]: no normal (repeated point or reversal)`);
    const pole = r <= eps;
    let nr = m[0] / len;
    let na = m[1] / len;
    if (pole) {
      nr = 0;
      na = na < 0 ? -1 : 1;
    }
    return {
      c: [0, y, 0],
      nAxis: LATHE_N,
      bAxis: LATHE_B,
      aAxis: LATHE_A,
      r: pole ? 0 : r,
      nr,
      na,
      v: arc[i] / total,
      pole,
      crease: i > 0 && segN[i - 1] === null,
    };
  });
  const b = new Builder(opts);
  emitRings(b, rings, segments);
  return b.finish();
}

/**
 * Sphere centred at the origin with exact radial normals. segU >= 3 segments around Y,
 * segV >= 2 from the south pole (v = 0) to the north pole (v = 1). opts: { color }.
 */
export function uvSphere(radius, segU, segV, opts = {}) {
  checkPositive('uvSphere radius', radius);
  checkCount('uvSphere segU', segU, 3);
  checkCount('uvSphere segV', segV, 2);
  const rings = [];
  for (let j = 0; j <= segV; j++) {
    const pole = j === 0 || j === segV;
    const phi = -Math.PI / 2 + (Math.PI * j) / segV;
    rings.push({
      c: [0, pole ? (j === 0 ? -radius : radius) : radius * Math.sin(phi), 0],
      nAxis: LATHE_N,
      bAxis: LATHE_B,
      aAxis: LATHE_A,
      r: pole ? 0 : radius * Math.cos(phi),
      nr: pole ? 0 : Math.cos(phi),
      na: pole ? (j === 0 ? -1 : 1) : Math.sin(phi),
      v: j / segV,
      pole,
      crease: false,
    });
  }
  const b = new Builder(opts);
  emitRings(b, rings, segU);
  return b.finish();
}

// Rotation-minimising frames along a polyline (double reflection, Wang et al. 2008).
function transport(points, tangents, n0) {
  const frames = [n0];
  for (let i = 0; i + 1 < points.length; i++) {
    const v1 = sub(points[i + 1], points[i]);
    const c1 = dot(v1, v1);
    let r = frames[i];
    let t = tangents[i];
    if (c1 > 0) {
      r = sub(r, scale(v1, (2 / c1) * dot(v1, r)));
      t = sub(t, scale(v1, (2 / c1) * dot(v1, t)));
    }
    const v2 = sub(tangents[i + 1], t);
    const c2 = dot(v2, v2);
    if (c2 > 0) r = sub(r, scale(v2, (2 / c2) * dot(v2, r)));
    const t1 = tangents[i + 1];
    frames.push(normalize(sub(r, scale(t1, dot(r, t1)))) || defaultTangent(t1));
  }
  return frames;
}

function rotateAbout(v, axis, angle) {
  return add(scale(v, Math.cos(angle)), scale(cross(axis, v), Math.sin(angle)));
}

/**
 * Tube of `radius` with `sides` segments around the polyline `path` ([x, y, z][], no two
 * consecutive points equal), oriented by rotation-minimising frames. Cross-sections are
 * perpendicular to the averaged path direction, so feed it a finely sampled smooth path.
 * opts: { closed: false (the path loops back to its first point, which is not repeated;
 * the frame twist is spread over the loop), caps: 'round' | 'flat' | 'none' (open paths,
 * default 'round'), capSegments (default max(2, round(sides / 4))), color }.
 * u runs around the tube, v = arc length / (2 pi radius) along the path (caps extend it).
 */
export function sweepTube(path, radius, sides, opts = {}) {
  checkPositive('sweepTube radius', radius);
  checkCount('sweepTube sides', sides, 3);
  if (!Array.isArray(path)) fail('sweepTube path must be an array of [x, y, z]');
  path.forEach((p, i) => checkVec3(`sweepTube path[${i}]`, p));
  const closed = Boolean(opts.closed);
  const caps = closed ? 'none' : (opts.caps ?? 'round');
  if (!['round', 'flat', 'none'].includes(caps)) fail(`sweepTube caps '${caps}' is unknown`);
  const capSeg = opts.capSegments ?? Math.max(2, Math.round(sides / 4));
  checkCount('sweepTube capSegments', capSeg, 1);
  const pts = path.map((p) => [p[0], p[1], p[2]]);
  const n = pts.length;
  if (n < (closed ? 3 : 2)) fail(`sweepTube path needs >= ${closed ? 3 : 2} points`);
  const segs = closed ? n : n - 1;
  const arc = [0];
  for (let i = 0; i < segs; i++) {
    const len = length(sub(pts[(i + 1) % n], pts[i]));
    if (!(len > 0)) fail(`sweepTube path[${i}] and the next point are equal`);
    arc.push(arc[i] + len);
  }
  const tangents = pts.map((p, i) => {
    const hasPrev = closed || i > 0;
    const hasNext = closed || i < n - 1;
    const dIn = hasPrev ? normalize(sub(p, pts[(i - 1 + n) % n])) : null;
    const dOut = hasNext ? normalize(sub(pts[(i + 1) % n], p)) : null;
    const t = dIn && dOut ? normalize(add(dIn, dOut)) : dIn || dOut;
    if (!t) fail(`sweepTube path reverses at point ${i}`);
    return t;
  });
  const t0 = tangents[0];
  const ax = [0, 1, 2].reduce((m, i) => (Math.abs(t0[i]) < Math.abs(t0[m]) ? i : m), 0);
  const e = [0, 0, 0];
  e[ax] = 1;
  const n0 = normalize(sub(e, scale(t0, dot(e, t0))));
  let normals;
  if (closed) {
    const loop = transport([...pts, pts[0]], [...tangents, t0], n0);
    const end = loop[n];
    const twist = Math.atan2(dot(cross(end, n0), t0), dot(end, n0));
    normals = loop.slice(0, n).map((f, i) => rotateAbout(f, tangents[i], (twist * i) / n));
  } else {
    normals = transport(pts, tangents, n0);
  }
  const vs = 1 / (TAU * radius);
  const frame = (i) => ({ nAxis: normals[i], bAxis: cross(tangents[i], normals[i]), aAxis: tangents[i] });
  const ring = (i, fields) => ({ ...frame(i), r: radius, nr: 1, na: 0, pole: false, crease: false, ...fields });
  const rings = [];
  const last = n - 1;
  const sEnd = arc[segs];
  if (caps === 'round') {
    for (let m = capSeg; m >= 1; m--) {
      const a = ((Math.PI / 2) * m) / capSeg;
      const pole = m === capSeg;
      rings.push(ring(0, {
        c: sub(pts[0], scale(t0, radius * Math.sin(a))),
        r: pole ? 0 : radius * Math.cos(a),
        nr: pole ? 0 : Math.cos(a),
        na: pole ? -1 : -Math.sin(a),
        v: -radius * a * vs,
        pole,
      }));
    }
  } else if (caps === 'flat') {
    rings.push(ring(0, { c: pts[0], r: 0, nr: 0, na: -1, v: -radius * vs, pole: true }));
    rings.push(ring(0, { c: pts[0], nr: 0, na: -1, v: 0 }));
  }
  for (let i = 0; i < n; i++) {
    rings.push(ring(i, { c: pts[i], v: arc[i] * vs, crease: i === 0 && caps === 'flat' }));
  }
  if (closed) rings.push(ring(0, { c: pts[0], v: sEnd * vs }));
  if (caps === 'round') {
    const tn = tangents[last];
    for (let m = 1; m <= capSeg; m++) {
      const a = ((Math.PI / 2) * m) / capSeg;
      const pole = m === capSeg;
      rings.push(ring(last, {
        c: add(pts[last], scale(tn, radius * Math.sin(a))),
        r: pole ? 0 : radius * Math.cos(a),
        nr: pole ? 0 : Math.cos(a),
        na: pole ? 1 : Math.sin(a),
        v: (sEnd + radius * a) * vs,
        pole,
      }));
    }
  } else if (caps === 'flat') {
    rings.push(ring(last, { c: pts[last], nr: 0, na: 1, v: sEnd * vs, crease: true }));
    rings.push(ring(last, { c: pts[last], r: 0, nr: 0, na: 1, v: (sEnd + radius) * vs, pole: true }));
  }
  const b = new Builder(opts);
  emitRings(b, rings, sides);
  return b.finish();
}

// Faces of roundedBox: outward axis `a` with sign `s`, and the signed in-face axes u
// (right) and v (up) with u x v = s * e_a (counter-clockwise from outside).
const BOX_FACES = [
  { name: '+x', a: 0, s: 1, u: [2, -1], v: [1, 1] },
  { name: '-x', a: 0, s: -1, u: [2, 1], v: [1, 1] },
  { name: '+y', a: 1, s: 1, u: [0, 1], v: [2, -1] },
  { name: '-y', a: 1, s: -1, u: [0, 1], v: [2, 1] },
  { name: '+z', a: 2, s: 1, u: [0, 1], v: [1, 1] },
  { name: '-z', a: 2, s: -1, u: [0, -1], v: [1, 1] },
];

// Samples of one face axis on the outer box: the flat span [-c, c] plus `m` steps of
// each 45 degree half of the rounded edges (tan spacing gives even angles on the arc).
function boxSamples(h, r, m) {
  const c = h - r;
  const arc = [];
  if (r > 0) for (let k = 1; k <= m; k++) arc.push(c + r * Math.tan((Math.PI / 4) * (k / m)));
  const inner = c > 1e-9 * h ? [-c, c] : [0];
  return [...arc.map((x) => -x).reverse(), ...inner, ...arc];
}

/**
 * Box of half extents `half` ([x, y, z], each > 0) with edges and corners rounded by
 * `radius` (0 <= radius <= min(half); 0 gives a sharp box with hard edges), centred at
 * the origin. `segments` (>= 1) is the number of segments of each quarter-circle edge.
 * Six face grids that meet on the 45 degree lines of the rounding (positions and normals
 * equal there, so shading is smooth). uv in [0, 1] per face, v up on the side faces.
 * opts: { color }; colour info = { face: '+x' | '-x' | ..., i, j }.
 */
export function roundedBox(half, radius, segments, opts = {}) {
  checkVec3('roundedBox half', half);
  half.forEach((h, i) => checkPositive(`roundedBox half[${i}]`, h));
  const minHalf = Math.min(half[0], half[1], half[2]);
  if (!Number.isFinite(radius) || radius < 0 || radius > minHalf) {
    fail(`roundedBox radius must be in [0, ${minHalf}] (got ${radius})`);
  }
  checkCount('roundedBox segments', segments, 1);
  const m = Math.max(1, Math.ceil(segments / 2));
  const samples = [0, 1, 2].map((i) => boxSamples(half[i], radius, m));
  const inner = half.map((h) => h - radius);
  const b = new Builder(opts);
  for (const f of BOX_FACES) {
    const [ui, us] = f.u;
    const [vi, vs] = f.v;
    const su = samples[ui];
    const sv = samples[vi];
    const start = b.count;
    sv.forEach((qv, j) => {
      su.forEach((qu, i) => {
        const p = [0, 0, 0];
        p[f.a] = f.s * half[f.a];
        p[ui] = us * qu;
        p[vi] = vs * qv;
        let pos = p;
        let nrm = [0, 0, 0];
        nrm[f.a] = f.s;
        if (radius > 0) {
          const core = p.map((x, k) => Math.min(Math.max(x, -inner[k]), inner[k]));
          nrm = normalize(sub(p, core));
          pos = add(core, scale(nrm, radius));
        }
        const uv = [(qu + half[ui]) / (2 * half[ui]), (qv + half[vi]) / (2 * half[vi])];
        b.vertex(pos, nrm, uv, { face: f.name, i, j });
      });
    });
    const nu = su.length;
    for (let j = 0; j + 1 < sv.length; j++) {
      for (let i = 0; i + 1 < nu; i++) {
        const a = start + j * nu + i;
        b.tri(a, a + 1, a + nu);
        b.tri(a + 1, a + nu + 1, a + nu);
      }
    }
  }
  return b.finish();
}

/**
 * Flat grid in the XZ plane at y = 0, normal +Y, centred at the origin: width `w` along
 * X, depth `d` along Z, nx x nz quads (each >= 1). u = 0..1 along +X, v = 0..1 towards -Z
 * (the far edge is v = 1). opts: { color }; colour info = { i, j } (vertex column, row).
 */
export function grid(w, d, nx, nz, opts = {}) {
  checkPositive('grid w', w);
  checkPositive('grid d', d);
  checkCount('grid nx', nx, 1);
  checkCount('grid nz', nz, 1);
  const b = new Builder(opts);
  for (let j = 0; j <= nz; j++) {
    for (let i = 0; i <= nx; i++) {
      const p = [-w / 2 + (w * i) / nx, 0, -d / 2 + (d * j) / nz];
      b.vertex(p, [0, 1, 0], [i / nx, 1 - j / nz], { i, j });
    }
  }
  const row = nx + 1;
  for (let j = 0; j < nz; j++) {
    for (let i = 0; i < nx; i++) {
      const a = j * row + i;
      b.tri(a, a + row, a + 1);
      b.tri(a + row, a + row + 1, a + 1);
    }
  }
  return b.finish();
}

/**
 * Returns `desc` with per-vertex tangents (the other arrays are shared, not copied).
 * Rule (as raa-mesh): per triangle, the uv gradient directions dP/du (tangent) and dP/dv
 * (bitangent) are summed on its three vertices (triangles with a degenerate uv mapping
 * add nothing); per vertex the sum is Gram-Schmidt orthogonalised against the normal and
 * normalised, w = -1 when cross(normal, tangent) points away from the summed dP/dv,
 * else +1. Vertices without a usable sum (or a desc without uvs) get the S10 default
 * tangent of their normal (Duff et al. 2017), w = 1.
 */
export function withTangents(desc) {
  const pos = desc.positions;
  const nrm = desc.normals;
  const uv = desc.uvs;
  const idx = desc.indices;
  const n = pos.length / 3;
  const tan = new Float64Array(3 * n);
  const bit = new Float64Array(3 * n);
  if (uv && uv.length === 2 * n) {
    for (let t = 0; t + 2 < idx.length; t += 3) {
      const [i0, i1, i2] = [idx[t], idx[t + 1], idx[t + 2]];
      const e1 = [pos[3 * i1] - pos[3 * i0], pos[3 * i1 + 1] - pos[3 * i0 + 1], pos[3 * i1 + 2] - pos[3 * i0 + 2]];
      const e2 = [pos[3 * i2] - pos[3 * i0], pos[3 * i2 + 1] - pos[3 * i0 + 1], pos[3 * i2 + 2] - pos[3 * i0 + 2]];
      const du1 = uv[2 * i1] - uv[2 * i0];
      const dv1 = uv[2 * i1 + 1] - uv[2 * i0 + 1];
      const du2 = uv[2 * i2] - uv[2 * i0];
      const dv2 = uv[2 * i2 + 1] - uv[2 * i0 + 1];
      const det = du1 * dv2 - du2 * dv1;
      if (!(Math.abs(det) > 1e-14)) continue;
      const sdir = scale(sub(scale(e1, dv2), scale(e2, dv1)), 1 / det);
      const tdir = scale(sub(scale(e2, du1), scale(e1, du2)), 1 / det);
      if (!sdir.every(Number.isFinite) || !tdir.every(Number.isFinite)) continue;
      for (const v of [i0, i1, i2]) {
        for (let k = 0; k < 3; k++) {
          tan[3 * v + k] += sdir[k];
          bit[3 * v + k] += tdir[k];
        }
      }
    }
  }
  const out = new Float32Array(4 * n);
  for (let v = 0; v < n; v++) {
    const nv = normalize([nrm[3 * v], nrm[3 * v + 1], nrm[3 * v + 2]]) || [0, 1, 0];
    const raw = [tan[3 * v], tan[3 * v + 1], tan[3 * v + 2]];
    const ortho = sub(raw, scale(nv, dot(nv, raw)));
    const rawLen = length(raw);
    const len = length(ortho);
    let t;
    let w = 1;
    if (rawLen > 0 && len > 1e-6 * rawLen) {
      t = scale(ortho, 1 / len);
      if (dot(cross(nv, t), [bit[3 * v], bit[3 * v + 1], bit[3 * v + 2]]) < 0) w = -1;
    } else {
      t = defaultTangent(nv);
    }
    out.set([t[0], t[1], t[2], w], 4 * v);
  }
  return { ...desc, tangents: out };
}

function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

/**
 * Concatenates `parts` into one MeshDesc. Indices are offset by each part's vertex base;
 * sub-meshes are kept (first offset by the part's index base, `material` copied), a part
 * without subMeshes becomes one sub-mesh over its indices, so the result always has one
 * sub-mesh per part range. Missing uvs become (0, 0), missing tangents are generated
 * (withTangents); when any part has colours, the others get (1, 1, 1, 1), and Uint8Array
 * (sRGB rgba8) colours are converted to linear floats.
 */
export function merge(parts) {
  if (!Array.isArray(parts) || parts.length === 0) fail('merge needs a non-empty array of MeshDesc');
  const list = parts.map((p, i) => {
    if (!p || !(p.positions instanceof Float32Array) || !(p.normals instanceof Float32Array)) {
      fail(`merge part ${i} needs Float32Array positions and normals`);
    }
    if (!(p.indices instanceof Uint32Array)) fail(`merge part ${i} needs Uint32Array indices`);
    const n = p.positions.length / 3;
    return p.tangents && p.tangents.length === 4 * n ? p : withTangents(p);
  });
  const vertices = list.reduce((s, p) => s + p.positions.length / 3, 0);
  const indexCount = list.reduce((s, p) => s + p.indices.length, 0);
  const anyColors = list.some((p) => p.colors);
  const out = {
    positions: new Float32Array(3 * vertices),
    normals: new Float32Array(3 * vertices),
    uvs: new Float32Array(2 * vertices),
    tangents: new Float32Array(4 * vertices),
    indices: new Uint32Array(indexCount),
  };
  if (anyColors) out.colors = new Float32Array(4 * vertices).fill(1);
  const subMeshes = [];
  let vBase = 0;
  let iBase = 0;
  for (const p of list) {
    const n = p.positions.length / 3;
    out.positions.set(p.positions, 3 * vBase);
    out.normals.set(p.normals, 3 * vBase);
    if (p.uvs && p.uvs.length === 2 * n) out.uvs.set(p.uvs, 2 * vBase);
    out.tangents.set(p.tangents, 4 * vBase);
    if (p.colors) {
      const c = p.colors instanceof Uint8Array
        ? Float32Array.from(p.colors, (x, k) => (k % 4 === 3 ? x / 255 : srgbToLinear(x / 255)))
        : p.colors;
      out.colors.set(c, 4 * vBase);
    }
    for (let k = 0; k < p.indices.length; k++) out.indices[iBase + k] = p.indices[k] + vBase;
    const subs = p.subMeshes && p.subMeshes.length
      ? p.subMeshes
      : [{ first: 0, count: p.indices.length }];
    for (const s of subs) {
      const sm = { first: s.first + iBase, count: s.count };
      if (s.material !== undefined) sm.material = s.material;
      subMeshes.push(sm);
    }
    vBase += n;
    iBase += p.indices.length;
  }
  out.subMeshes = subMeshes;
  return out;
}

// Signed volume of a closed mesh (> 0 when faces are counter-clockwise from outside).
function signedVolume(d) {
  const p = d.positions;
  const idx = d.indices;
  let v = 0;
  for (let t = 0; t + 2 < idx.length; t += 3) {
    const a = 3 * idx[t];
    const b = 3 * idx[t + 1];
    const c = 3 * idx[t + 2];
    v += dot([p[a], p[a + 1], p[a + 2]], cross([p[b], p[b + 1], p[b + 2]], [p[c], p[c + 1], p[c + 2]]));
  }
  return v / 6;
}

// Asserts the MeshDesc invariants of `d`; returns { vertices, triangles, volume }.
function checkMesh(name, d, closed) {
  const must = (ok, what) => {
    if (!ok) throw new Error(`geometry selfCheck ${name}: ${what}`);
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
  for (let v = 0; v < n; v++) {
    const nv = [d.normals[3 * v], d.normals[3 * v + 1], d.normals[3 * v + 2]];
    must(Math.abs(length(nv) - 1) <= 1e-4, `normal ${v} length ${length(nv)}`);
    const t = [d.tangents[4 * v], d.tangents[4 * v + 1], d.tangents[4 * v + 2]];
    must(Math.abs(length(t) - 1) <= 1e-4, `tangent ${v} length ${length(t)}`);
    must(Math.abs(dot(t, nv)) <= 1e-3, `tangent ${v} not orthogonal to its normal`);
    const w = d.tangents[4 * v + 3];
    must(w === 1 || w === -1, `tangent ${v} w = ${w}`);
  }
  if (d.subMeshes) {
    let end = 0;
    for (const [i, s] of d.subMeshes.entries()) {
      must(s.first % 3 === 0 && s.count % 3 === 0 && s.count > 0, `subMeshes[${i}] first/count`);
      must(s.first >= end && s.first + s.count <= d.indices.length, `subMeshes[${i}] range`);
      end = s.first + s.count;
    }
  }
  const volume = signedVolume(d);
  if (closed) must(volume > 0, `signed volume ${volume} is not > 0 (inward winding)`);
  return { vertices: n, triangles: d.indices.length / 3, volume };
}

/**
 * Builds every builder on sample parameters and asserts the MeshDesc invariants: array
 * lengths 3n/3n/2n/4n(/4n), indices % 3 and < n, finite values, unit normals and
 * tangents within 1e-4, tangent w = +-1, sub-mesh ranges, and outward winding (signed
 * volume > 0, close to the analytic volume) on the closed meshes. Throws on the first
 * failure; prints one line per mesh and returns the measured values.
 * Run: node --input-type=module -e "import {selfCheck} from './examples/lookdev/lib/geometry.js'; selfCheck()"
 */
export function selfCheck(log = console.log) {
  const r = 0.15;
  const half = [0.7, 0.6, 0.5];
  const c = half.map((h) => h - r);
  const boxVolume = 8 * c[0] * c[1] * c[2] + 8 * r * (c[0] * c[1] + c[1] * c[2] + c[2] * c[0])
    + 2 * Math.PI * r * r * (c[0] + c[1] + c[2]) + (4 / 3) * Math.PI * r ** 3;
  const circle = Array.from({ length: 48 }, (_, i) => [Math.cos((TAU * i) / 48), 0.2, Math.sin((TAU * i) / 48)]);
  const wave = Array.from({ length: 24 }, (_, i) => [i * 0.1 - 1.2, 0.3 * Math.sin(i * 0.4), 0.1 * i]);
  const stripes = (p) => (Math.floor(p[1] * 8) % 2 ? [1, 0.2, 0.1, 1] : [1, 1, 1, 1]);
  const sphere = uvSphere(0.8, 48, 24, { color: stripes });
  const box = roundedBox(half, r, 8);
  const cases = [
    ['uvSphere', sphere, (4 / 3) * Math.PI * 0.8 ** 3],
    ['roundedBox', box, boxVolume],
    ['roundedBox sharp', roundedBox([0.5, 0.4, 0.3], 0, 1), 8 * 0.5 * 0.4 * 0.3],
    ['roundedBox full radius', roundedBox([0.3, 0.3, 0.3], 0.3, 16), (4 / 3) * Math.PI * 0.3 ** 3],
    ['lathe cylinder', lathe([[0, -1], [0.5, -1], [0.5, -1], [0.5, 1], [0.5, 1], [0, 1]], 64, { color: [0.2, 0.4, 0.9] }), Math.PI * 0.25 * 2],
    ['sweepTube loop', sweepTube(circle, 0.1, 16, { closed: true }), TAU * Math.PI * 0.01],
    ['sweepTube round caps', sweepTube(wave, 0.05, 12), null],
    ['sweepTube flat caps', sweepTube(wave, 0.05, 12, { caps: 'flat' }), null],
    ['grid', grid(2, 1, 8, 4, { color: (p) => [p[0] + 1, 0.5, 0.5, 1] }), undefined],
    ['lathe open', lathe([[0.2, 0], [0.6, 0.1], [0.7, 0.4]], 32), undefined],
  ];
  cases.push(['merge', merge([sphere, box, cases[8][1]]), undefined]);
  const results = [];
  for (const [name, desc, expected] of cases) {
    const closed = expected !== undefined;
    const m = checkMesh(name, desc, closed);
    if (typeof expected === 'number') {
      const rel = Math.abs(m.volume - expected) / expected;
      if (!(rel <= 0.03)) {
        throw new Error(`geometry selfCheck ${name}: volume ${m.volume} vs ${expected} (rel ${rel})`);
      }
    }
    results.push({ name, ...m });
    log(`selfCheck ${name}: vertices ${m.vertices}, triangles ${m.triangles}${closed ? `, signed volume ${m.volume.toFixed(5)}` : ''} | PASS`);
  }
  const merged = cases[cases.length - 1][1];
  const subs = merged.subMeshes;
  const firsts = [0, sphere.indices.length, sphere.indices.length + box.indices.length];
  if (subs.length !== 3 || subs.some((s, i) => s.first !== firsts[i])) {
    throw new Error(`geometry selfCheck merge: sub-mesh firsts ${subs.map((s) => s.first)} != ${firsts}`);
  }
  if (!merged.colors || merged.colors[4 * sphere.positions.length / 3] !== 1) {
    throw new Error('geometry selfCheck merge: parts without colours must get white');
  }
  log(`selfCheck: ${results.length} meshes PASS`);
  return results;
}
