// Procedural room content of the global-illumination page. Pure ES module (no DOM, no
// engine, no imports, importable by node). Every builder returns a fresh S13 MeshDesc:
// Float32Array positions (3n), unit normals (3n) and Uint32Array indices (3m), plus
// `subMeshes` when the mesh is split. Metres, +Y up, right-handed; triangles are
// counter-clockwise seen from outside and normals point outward.
//
// Determinism: no Math.random and no clock.

const TAU = Math.PI * 2;
const WALL = 0.4;

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
      // For +axis the in-face axes are (a+1, a+2); for -axis they swap, so cross(u, v)
      // equals the outward normal.
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

// Packs a plain accumulator into a MeshDesc.
function finish(out) {
  return {
    positions: Float32Array.from(out.positions),
    normals: Float32Array.from(out.normals),
    indices: Uint32Array.from(out.indices),
  };
}

/**
 * Closed axis-aligned box centred at `center` with half extents `half`: 24 vertices and
 * 36 indices. `center` and `half` are [x, y, z]; every face has its own outward normal.
 */
export function box(center, half) {
  const out = { positions: [], normals: [], indices: [] };
  pushBox(out, center, half);
  return finish(out);
}

/**
 * An XZ plane at y = 0 with normal +Y, centred at the origin: width along X, depth
 * along Z, four vertices and six indices.
 */
export function plane(width, depth) {
  const hw = width / 2;
  const hd = depth / 2;
  const out = { positions: [], normals: [], indices: [] };
  pushQuad(out, [[-hw, 0, -hd], [-hw, 0, hd], [hw, 0, hd], [hw, 0, -hd]], [0, 1, 0]);
  return finish(out);
}

/**
 * UV sphere of `radius`, `segments` around and `segments / 2` rings (rounded), with
 * outward unit normals and no duplicated seam. Poles are single vertices.
 */
export function sphere(radius, segments) {
  const around = Math.max(3, Math.round(segments));
  const rings = Math.max(2, Math.round(segments / 2));
  const out = { positions: [], normals: [], indices: [] };
  out.positions.push(0, -radius, 0);
  out.normals.push(0, -1, 0);
  const ringStart = [];
  for (let j = 1; j < rings; j++) {
    const phi = -Math.PI / 2 + (Math.PI * j) / rings;
    const r = radius * Math.cos(phi);
    const y = radius * Math.sin(phi);
    ringStart[j] = out.positions.length / 3;
    for (let k = 0; k < around; k++) {
      const theta = (TAU * k) / around;
      const x = r * Math.cos(theta);
      const z = r * Math.sin(theta);
      out.positions.push(x, y, z);
      out.normals.push(x / radius, y / radius, z / radius);
    }
  }
  const south = 0;
  const north = out.positions.length / 3;
  out.positions.push(0, radius, 0);
  out.normals.push(0, 1, 0);
  for (let j = 0; j < rings; j++) {
    const southBand = j === 0;
    const northBand = j === rings - 1;
    const lo = southBand ? south : ringStart[j];
    const hi = northBand ? north : ringStart[j + 1];
    for (let k = 0; k < around; k++) {
      const k1 = (k + 1) % around;
      if (southBand) {
        out.indices.push(south, hi + k, hi + k1);
      } else if (northBand) {
        out.indices.push(lo + k, north, lo + k1);
      } else {
        out.indices.push(lo + k, hi + k, hi + k1, lo + k, hi + k1, lo + k1);
      }
    }
  }
  return finish(out);
}

/**
 * A quad in the XY plane facing +Z, centred at the origin: width along X, height along
 * Y, four vertices and six indices.
 */
export function panel(width, height) {
  const hw = width / 2;
  const hh = height / 2;
  const out = { positions: [], normals: [], indices: [] };
  pushQuad(out, [[-hw, -hh, 0], [hw, -hh, 0], [hw, hh, 0], [-hw, hh, 0]], [0, 0, 1]);
  return finish(out);
}

/**
 * The enclosed 6 x 3 x 8 m room of the volume stage. The interior is x in
 * [-width/2, width/2], y in [0, height] and z in [-depth, 0]; the 0.4 m walls sit outside
 * it and the front (z = 0) is open. The mesh merges boxes into one MeshDesc whose
 * subMeshes are, in order: floor (a box under y = 0), ceiling (four boxes around a square
 * hole of side `opening` centred at (0, height, -depth/2)), back (z = -depth), left
 * (x = -width/2) and right (x = width/2).
 */
export function roomShell({ width, height, depth, opening }) {
  const out = { positions: [], normals: [], indices: [] };
  const subMeshes = [];
  const part = (fill) => {
    const first = out.indices.length;
    fill();
    subMeshes.push({ first, count: out.indices.length - first });
  };

  part(() => pushBox(out, [0, -WALL / 2, -depth / 2], [width / 2, WALL / 2, depth / 2]));
  part(() => {
    const y = height + WALL / 2;
    const hy = WALL / 2;
    const hx = (width - opening) / 4;
    const hz = (depth - opening) / 4;
    pushBox(out, [-(width + opening) / 4, y, -depth / 2], [hx, hy, depth / 2]);
    pushBox(out, [(width + opening) / 4, y, -depth / 2], [hx, hy, depth / 2]);
    pushBox(out, [0, y, (opening - depth) / 4], [opening / 2, hy, hz]);
    pushBox(out, [0, y, -(3 * depth + opening) / 4], [opening / 2, hy, hz]);
  });
  part(() => pushBox(out, [0, height / 2, -depth - WALL / 2], [width / 2, height / 2, WALL / 2]));
  part(() => pushBox(out, [-width / 2 - WALL / 2, height / 2, -depth / 2],
    [WALL / 2, height / 2, depth / 2]));
  part(() => pushBox(out, [width / 2 + WALL / 2, height / 2, -depth / 2],
    [WALL / 2, height / 2, depth / 2]));
  return { ...finish(out), subMeshes };
}

/**
 * Handoff: gi-page: scenes/volume.js calls `volumeRoom() -> object` exported by
 * examples/global-illumination/lib/content.js; the scene attaches the result as its room
 * data and derives meshes with `roomShell(room.room)`, `box(room.box.center,
 * room.box.half)`, `sphere(room.sphere.radius, room.sphere.segments)` and
 * `plane(room.ground.width, room.ground.depth)`. State/keys: the fresh room descriptor
 * (room, wall, colors, box, sphere, ground, sun, camera, elevatedCamera, subjects,
 * elevatedSubjects); no registry.
 * Handoff: gi-page: volume.test.mjs imports `volumeRoom() -> object` and the builders
 * `roomShell(room) -> MeshDesc`, `box(center, half) -> MeshDesc`, `plane(width, depth) ->
 * MeshDesc` and `sphere(radius, segments) -> MeshDesc` from
 * examples/global-illumination/lib/content.js and checks their typed arrays and index
 * ranges; state/keys: the room descriptor and MeshDesc arrays; no registry.
 *
 * The page entry (examples/global-illumination/main.js) imports nothing from here:
 * no handoff: gi-page: page entry, build script and p4-gi commands takes nothing from this card.
 *
 * Returns a fresh plain object with the S40 volume-stage room data: the 6 x 3 x 8 m
 * shell and its 1.2 m ceiling opening, the 0.4 m wall thickness, sRGB wall colours, the
 * box and sphere placements, the 200 x 200 m ground at y = -0.42, the linear sun, both
 * cameras and the subject bounds. It carries no meshes; the builders in this module
 * turn its fields into MeshDesc objects.
 */
export function volumeRoom() {
  return {
    room: { width: 6, height: 3, depth: 8, opening: 1.2 },
    wall: WALL,
    colors: {
      red: [0.8, 0.08, 0.06],
      green: [0.1, 0.65, 0.12],
      white: [0.9, 0.9, 0.88],
      ground: [0.42, 0.42, 0.4],
    },
    box: { center: [1.2, 0.5, -5.2], half: [0.5, 0.5, 0.5] },
    sphere: { center: [-1.2, 0.6, -4.6], radius: 0.6, segments: 32 },
    ground: { width: 200, depth: 200, y: -0.42 },
    sun: { direction: [0.25, -1, 0.35], color: [1, 0.96, 0.9], intensity: 3 },
    camera: { eye: [0, 1.6, -0.6], target: [0, 1.1, -8], fovYDeg: 70, near: 0.05, far: 200 },
    elevatedCamera: { eye: [6, 8, 8], target: [0, 0, -4], fovYDeg: 70, near: 0.1, far: 400 },
    subjects: [
      { name: 'room', min: [-3.4, -0.4, -8.4], max: [3.4, 3.4, 0] },
      { name: 'box', min: [0.7, 0, -5.7], max: [1.7, 1, -4.7] },
      { name: 'sphere', min: [-1.8, 0, -5.2], max: [-0.6, 1.2, -4.0] },
    ],
    elevatedSubjects: [
      { name: 'room', min: [-3.4, -0.4, -8.4], max: [3.4, 3.4, 0] },
      { name: 'ground', min: [-20, -0.42, -24], max: [20, -0.42, 16] },
    ],
  };
}
