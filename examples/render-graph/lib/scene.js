// Room scene of the render-graph page (contract rg-room-scene). Importing this module has no
// top-level side effect and needs no DOM, engine or browser; the page builds it on a fresh
// Engine. Metres, +Y up, right-handed. Material colours are sRGB at the JS API, except
// `emissive`, which the typings declare linear (Conventions).
//
// The room: a 20 x 20 m floor whose top is y = 0 with a 2 m glossy strip along x, a white back
// wall at z = -5 and a red left wall at x = -5 (both 4 m high, 0.2 m thick), three boxes, one
// metal sphere and an emissive panel on the back wall, lit by an area light, a point light and
// the 35-degree sun under the physical day sky.

export const RG_DT_MS = 1000 / 60;

// RG_RUN: 240 warm-up frames at RG_DT_MS each, then the captured frame.
export const RG_WARMUP_FRAMES = 240;

// The camera time of the captured frame 240: 240 x RG_DT_MS / 1000 = 4 s.
export const RG_CAPTURE_T_S = 4;

// ROOM_SUBJECTS boxes each visible subject in metres, in scene space.
export const ROOM_SUBJECTS = [
  { name: 'floor', min: [-10, 0, -10], max: [10, 0, 10] },
  { name: 'back-wall', min: [-5, 0, -5.1], max: [10, 4, -4.9] },
  { name: 'left-wall', min: [-5.1, 0, -5], max: [-4.9, 4, 10] },
  { name: 'box-a', min: [0.2, 0, -2.8], max: [1.0, 0.8, -2.0] },
  { name: 'box-b', min: [-2.2, 0, -3.6], max: [-1.0, 1.2, -2.4] },
  { name: 'box-c', min: [1.9, 0, -0.9], max: [2.5, 0.6, -0.3] },
  { name: 'sphere', min: [-1.3, 0, -1.7], max: [0.1, 1.4, -0.3] },
  { name: 'emissive-panel', min: [1.85, 2.0, -4.9], max: [3.35, 2.8, -4.88] },
];

/**
 * `cameraAt(t)` is the RG_CAMERA_PATH pose at `t` seconds: an orbit about (0, 1, 0) at radius
 * 9 m, height 3 m, azimuth a = 0.6 + 0.1 t rad (0 = +Z), vertical fov 50 degrees. Today it
 * returns `{ position: [9 sin a, 3, 9 cos a], target: [0, 1, 0], fovDeg: 50 }`, `applyCamera`
 * sets that camera with near 0.1 m and far 200 m, and `selectionAt` returns the same camera
 * with aspect 16/9 and ROOM_SUBJECTS. `buildRoom` returns the built scene's subjects.
 * Handoff: render-graph page 2/5: RG_RUN page runtime with pair images and caption strip takes
 * `cameraAt(t: number) -> { position: number[3], target: number[3], fovDeg: number }`,
 * `applyCamera(engine: Engine, t: number) -> void` exported by
 * examples/render-graph/lib/scene.js:56 cameraAt and `validateRequiredFacts(row: object, facts:
 * object, factNames: string[]) -> void` exported by examples/render-graph/lib/inventory.js:137
 * validateRequiredFacts; the card attaches them by ES import in lib/runtime.js and passes
 * validateRequiredFacts to createPageRunner. State passed: the rg-page-facts shape and the
 * constants RG_DT_MS, RG_WARMUP_FRAMES and RG_CAPTURE_T_S.
 * Handoff: render-graph page 3/5: interpreter scene and its test takes
 * `buildRoom(engine: Engine) -> Promise<{ subjects: { name: string, min: number[3], max: number[3] }[] }>`
 * and `selectionAt(t: number) -> { camera: { eye: number[3], target: number[3], fovYDeg: number,
 * near: number, far: number, aspect: number }, subjects: object[] }` exported by
 * examples/render-graph/lib/scene.js:112 buildRoom and :68 selectionAt, plus
 * `validateInventory(scenes: object[], stage: string) -> object[]` exported by
 * examples/render-graph/lib/inventory.js:92 validateInventory and the validateRequiredFacts
 * above; its scene's build calls buildRoom, its variants use selectionAt(RG_CAPTURE_T_S) and
 * its tests call both validators. State passed: ROOM_SUBJECTS, the rg-page-facts shape and the
 * stage list.
 */
export function cameraAt(t) {
  const a = 0.6 + 0.1 * t;
  return { position: [9 * Math.sin(a), 3, 9 * Math.cos(a)], target: [0, 1, 0], fovDeg: 50 };
}

/** Sets the engine camera from `cameraAt(t)`; near 0.1 m, far 200 m. Returns undefined. */
export function applyCamera(engine, t) {
  const { position, target, fovDeg } = cameraAt(t);
  engine.setCamera({ eye: position, target, fovYDeg: fovDeg, near: 0.1, far: 200 });
}

/** The rgba-free selection of {@link validateInventory}: camera at `t` and ROOM_SUBJECTS. */
export function selectionAt(t) {
  const { position, target, fovDeg } = cameraAt(t);
  return {
    camera: { eye: position, target, fovYDeg: fovDeg, near: 0.1, far: 200, aspect: 16 / 9 },
    subjects: ROOM_SUBJECTS,
  };
}

// Linear 0.5 grey reaches the sRGB JS material API as this value.
const linearToSrgb = value => value <= 0.0031308 ? 12.92 * value : 1.055 * value ** (1 / 2.4) - 0.055;
const nodeAt = (engine, position = [0, 0, 0]) => engine.addNode(null, position, [0, 0, 0, 1], [1, 1, 1]);
const meshWith = (engine, desc, material) => engine.addMesh({
  ...desc,
  subMeshes: [{ first: 0, count: desc.indices.length, material }],
});
const instance = (engine, desc, material, position) =>
  engine.addMeshInstance(nodeAt(engine, position), meshWith(engine, desc, material));
const plane = (engine, material, x0, x1, z0, z1, y = 0) => instance(engine, {
  positions: Float32Array.from([x0, y, z0, x0, y, z1, x1, y, z1, x1, y, z0]),
  normals: Float32Array.from([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
  indices: Uint32Array.from([0, 1, 2, 0, 2, 3]),
}, material);
const panel = (engine, material, width, height, position) => {
  const hw = width / 2, hh = height / 2;
  return instance(engine, {
    positions: Float32Array.from([-hw, -hh, 0, hw, -hh, 0, hw, hh, 0, -hw, hh, 0]),
    normals: Float32Array.from([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
    indices: Uint32Array.from([0, 1, 2, 0, 2, 3]),
  }, material, position);
};
const primitive = (engine, kind, params, material, position, segments) =>
  engine.addInstance(nodeAt(engine, position), engine.primitiveMesh(kind, params, segments), material);

/**
 * `buildRoom(engine)` builds contract rg-room-scene on `engine` and resolves to
 * `{ subjects: ROOM_SUBJECTS }`. Today it sets the physical `day` sky, the sun at 35 degrees
 * elevation and 230 degrees azimuth (travel toward the camera side, so box shadows lie on the
 * visible floor), a low ambient, one 2 x 1 m area light of intensity 6 above the room and one
 * point light of intensity 4 and range 8 m; it adds the 20 x 20 m floor (linear grey 0.5,
 * roughness 0.8, glossy roughness 0.15 strip for z in [-1, 1]), the white back wall at z = -5,
 * the red left wall at x = -5 (both 4 m high and 0.2 m thick), three boxes of 0.8, 1.2 and 0.6 m,
 * a metallic sphere of radius 0.7 and the 1.5 x 0.8 m emissive panel (linear 4, 3, 1.5) on the
 * back wall.
 */
export async function buildRoom(engine) {
  engine.setSky({ model: 'physical', preset: 'day' });
  const elevation = 35 * Math.PI / 180, azimuth = 230 * Math.PI / 180;
  engine.setLight({
    // Travel direction of the sun: elevation 35 degrees, azimuth 230 degrees from +Z.
    direction: [-Math.sin(azimuth) * Math.cos(elevation), -Math.sin(elevation), -Math.cos(azimuth) * Math.cos(elevation)],
    color: [1, 0.96, 0.9],
    intensity: 3,
  });
  engine.setAmbient({ color: [0.55, 0.62, 0.75], intensity: 0.05 });
  engine.addLight({
    kind: 'area', position: [0, 5, -1], direction: [0, -1, 0], size: [2, 1],
    intensity: 6, color: [1, 0.97, 0.92],
  });
  engine.addLight({ kind: 'point', position: [-2.2, 2.2, -2.2], intensity: 4, range: 8, color: [1, 0.87, 0.72] });
  const matte = (color, roughness = 0.8) => engine.addMaterial({ color, roughness });
  const grey = linearToSrgb(0.5);
  const floor = matte([grey, grey, grey], 0.8), glossy = matte([grey, grey, grey], 0.15);
  const white = matte([0.9, 0.9, 0.88]), red = matte([0.6, 0.1, 0.1]);
  plane(engine, floor, -10, 10, -10, -1);
  plane(engine, glossy, -10, 10, -1, 1);
  plane(engine, floor, -10, 10, 1, 10);
  primitive(engine, 'box', { halfExtents: [7.5, 2, 0.1] }, white, [2.5, 2, -5]);
  primitive(engine, 'box', { halfExtents: [0.1, 2, 7.5] }, red, [-5, 2, 2.5]);
  const boxColors = [[0.85, 0.62, 0.28], [0.28, 0.5, 0.78], [0.42, 0.68, 0.4]];
  [[0.6, 0.4, -2.4], [-1.6, 0.6, -3.0], [2.2, 0.3, -0.6]].forEach((position, index) => {
    const half = [0.4, 0.6, 0.3][index];
    primitive(engine, 'box', { halfExtents: [half, half, half] }, matte(boxColors[index], 0.6), position);
  });
  primitive(engine, 'sphere', { radius: 0.7 },
    engine.addMaterial({ color: [0.9, 0.9, 0.95], roughness: 0.2, metallic: 1 }), [-0.6, 0.7, -1.0], 48);
  const glow = engine.addMaterial({ color: [0.95, 0.9, 0.8], roughness: 0.5, emissive: [4, 3, 1.5] });
  panel(engine, glow, 1.5, 0.8, [2.6, 2.4, -4.89]);
  return { subjects: ROOM_SUBJECTS };
}
