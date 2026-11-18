// Framing check of the P5 post page (S49): whether a subject's box reaches the camera frustum.
// A pure ES module with no imports and no DOM, so node tests and the staged page (common/view-check.mjs) load the same code.
// The maths mirrors the framing code of examples/global-illumination/lib/inventory.js (`selection`); that copy is kept, not imported.

const finite = value => typeof value === 'number' && Number.isFinite(value);
const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const text = value => typeof value === 'string' && value.trim().length > 0;
const vec = value => Array.isArray(value) && value.length === 3 && value.every(finite);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = v => {
  const n = Math.hypot(...v);
  return n > 1e-12 ? v.map(x => x / n) : null;
};
const isBox = (min, max) => vec(min) && vec(max) && min.every((v, i) => v <= max[i]);
const fail = message => {
  throw new Error(message);
};
const invalidCamera = () => fail('view-check: invalid camera');

// Camera basis (forward -Z convention of the page, +Y up) and frustum half-extents; refuses an invalid camera.
function frameOf(camera) {
  if (!record(camera)) invalidCamera();
  const aspect = camera.aspect === undefined ? 16 / 9 : camera.aspect;
  if (!vec(camera.eye) || !vec(camera.target) || !finite(camera.fovYDeg) || camera.fovYDeg <= 0 || camera.fovYDeg >= 180
    || !finite(camera.near) || camera.near <= 0 || !finite(camera.far) || camera.far <= camera.near
    || !finite(aspect) || aspect <= 0) invalidCamera();
  const forward = unit(sub(camera.target, camera.eye));
  const right = forward && unit(cross(forward, [0, 1, 0]));
  if (!right) invalidCamera();
  const vertical = Math.tan(camera.fovYDeg * Math.PI / 360);
  return {
    eye: camera.eye,
    forward,
    right,
    up: cross(right, forward),
    near: camera.near,
    far: camera.far,
    vertical,
    horizontal: vertical * aspect,
  };
}

// True only when all eight corners of [min, max] lie beyond the same frustum plane, so a box larger than the frame stays in view.
function outside(frame, min, max) {
  const corners = [];
  for (let mask = 0; mask < 8; mask++) {
    const point = sub(min.map((v, i) => (mask & (1 << i) ? max[i] : v)), frame.eye);
    const x = dot(point, frame.right);
    const y = dot(point, frame.up);
    const z = dot(point, frame.forward);
    const h = z * frame.horizontal;
    const w = z * frame.vertical;
    corners.push([z - frame.near, frame.far - z, h + x, h - x, w + y, w - y]);
  }
  return corners[0].some((_, plane) => corners.every(corner => corner[plane] < 0));
}

/**
 * True when the box [min, max] reaches the camera frustum; false only when all eight corners lie beyond one frustum plane.
 * Refuses an invalid camera with `view-check: invalid camera` and a box that is not finite or has min > max with
 * `view-check: invalid box`.
 */
export function boxInView(camera, min, max) {
  const frame = frameOf(camera);
  if (!isBox(min, max)) fail('view-check: invalid box');
  return !outside(frame, min, max);
}

/**
 * Handoff: `assertSubjectsInView(selection, context) -> void` at web/common/view-check.mjs:80 is the entry point that the
 * card 'post-page: inventory and fact validation' calls. Today it refuses an invalid `selection.camera` with
 * `view-check: invalid camera`, refuses each subject without a name or with a non-finite or min > max box with
 * `<context>: invalid subject`, and throws `<context>: subject <name> outside camera frustum` for the first subject whose
 * box is out (boxInView); it returns undefined otherwise. The downstream card imports it from
 * '../../../web/common/view-check.mjs' (staged as ../common/view-check.mjs) and calls it once per variant with context
 * `<sceneId>/<variantId>`; the state it passes is the variant selection `{ camera, subjects }` only, no registry.
 */
export function assertSubjectsInView(selection, context) {
  const frame = frameOf(selection?.camera);
  const subjects = selection.subjects;
  if (!Array.isArray(subjects)) fail(`${context}: invalid subject`);
  for (const subject of subjects) {
    if (!record(subject) || !text(subject.name) || !isBox(subject.min, subject.max)) fail(`${context}: invalid subject`);
    if (outside(frame, subject.min, subject.max)) fail(`${context}: subject ${subject.name} outside camera frustum`);
  }
}
