// M10 first pixel: one box and one sphere as a deterministic TraceProgram v1
// (docs/api/m23-sdf-trace.md, crates/raa-sdf-trace/src/program.rs). Pure data, no DOM:
// the page installs it with engine.setTraceScene and scripts/trace-first-pixel.mjs
// records the same parameters next to the capture.

const MAGIC = 0x50544453; // "SDTP" little-endian
const VERSION = 1;
const OP_BOX = 1;
const OP_SPHERE = 2;
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];

export const SCENE = Object.freeze({
  box: { center: [-1.1, 0, 0], halfExtents: [0.8, 0.8, 0.8], color: [0.85, 0.25, 0.2] },
  sphere: { center: [1.1, 0, 0], radius: 0.9, color: [0.2, 0.45, 0.9] },
  camera: { target: [0, 0, 0], distance: 6.5, yaw: 0.35, pitch: 0.35 },
  light: { direction: [-0.4, -1, -0.3], color: [1, 0.96, 0.9], intensity: 3 },
  ambient: { color: [0.55, 0.6, 0.7], intensity: 0.35 },
});

const translate = ([x, y, z]) => [1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z];

// Source record: code start/len, data start/len, local bounds, lipschitz 1, flags 0.
function source(codeStart, dataStart, dataLen, half) {
  return { codeStart, dataStart, dataLen, min: half.map((h) => -h), max: half };
}

// TraceProgram v1 words: header, sources, code, data, materials, instances.
export function programWords(scene = SCENE) {
  const r = scene.sphere.radius;
  const boxData = [...IDENTITY, 1, 0, ...scene.box.halfExtents, 0];
  const sphereData = [...IDENTITY, 1, 1, r];
  const data = [...boxData, ...sphereData];
  // Instruction data offsets are relative to the source's own data slice.
  const code = [OP_BOX, OP_SPHERE];
  const sources = [
    source(0, 0, boxData.length, scene.box.halfExtents),
    source(1, boxData.length, sphereData.length, [r, r, r]),
  ];
  const materials = [scene.box.color, scene.sphere.color];
  const instances = [
    { center: scene.box.center, source: 0, material: 0, id: 1 },
    { center: scene.sphere.center, source: 1, material: 1, id: 2 },
  ];
  const f = new Float32Array(1);
  const u = new Uint32Array(f.buffer);
  const bits = (v) => { f[0] = v; return u[0]; };
  const w = [MAGIC, VERSION, sources.length, instances.length, code.length, data.length, materials.length, 0];
  for (const s of sources) {
    w.push(s.codeStart, 1, s.dataStart, s.dataLen, ...s.min.map(bits), ...s.max.map(bits), bits(1), 0);
  }
  w.push(...code, ...data.map(bits));
  for (const c of materials) w.push(...[...c, 0.6, 0, 0, 0, 0].map(bits));
  for (const i of instances) {
    w.push(...translate(i.center).map(bits), ...translate(i.center.map((v) => -v)).map(bits));
    w.push(bits(1), i.source, i.material, i.id, 0, 0, 0, 0);
  }
  return Uint32Array.from(w);
}

export function programBytes(scene = SCENE) {
  const words = programWords(scene);
  const bytes = new Uint8Array(words.length * 4);
  const view = new DataView(bytes.buffer);
  words.forEach((word, i) => view.setUint32(i * 4, word, true));
  return bytes;
}
