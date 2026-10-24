// MAMA reference viewport of the compare page (left pane).
// MAMA 0.1.0 @ 864ebc3 has no browser viewport host (only setupBrowserHost, a meshing pool), so this
// pane draws the meshes MAMA itself produces: the document is meshed by MAMA's meshAssets on the
// setupBrowserHost pool (through the same bridge as @etcito/raa-sdfm), each vertex gets the base colour
// MAMA's CPU material evaluator (cpuGraphOrNull) gives it, and plain WebGPU draws them with Lambert
// shading, the RAA pane's camera and size. Every MAMA call has a row in packages/raa-sdfm/README.md
// (## MAMA calls).
import { importMama, createBrowserPool, meshDocument } from '../../../packages/raa-sdfm/src/mama-bridge.js';
import { SdfmError } from '../../../packages/raa-sdfm/src/errors.js';

export const MAMA_CHECKED_COMMIT = '864ebc3';
export const PANE_LABEL = `MAMA ${MAMA_CHECKED_COMMIT} mesher (reference viewport)`;

/** Linear colour of a sub-mesh without material, or of a material MAMA treats as "no material". */
export const NO_MATERIAL_COLOR = [0.5, 0.5, 0.5];
/** Display-referred linear background (the RAA pane's background colour). */
export const DEFAULT_BACKGROUND = [0.3, 0.34, 0.4];
/** Lambert terms: colour = base * (AMBIENT + DIFFUSE * max(n . l, 0)). */
const AMBIENT = 0.3;
const DIFFUSE = 0.7;
/** Key light in camera space (right, up, back): above, left of and in front of the camera. */
const LIGHT_VIEW = [-0.35, 0.6, 0.7];
const MSAA = 4;
const VERTEX_FLOATS = 9; // position, normal, linear base colour
const UNIFORM_BYTES = 96;
// Surface masks of MAMA's CPU evaluator at a point without SDF context: neutral values.
const NEUTRAL_MASKS = Object.freeze({ curvature: 0, cavity: 0, ao: 1, height: 0, proximity: 0 });

const SHADER = /* wgsl */ `
struct Frame { view_proj: mat4x4<f32>, light: vec4<f32>, params: vec4<f32> };
@group(0) @binding(0) var<uniform> frame: Frame;
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) n: vec3<f32>, @location(1) c: vec3<f32> };
@vertex fn vs(@location(0) p: vec3<f32>, @location(1) n: vec3<f32>, @location(2) c: vec3<f32>) -> VOut {
  var o: VOut;
  o.pos = frame.view_proj * vec4<f32>(p, 1.0);
  o.n = n;
  o.c = c;
  return o;
}
fn oetf(x: vec3<f32>) -> vec3<f32> {
  let lo = x * 12.92;
  let hi = 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055;
  return select(hi, lo, x <= vec3<f32>(0.0031308));
}
@fragment fn fs(i: VOut) -> @location(0) vec4<f32> {
  let n = normalize(i.n);
  let lambert = max(dot(n, frame.light.xyz), 0.0);
  var rgb = i.c * (frame.params.x + frame.params.y * lambert);
  if (frame.params.z > 0.5) { rgb = oetf(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0))); }
  return vec4<f32>(rgb, 1.0);
}
`;

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
function normalize(v) {
  const l = Math.hypot(v[0], v[1], v[2]);
  return l > 0 ? [v[0] / l, v[1] / l, v[2] / l] : [0, 0, 0];
}

/** sRGB OETF of one linear channel in [0, 1]. */
export function linearToSrgb(x) {
  const c = Math.min(1, Math.max(0, x));
  return c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055;
}

/**
 * The camera basis and column-major view-projection (right-handed, Y up, clip depth [0, 1]).
 * @param {{ position: number[], target: number[], up?: number[], fovYDeg: number, near: number, far: number }} camera
 * @param {number} aspect width / height
 * @returns {{ viewProj: Float32Array, right: number[], up: number[], back: number[] }}
 */
export function cameraMatrices(camera, aspect) {
  const eye = camera.position;
  const back = normalize(sub(eye, camera.target));
  let right = normalize(cross(camera.up ?? [0, 1, 0], back));
  if (dot(right, right) === 0) right = normalize(cross([0, 0, 1], back));
  const up = cross(back, right);
  const view = [
    right[0], up[0], back[0], 0,
    right[1], up[1], back[1], 0,
    right[2], up[2], back[2], 0,
    -dot(right, eye), -dot(up, eye), -dot(back, eye), 1,
  ];
  const f = 1 / Math.tan((camera.fovYDeg * Math.PI) / 360);
  const { near, far } = camera;
  const proj = [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far / (near - far), -1, 0, 0, (near * far) / (near - far), 0];
  const viewProj = new Float32Array(16);
  for (let c = 0; c < 4; c += 1) {
    for (let r = 0; r < 4; r += 1) {
      let s = 0;
      for (let k = 0; k < 4; k += 1) s += proj[k * 4 + r] * view[c * 4 + k];
      viewProj[c * 4 + r] = s;
    }
  }
  return { viewProj, right, up, back };
}

/** The upper 3x3 inverse-transpose of a column-major 4x4 (normal matrix), column-major 3x3. */
function normalMatrix(m) {
  const a = [m[0], m[1], m[2]];
  const b = [m[4], m[5], m[6]];
  const c = [m[8], m[9], m[10]];
  const bc = cross(b, c);
  const det = dot(a, bc);
  if (!(Math.abs(det) > 0)) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const r0 = bc;
  const r1 = cross(c, a);
  const r2 = cross(a, b);
  // inverse rows are r0/det, r1/det, r2/det; its transpose has them as columns
  return [...r0, ...r1, ...r2].map((x) => x / det);
}

/**
 * One world-space vertex stream of every MAMA mesh: positions and normals placed by the part's
 * MAMA frame, and per vertex the linear base colour MAMA's CPU material evaluator returns there.
 *
 * @param {{ groups: { parts: { frame: ArrayLike<number>, meshes: { material: string | null,
 *   positions: Float32Array, normals: Float32Array, indices: Uint32Array }[] }[] }[] }} meshed meshDocument output
 * @param {{ materials: Record<string, unknown> }} doc MAMA's SdfDocument (readSdfm)
 * @param {(def: unknown, doc: unknown) => { eval: Function } | null} cpuGraphOrNull MAMA's CPU material graph
 * @returns {{ vertices: Float32Array, indices: Uint32Array, triangles: number }}
 */
export function bakeReferenceMesh(meshed, doc, cpuGraphOrNull) {
  const meshes = meshed.groups.flatMap((g) => g.parts.flatMap((p) => p.meshes.map((m) => ({ frame: p.frame, m }))));
  const vertexCount = meshes.reduce((n, { m }) => n + m.positions.length / 3, 0);
  const indexCount = meshes.reduce((n, { m }) => n + m.indices.length, 0);
  const vertices = new Float32Array(vertexCount * VERTEX_FLOATS);
  const indices = new Uint32Array(indexCount);
  const graphs = new Map();
  const graphOf = (id) => {
    if (id === null) return null;
    if (!graphs.has(id)) {
      const def = doc.materials ? doc.materials[id] : undefined;
      if (def === undefined) {
        throw new SdfmError('mama_view.material', `material '${id}' of a MAMA mesh is not in the document`);
      }
      try {
        graphs.set(id, cpuGraphOrNull(def, doc));
      } catch (e) {
        throw new SdfmError('mama_view.material', `MAMA cannot evaluate material '${id}' on the CPU: ${messageOf(e)}`);
      }
    }
    return graphs.get(id);
  };
  let base = 0;
  let first = 0;
  for (const { frame: f, m } of meshes) {
    const nm = normalMatrix(f);
    const graph = graphOf(m.material);
    const n = m.positions.length / 3;
    for (let v = 0; v < n; v += 1) {
      const lx = m.positions[v * 3], ly = m.positions[v * 3 + 1], lz = m.positions[v * 3 + 2];
      const p = [
        f[0] * lx + f[4] * ly + f[8] * lz + f[12],
        f[1] * lx + f[5] * ly + f[9] * lz + f[13],
        f[2] * lx + f[6] * ly + f[10] * lz + f[14],
      ];
      const nx = m.normals[v * 3], ny = m.normals[v * 3 + 1], nz = m.normals[v * 3 + 2];
      const wn = normalize([
        nm[0] * nx + nm[3] * ny + nm[6] * nz,
        nm[1] * nx + nm[4] * ny + nm[7] * nz,
        nm[2] * nx + nm[5] * ny + nm[8] * nz,
      ]);
      let color = NO_MATERIAL_COLOR;
      if (graph) {
        try {
          color = graph.eval(p, p, wn, () => NEUTRAL_MASKS, {}).baseColor;
        } catch (e) {
          throw new SdfmError('mama_view.material', `MAMA material '${m.material}' failed on the CPU: ${messageOf(e)}`);
        }
      }
      const o = (base + v) * VERTEX_FLOATS;
      vertices.set(p, o);
      vertices.set(wn, o + 3);
      vertices.set([color[0], color[1], color[2]], o + 6);
    }
    for (let i = 0; i < m.indices.length; i += 1) indices[first + i] = m.indices[i] + base;
    base += n;
    first += m.indices.length;
  }
  if (!vertices.every(Number.isFinite)) {
    throw new SdfmError('mama_view.not_finite', 'a MAMA mesh or base colour has a non-finite value');
  }
  return { vertices, indices, triangles: indexCount / 3 };
}

/** MAMA modules of the page: the bridge exports plus the CPU material evaluator. */
async function importPageMama(mamaUrl) {
  const mama = await importMama(mamaUrl);
  const href = new URL('core/material-graph/cpu-eval.js', mamaUrl).href;
  let cpu;
  try {
    cpu = await import(href);
  } catch (e) {
    throw new SdfmError('sdfm.worker', `cannot import MAMA module ${href}: ${messageOf(e)}`);
  }
  return { mama, cpuGraphOrNull: cpu.cpuGraphOrNull };
}

/** Meshes `document` with MAMA on the setupBrowserHost pool (classic blob: workers: opaque origin). */
async function meshWithBrowserHost(mama, mamaUrl, document) {
  const pool = createBrowserPool(mama, {
    workerUrl: new URL('browser/compute-worker.js', mamaUrl).href,
    manifoldWasmUrl: new URL('manifold.wasm', mamaUrl).href,
    workerMode: 'blob',
  });
  try {
    return await meshDocument(mama, document, { pool });
  } finally {
    pool.dispose();
  }
}

/**
 * Creates the reference viewport on `canvas` (a core WebGPU adapter, no compatibility level).
 * API: `{ load(asset), setCamera({ position, target, up?, fovYDeg, near, far }), render(),
 * probeNonEmpty(), dispose() }`; render() draws one frame synchronously (no rAF loop), so the page
 * reads the canvas in the same task.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {{ mamaUrl: string, background?: number[], gpu?: GPU | null,
 *   modules?: { mama: object, cpuGraphOrNull: Function } }} opts mamaUrl: URL of the staged
 *   `mama/index.js`; background: display-referred linear rgb; gpu/modules: injected by tests
 * @returns {Promise<object>}
 */
export async function createMamaView(canvas, opts) {
  const { mamaUrl, background = DEFAULT_BACKGROUND } = opts ?? {};
  if (typeof mamaUrl !== 'string' || mamaUrl === '') {
    throw new SdfmError('mama_view.options', 'createMamaView: opts.mamaUrl (URL of the staged mama/index.js) is required');
  }
  const gpu = opts.gpu !== undefined ? opts.gpu : globalThis.navigator?.gpu;
  if (!gpu || typeof gpu.requestAdapter !== 'function') {
    throw new SdfmError('mama_view.no_webgpu', 'the MAMA reference viewport needs WebGPU (navigator.gpu is missing)');
  }
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new SdfmError('mama_view.no_webgpu', 'the MAMA reference viewport got no core WebGPU adapter');
  const device = await adapter.requestDevice();
  const context = canvas.getContext('webgpu');
  if (!context) throw new SdfmError('mama_view.no_webgpu', 'the MAMA reference viewport canvas has no webgpu context');
  const format = gpu.getPreferredCanvasFormat();
  const encodeSrgb = !format.endsWith('-srgb');
  context.configure({
    device,
    format,
    alphaMode: 'opaque',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const clearRgb = background.map((c) => (encodeSrgb ? linearToSrgb(c) : c));
  const clearValue = { r: clearRgb[0], g: clearRgb[1], b: clearRgb[2], a: 1 };

  const module = device.createShaderModule({ code: SHADER });
  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module,
      entryPoint: 'vs',
      buffers: [{
        arrayStride: VERTEX_FLOATS * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x3' },
          { shaderLocation: 1, offset: 12, format: 'float32x3' },
          { shaderLocation: 2, offset: 24, format: 'float32x3' },
        ],
      }],
    },
    fragment: { module, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list', frontFace: 'ccw', cullMode: 'back' },
    depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    multisample: { count: MSAA },
  });
  const uniform = device.createBuffer({ size: UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: uniform } }] });

  let modules = opts.modules ?? null;
  let mesh = null; // { vertexBuffer, indexBuffer, indexCount, triangles }
  let camera = null;
  let targets = null; // { width, height, color, depth }
  let disposed = false;

  const live = () => {
    if (disposed) throw new SdfmError('sdfm.disposed', 'the MAMA reference viewport was disposed');
  };

  function sizedTargets() {
    const width = canvas.width;
    const height = canvas.height;
    if (targets && targets.width === width && targets.height === height) return targets;
    if (targets) {
      targets.color.destroy();
      targets.depth.destroy();
    }
    const size = { width, height };
    const usage = GPUTextureUsage.RENDER_ATTACHMENT;
    targets = {
      width,
      height,
      color: device.createTexture({ size, format, sampleCount: MSAA, usage }),
      depth: device.createTexture({ size, format: 'depth32float', sampleCount: MSAA, usage }),
    };
    return targets;
  }

  function writeFrame(width, height) {
    const { viewProj, right, up, back } = cameraMatrices(camera, width / height);
    const l = normalize([0, 1, 2].map((i) => LIGHT_VIEW[0] * right[i] + LIGHT_VIEW[1] * up[i] + LIGHT_VIEW[2] * back[i]));
    const data = new Float32Array(UNIFORM_BYTES / 4);
    data.set(viewProj, 0);
    data.set([l[0], l[1], l[2], 0], 16);
    data.set([AMBIENT, DIFFUSE, encodeSrgb ? 1 : 0, 0], 20);
    device.queue.writeBuffer(uniform, 0, data);
  }

  /** Encodes one frame into the canvas texture; returns it (for the probe copy) and the encoder. */
  function encodeFrame() {
    live();
    if (!mesh) throw new SdfmError('mama_view.not_loaded', 'render: no MAMA document is loaded');
    if (!camera) throw new SdfmError('mama_view.no_camera', 'render: setCamera was not called');
    const t = sizedTargets();
    writeFrame(t.width, t.height);
    const texture = context.getCurrentTexture();
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: t.color.createView(),
        resolveTarget: texture.createView(),
        clearValue,
        loadOp: 'clear',
        storeOp: 'discard',
      }],
      depthStencilAttachment: { view: t.depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, mesh.vertexBuffer);
    pass.setIndexBuffer(mesh.indexBuffer, 'uint32');
    pass.drawIndexed(mesh.indexCount);
    pass.end();
    return { texture, encoder, width: t.width, height: t.height };
  }

  return {
    label: PANE_LABEL,
    format,

    /** Meshes the asset's `document` with MAMA and uploads one vertex and index buffer. */
    async load(asset) {
      live();
      const document = asset && typeof asset === 'object' ? asset.document : undefined;
      if (!document || typeof document !== 'object') {
        throw new SdfmError('mama_view.document', 'load: the asset has no sdfm document', '/document');
      }
      modules ??= await importPageMama(mamaUrl);
      const meshed = await meshWithBrowserHost(modules.mama, mamaUrl, document);
      let doc;
      try {
        doc = modules.mama.readSdfm(document);
      } catch (e) {
        throw new SdfmError('sdfm.worker', `MAMA readSdfm failed: ${messageOf(e)}`);
      }
      const baked = bakeReferenceMesh(meshed, doc, modules.cpuGraphOrNull);
      if (baked.triangles === 0) throw new SdfmError('mama_view.empty', 'MAMA produced no triangle for this document');
      live();
      if (mesh) {
        mesh.vertexBuffer.destroy();
        mesh.indexBuffer.destroy();
      }
      const upload = (data, usage) => {
        const buffer = device.createBuffer({ size: Math.max(4, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(buffer, 0, data);
        return buffer;
      };
      mesh = {
        vertexBuffer: upload(baked.vertices, GPUBufferUsage.VERTEX),
        indexBuffer: upload(baked.indices, GPUBufferUsage.INDEX),
        indexCount: baked.indices.length,
        triangles: baked.triangles,
      };
      return { triangles: baked.triangles };
    },

    setCamera(desc) {
      live();
      const v3 = (v) => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite);
      if (!desc || !v3(desc.position) || !v3(desc.target) || !(desc.fovYDeg > 0 && desc.fovYDeg < 180)
        || !(desc.near > 0) || !(desc.far > desc.near) || (desc.up !== undefined && !v3(desc.up))) {
        throw new SdfmError('mama_view.camera', 'setCamera: position, target, up?, fovYDeg in (0, 180), 0 < near < far required');
      }
      camera = { position: desc.position, target: desc.target, up: desc.up ?? [0, 1, 0],
        fovYDeg: desc.fovYDeg, near: desc.near, far: desc.far };
    },

    /** Draws one frame into the canvas (synchronous encode and submit). */
    render() {
      const { encoder } = encodeFrame();
      device.queue.submit([encoder.finish()]);
    },

    /** Renders one frame and reads it back: true when a pixel differs from the background. */
    async probeNonEmpty() {
      if (!/^(bgra8unorm|rgba8unorm)(-srgb)?$/.test(format)) {
        throw new SdfmError('mama_view.format', `probeNonEmpty: canvas format ${format} cannot be read back as 8-bit`);
      }
      const { texture, encoder, width, height } = encodeFrame();
      const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
      const readback = device.createBuffer({ size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      encoder.copyTextureToBuffer({ texture }, { buffer: readback, bytesPerRow }, { width, height });
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const px = new Uint8Array(readback.getMappedRange());
      const bg = background.map((c) => Math.round(linearToSrgb(c) * 255));
      const order = format.startsWith('bgra') ? [2, 1, 0] : [0, 1, 2];
      let differing = 0;
      for (let y = 0; y < height && differing === 0; y += 1) {
        for (let x = 0; x < width; x += 1) {
          const o = y * bytesPerRow + x * 4;
          if (order.some((ch, i) => Math.abs(px[o + ch] - bg[i]) > 2)) {
            differing += 1;
            break;
          }
        }
      }
      readback.unmap();
      readback.destroy();
      return differing > 0;
    },

    dispose() {
      if (disposed) return;
      disposed = true;
      if (mesh) {
        mesh.vertexBuffer.destroy();
        mesh.indexBuffer.destroy();
        mesh = null;
      }
      if (targets) {
        targets.color.destroy();
        targets.depth.destroy();
        targets = null;
      }
      uniform.destroy();
      try {
        context.unconfigure();
      } catch {
        // an unconfigured context has nothing to release
      }
      device.destroy();
    },
  };
}

function messageOf(e) {
  return e instanceof Error ? e.message : String(e);
}
