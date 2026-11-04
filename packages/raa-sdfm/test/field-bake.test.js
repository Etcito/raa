import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SdfmError } from '../src/errors.js';
import { fieldShadingCore } from '../src/field-shading.js';
import { forbiddenToken } from '../src/map-material.js';
import {
  BAKED_DATA_VERSION, BAKED_MATERIAL_WGSL, BAKED_NO_TEXTURE, BAKE_ATLAS_BUDGET, BAKE_DILATE_PASSES, BAKE_MIN_SIZE, atlasScale, bakedMaterialWords, fitBakeSizes, createFieldBaker, fieldBakeWgsl, partTangents,
} from '../src/field-bake.js';

const FIELD = `struct PmfSample { d: f32, g: vec3f, la: u32, lb: u32, t: f32 }
fn pmf_eval(base: u32, p: vec3f) -> PmfSample { return PmfSample(length(p) - pmf_f(base + 3u), p, 0u, 1u, 0.5); }`;
const ROWS = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];
const CONSTANT = { color: [0.25, 0.5, 0.75], metallic: 1, roughness: 0.2, emissive: [0, 0, 0.5] };
const shading = {
  program: [1, 0, 8, 0.25, 0, 0, 0, 0], single: true, look: { roughness: 0.4, metallic: 0.1 },
  operands: [{ color: [1, 0, 0], material: 'flat', toNode: ROWS, constant: CONSTANT }, { color: [0, 0, 1], material: null, toNode: ROWS }],
};

// A unit quad in the z = 0 plane, uv = xy: tangent +x, bitangent +y, handedness +1; mirrored uvs flip the handedness.
const quad = {
  positions: Float32Array.of(0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0),
  normals: Float32Array.of(0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1),
  uvs: Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1),
  indices: Uint32Array.of(0, 1, 2, 0, 2, 3),
};

test('partTangents: the uv gradient, orthogonal to the normal, with its handedness', () => {
  const t = partTangents(quad.positions, quad.normals, quad.uvs, quad.indices);
  for (let v = 0; v < 4; v++) assert.deepEqual(Array.from(t.subarray(v * 4, v * 4 + 4)), [1, 0, 0, 1]);
  const mirrored = partTangents(quad.positions, quad.normals, Float32Array.of(0, 0, 1, 0, 1, -1, 0, -1), quad.indices);
  for (let v = 0; v < 4; v++) assert.equal(mirrored[v * 4 + 3], -1);
  // A vertex no triangle reaches still gets a unit tangent perpendicular to its normal.
  const lone = partTangents(Float32Array.of(0, 0, 0), Float32Array.of(0, 1, 0), Float32Array.of(0, 0), new Uint32Array(0));
  assert.ok(Math.abs(Math.hypot(lone[0], lone[1], lone[2]) - 1) < 1e-6 && Math.abs(lone[1]) < 1e-6);
  assert.throws(() => partTangents(quad.positions, quad.normals, Float32Array.of(0, 0), quad.indices), SdfmError);
});

test('atlasScale: sqrt(uv area / surface area)', () => {
  // The unit quad mapped onto a quarter of the atlas: 0.25 uv units^2 over 1 m^2.
  assert.equal(atlasScale(quad.positions, Float32Array.from(quad.uvs, (u) => u / 2), quad.indices), 0.5);
  assert.equal(atlasScale(quad.positions, new Float32Array(8), quad.indices), 0);
});

test('fitBakeSizes: the largest sides are halved first until the bakes fit the atlas budget', () => {
  const parts = [{ size: 512, textures: 2 }, { size: 128, textures: 3 }, { size: 512, textures: 2 }];
  assert.deepEqual(fitBakeSizes(parts, 1 << 30), [512, 128, 512]);
  // 2 * 512^2 * 2 + 128^2 * 3 = 1,097,728; halving the first 512 brings it to 704,512, the second to 311,296, then the
  // first 256 (the largest side, first on the tie) to 212,992.
  assert.deepEqual(fitBakeSizes(parts, 710000), [256, 128, 512]);
  assert.deepEqual(fitBakeSizes(parts, 400000), [256, 128, 256]);
  assert.deepEqual(fitBakeSizes(parts, 300000), [128, 128, 256]);
  // Never below BAKE_MIN_SIZE, even when the budget cannot be met.
  assert.deepEqual(fitBakeSizes(parts, 0), [BAKE_MIN_SIZE, BAKE_MIN_SIZE, BAKE_MIN_SIZE]);
  // With atlas scales the densest part goes first: the second 512 covers 4x the uv per metre of the first.
  const scaled = [{ size: 512, textures: 2, scale: 0.05 }, { size: 128, textures: 3, scale: 0.5 }, { size: 512, textures: 2, scale: 0.2 }];
  assert.deepEqual(fitBakeSizes(scaled, 710000), [512, 128, 256]);
  // Then 128 x 0.5 = 64 texels/m halves before 256 x 0.2 = 51.2, which halves next; 512 x 0.05 = 25.6 is kept.
  assert.deepEqual(fitBakeSizes(scaled, 600000), [512, 64, 128]);
  assert.ok(BAKE_ATLAS_BUDGET < 4096 * 4096);
});

test('bakedMaterialWords and the shared material function', () => {
  assert.deepEqual(Array.from(bakedMaterialWords(3, 4)), [BAKED_DATA_VERSION, 3, 4, BAKED_NO_TEXTURE]);
  assert.deepEqual(Array.from(bakedMaterialWords(3, 4, 5)), [BAKED_DATA_VERSION, 3, 4, 5]);
  assert.throws(() => bakedMaterialWords(-1, 4), SdfmError);
  // An S11 user function: one material_main, no binding of its own.
  assert.equal(forbiddenToken(BAKED_MATERIAL_WGSL), null);
  assert.match(BAKED_MATERIAL_WGSL, /fn material_main\(in: MaterialInput\) -> MaterialOutput/);
  assert.match(BAKED_MATERIAL_WGSL, /raa_perturb_normal\(n, in\.tangent, normalize\(ts\)\)/);
});

test('fieldBakeWgsl: the interpreter core between the data shim and the UV-space entry points', () => {
  const { core } = fieldShadingCore(shading, [], { field: FIELD });
  const wgsl = fieldBakeWgsl(core);
  for (const piece of ['fn raa_material_word(i: u32) -> u32', '@vertex fn sdfb_vertex', '@fragment fn sdfb_fragment', 'fn sdff_material(', 'fn pmf_eval(']) {
    assert.ok(wgsl.includes(piece), piece);
  }
  assert.ok(!/fn material_main/.test(wgsl), 'no RAA material entry in the bake');
  // Constant-material parts share one bake text whatever their program, operands and look.
  const other = fieldShadingCore({ ...shading, program: [1, 0, 8, 0.5, 0, 0, 0, 0], look: { roughness: 0.9, metallic: 1 } }, [], { field: FIELD });
  assert.equal(fieldBakeWgsl(other.core), wgsl);
  assert.throws(() => fieldBakeWgsl('fn nothing() {}'), SdfmError);
});

/** A recording stand-in for a GPUDevice: draws nothing, maps readback buffers filled with a row pattern. */
function fakeDevice() {
  const log = [];
  let live = 0;
  const device = {
    log,
    get live() { return live; },
    queue: { writeBuffer: (b, o, data, off, len) => log.push(['write', b.size, len]), submit: (c) => log.push(['submit', c.length]) },
    createShaderModule: ({ label }) => ({ label }),
    createRenderPipelineAsync: async (d) => { log.push(['pipeline', d.label]); return { label: d.label, getBindGroupLayout: () => ({}) }; },
    createBindGroup: (d) => ({ entries: d.entries.length }),
    createBuffer: ({ size, usage }) => {
      live++;
      let bytes = null;
      return {
        size, usage,
        destroy: () => { live--; },
        mapAsync: async () => { bytes = new Uint8Array(size); for (let i = 0; i < size; i++) bytes[i] = i % 7 === 2 ? 9 : 0; },
        getMappedRange: () => bytes.buffer,
        unmap: () => {},
      };
    },
    createTexture: ({ size }) => { live++; return { size, createView: () => ({}), destroy: () => { live--; } }; },
    createCommandEncoder: () => ({
      beginRenderPass: (d) => {
        log.push(['pass', d.colorAttachments.length]);
        return { setPipeline: () => {}, setBindGroup: () => {}, setVertexBuffer: () => {}, setIndexBuffer: () => {},
          drawIndexed: (n) => log.push(['drawIndexed', n]), draw: (n) => log.push(['draw', n]), end: () => {} };
      },
      copyTextureToBuffer: (src, dst, size) => log.push(['copy', dst.bytesPerRow, size[0]]),
      finish: () => ({}),
    }),
  };
  return device;
}

test('createFieldBaker: one raster, the dilation passes, three readbacks with row padding removed; everything freed', async () => {
  const device = fakeDevice();
  const baker = createFieldBaker(device);
  const tangents = partTangents(quad.positions, quad.normals, quad.uvs, quad.indices);
  const r = await baker.bake({ shading, materials: [], field: FIELD, ...quad, tangents, size: 32 });
  assert.equal(r.size, 32);
  assert.equal(r.normal.length, 32 * 32 * 4);
  // 32 texels * 4 bytes = 128 bytes per row, padded to 256 for the copy: each output row is the first 128 bytes.
  const row = Array.from({ length: 128 }, (_, i) => (i % 7 === 2 ? 9 : 0));
  assert.deepEqual(Array.from(r.color.subarray(128, 256)), Array.from({ length: 128 }, (_, i) => ((256 + i) % 7 === 2 ? 9 : 0)));
  assert.deepEqual(Array.from(r.color.subarray(0, 128)), row);
  assert.ok(r.emissive !== null, 'a non-black emissive readback is kept');
  assert.equal(device.log.filter((e) => e[0] === 'drawIndexed').length, 1);
  assert.equal(device.log.filter((e) => e[0] === 'draw').length, BAKE_DILATE_PASSES);
  assert.deepEqual(device.log.filter((e) => e[0] === 'copy'), [['copy', 256, 32], ['copy', 256, 32], ['copy', 256, 32]]);
  assert.equal(device.live, 0, 'buffers and textures destroyed');
  // A second constant-material part reuses the bake pipeline.
  await baker.bake({ shading: { ...shading, single: false }, materials: [], field: FIELD, ...quad, tangents, size: 16 });
  assert.equal(baker.pipelines, 1);
  await assert.rejects(baker.bake({ shading, materials: [], field: FIELD, ...quad, tangents, size: 8 }), SdfmError);
  await assert.rejects(baker.bake({ shading, materials: [], field: FIELD, ...quad, tangents: new Float32Array(4), size: 32 }), SdfmError);
});
