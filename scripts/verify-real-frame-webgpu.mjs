// Stage with Node, then run the staged page with capture-area-pilot-direct.mjs --prediction on Windows.
const files = [
  'index.html', 'verify-real-frame-webgpu.mjs', 'reproject.wgsl', 'real_model.wgsl',
  'manifest.json', 'golden.json', 'weights.fp32le', 'prediction_model.wgsl',
];

async function stage() {
  const { spawnSync } = await import('node:child_process');
  const { createHash } = await import('node:crypto');
  const { mkdir, readFile, writeFile, copyFile } = await import('node:fs/promises');
  const { dirname, join } = await import('node:path');
  const { fileURLToPath } = await import('node:url');
  const [out, weights] = process.argv.slice(2);
  if (!out || !weights) throw new Error('Usage: node scripts/verify-real-frame-webgpu.mjs <new staged page> <model package>');
  const root = dirname(dirname(fileURLToPath(import.meta.url)));
  await mkdir(out);
  const staged = [
    [fileURLToPath(import.meta.url), files[1]],
    [join(root, 'crates/raa-taa/src/wgsl/reproject.wgsl'), files[2]],
    [join(root, 'crates/raa-graph/src/exec/real_model.wgsl'), files[3]],
    ...files.slice(4, 7).map(name => [join(weights, name), name]),
    [join(root, 'crates/raa-js/src/prediction_model.wgsl'), files[7]],
  ];
  await Promise.all(staged.map(([src, name]) => copyFile(src, join(out, name))));
  await writeFile(join(out, files[0]), '<!doctype html><meta charset=utf-8><title>Real-frame WebGPU golden</title><script type=module src="./verify-real-frame-webgpu.mjs"></script>');
  const hash = createHash('sha256');
  for (const name of files) hash.update(name).update(await readFile(join(out, name)));
  const rev = spawnSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' });
  if (rev.status !== 0) throw new Error(rev.stderr);
  await writeFile(join(out, 'build.json'), JSON.stringify({
    status: 'ready', buildId: hash.digest('hex'), inputs: { revision: rev.stdout.trim() }, files,
  }));
}

// The fixture supplies already-linearized depth plus authored per-pixel masks.
// The production guide encoder converts standard-Z and obtains the masks from
// prepass/camera state before invoking the same packer. The reflection fixture
// scores the RGB residual separately from the live screen-hit admission policy.
const adapterShader = `
@group(0) @binding(18) var fixture: texture_2d<f32>;
@group(0) @binding(19) var<storage, read_write> diagnostics: array<f32>;
@compute @workgroup_size(8, 8)
fn encode_golden(@builtin(global_invocation_id) id: vec3<u32>) {
    if !inside(id.xy) { return; }
    let p = vec2<i32>(id.xy);
    let g = textureLoad(normal, p, 0);
    let mv = textureLoad(motion, p, 0);
    let f = textureLoad(fixture, p, 0);
    let packed = pack_guide(p, g, mv, f.x, jitter_px(), f.y > 0.5, f.z);
    diagnostics[26u * index(p) + 25u] = packed[7u];
    stem_at(index(p), packed);
}
@compute @workgroup_size(8, 8)
fn inspect_golden(@builtin(global_invocation_id) id: vec3<u32>) {
    if !inside(id.xy) { return; }
    let i = index(vec2<i32>(id.xy));
    for (var c = 0u; c < 12u; c++) {
        diagnostics[i * 26u + c] = guides[i].encoded[c];
        diagnostics[i * 26u + c + 12u] = guides[i].warped[c] * guides[i].gate;
    }
    diagnostics[i * 26u + 24u] = guides[i].gate;
}
@compute @workgroup_size(8, 8)
fn reflection_golden(@builtin(global_invocation_id) id: vec3<u32>) {
    if !inside(id.xy) { return; }
    let p = vec2<i32>(id.xy);
    let s = textureLoad(incident, p, 0);
    let rgb = select(vec3<f32>(0.0), reflection_residual(s.rgb, next[index(p)].hidden), s.a > 0.5);
    textureStore(incident_out, p, vec4<f32>(rgb, s.a));
}
`;

const asset = async name => {
  const response = await fetch(new URL(name, import.meta.url));
  if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
  return response;
};
const sha256 = async bytes => [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
  .map(byte => byte.toString(16).padStart(2, '0')).join('');

async function verifyGenerated(adapter, manifest, golden, weightBytes, buildId) {
  const source = await asset('prediction_model.wgsl').then(r => r.text());
  const device = await adapter.requestDevice();
  device.pushErrorScope('validation');
  const module = device.createShaderModule({ code: source.replaceAll('rgba16float', 'rgba32float') });
  const messages = (await module.getCompilationInfo()).messages.filter(({ type }) => type === 'error');
  if (messages.length) throw new Error(messages.map(({ message }) => message).join('\n'));
  const float = { texture: { sampleType: 'unfilterable-float' } };
  const read = { buffer: { type: 'read-only-storage' } };
  const write = { buffer: { type: 'storage' } };
  async function pipeline(entryPoint, bindings) {
    const entries = Object.entries(bindings).map(([binding, entry]) => ({
      binding: Number(binding), visibility: GPUShaderStage.COMPUTE, ...entry,
    }));
    const group = device.createBindGroupLayout({ entries });
    return device.createComputePipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts: [group] }),
      compute: { module, entryPoint },
    });
  }
  const [stem, residual] = await Promise.all([
    pipeline('generated_stem', { 0: read, 1: write, 2: float, 3: float, 4: float, 5: float,
      6: { texture: { sampleType: 'uint' } }, 7: { buffer: { type: 'uniform' } } }),
    pipeline('generated_residual', { 0: read, 1: write, 2: float, 3: float, 8: float,
      9: { storageTexture: { format: 'rgba32float' } } }),
  ]);
  const B = GPUBufferUsage, T = GPUTextureUsage;
  function buffer(size, usage, data) {
    const result = device.createBuffer({ size, usage: usage | (data ? B.COPY_DST : 0) });
    if (data) device.queue.writeBuffer(result, 0, data);
    return result;
  }
  const weights = buffer(weightBytes.byteLength, B.STORAGE, weightBytes);
  const vectors = [];
  for (const { id, inputs: input, outputs, sourceAgeRealFrames } of golden.vectors) {
    const [height, width] = input.warp.shape;
    const count = height * width;
    function texture(format, channels, values) {
      const size = [width, height];
      const texture = device.createTexture({ format, size,
        usage: values ? T.TEXTURE_BINDING | T.COPY_DST : T.STORAGE_BINDING | T.COPY_SRC });
      if (values) device.queue.writeTexture({ texture }, values, { bytesPerRow: width * channels * 4 }, size);
      return texture;
    }
    const warp = texture('rgba32float', 4, new Float32Array(input.warp.data));
    const fresh = texture('rgba32float', 4, new Float32Array(input.fresh.data));
    const depth = texture('r32float', 1, new Float32Array(input.targetDepth.data));
    const motion = texture('rgba32float', 4, new Float32Array(input.motion.data));
    const lighting = texture('r32uint', 1, new Uint32Array(input.lightingChanged.data));
    const base = new Float32Array(4 * count);
    for (let i = 0; i < base.length; i += 4) {
      const color = input.warp.data[i + 3] > 0.5 ? input.warp.data : input.fresh.data;
      base.set(color.slice(i, i + 4), i);
    }
    const baseline = texture('rgba32float', 4, base);
    const output = texture('rgba32float', 4);
    const features = buffer(count * 48, B.STORAGE);
    const age = buffer(4, B.UNIFORM, new Float32Array([Math.min(8, sourceAgeRealFrames) / 8]));
    const bytesPerRow = Math.ceil(width * 16 / 256) * 256;
    const mapped = buffer(bytesPerRow * height, B.COPY_DST | B.MAP_READ);
    const encoder = device.createCommandEncoder();
    function run(pipeline, bindings) {
      const entries = Object.entries(bindings).map(([binding, object]) => ({
        binding: Number(binding), resource: object instanceof GPUBuffer ? { buffer: object } : object.createView(),
      }));
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
      pass.end();
    }
    run(stem, { 0: weights, 1: features, 2: warp, 3: fresh, 4: depth, 5: motion, 6: lighting, 7: age });
    run(residual, { 0: weights, 1: features, 2: warp, 3: fresh, 8: baseline, 9: output });
    encoder.copyTextureToBuffer({ texture: output }, { buffer: mapped, bytesPerRow }, [width, height]);
    device.queue.submit([encoder.finish()]);
    await mapped.mapAsync(GPUMapMode.READ);
    const results = new Float32Array(mapped.getMappedRange());
    let maxAbs = 0, outOfTolerance = 0, freshExact = true, invalidExact = true;
    for (let i = 0; i < count; i++) {
      const index = Math.floor(i / width) * bytesPerRow / 4 + (i % width) * 4;
      for (let c = 0; c < 3; c++) {
        const actual = results[index + c];
        const expected = outputs.output.data[i * 3 + c];
        const delta = Math.abs(actual - expected);
        maxAbs = Math.max(maxAbs, delta);
        if (!Number.isFinite(delta) || delta > golden.f32Absolute + golden.f32Relative * Math.abs(expected)) outOfTolerance++;
        if (input.fresh.data[i * 4 + 3] > 0.5 && actual !== input.fresh.data[i * 4 + c]) freshExact = false;
        if (!outputs.coverage.data[i] && actual !== 0) invalidExact = false;
      }
    }
    mapped.unmap();
    vectors.push({ id, checked: count * 3, maxAbs, outOfTolerance, freshExact, invalidExact });
  }
  const error = await device.popErrorScope();
  device.destroy();
  if (error) throw error;
  const failures = vectors.filter(v => v.outOfTolerance || !v.freshExact || !v.invalidExact);
  return resultFor(adapter, manifest, golden, buildId, vectors, failures);
}

function resultFor(adapter, manifest, golden, buildId, vectors, failures) {
  const { vendor, device, description, isFallbackAdapter } = adapter.info;
  return { json: { buildId, facts: { browser: navigator.userAgent, hardware: !isFallbackAdapter,
    adapter: { vendor, device, description, isFallbackAdapter } },
  weightsSha256: manifest.weightsSha256, format: golden.format, parityPassed: failures.length === 0, failures, vectors }, files: [] };
}

async function verify({ buildId }) {
  if (!/Windows/.test(navigator.userAgent)) throw new Error('Use native Windows Chrome WebGPU');
  const [adapter, manifest, goldenBytes, weightBytes, frameShader, modelShader] = await Promise.all([
    navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }),
    asset('manifest.json').then(r => r.json()),
    asset('golden.json').then(r => r.arrayBuffer()),
    asset('weights.fp32le').then(r => r.arrayBuffer()),
    asset('reproject.wgsl').then(r => r.text()),
    asset('real_model.wgsl').then(r => r.text()),
  ]);
  if (!adapter || adapter.info.isFallbackAdapter) throw new Error('Hardware WebGPU required');
  if (await sha256(weightBytes) !== manifest.weightsSha256 || await sha256(goldenBytes) !== manifest.goldenSha256)
    throw new Error('Weight or golden hash differs from manifest');
  const golden = JSON.parse(new TextDecoder().decode(goldenBytes));
  if (manifest.format === 'raa.generated-frame-model/1') return verifyGenerated(adapter, manifest, golden, weightBytes, buildId);
  if (golden.format !== 'raa.real-frame-golden/1') throw new Error('Unknown golden format');
  const device = await adapter.requestDevice();
  device.pushErrorScope('validation');
  // Keep full precision at the fixture's output stores. The production graph
  // quantizes these two output textures to RGBA16F after FP32 inference.
  const module = device.createShaderModule({ code: frameShader + modelShader.replaceAll('rgba16float', 'rgba32float') + adapterShader });
  const messages = (await module.getCompilationInfo()).messages.filter(({ type }) => type === 'error');
  if (messages.length) throw new Error(messages.map(({ message, lineNum }) => `${lineNum}: ${message}`).join('\n'));
  const uniform = { buffer: { type: 'uniform' } };
  const read = { buffer: { type: 'read-only-storage' } };
  const storage = { buffer: { type: 'storage' } };
  const image = { texture: { sampleType: 'unfilterable-float' } };
  const array = { texture: { ...image.texture, viewDimension: '2d-array' } };
  const output = { storageTexture: { format: 'rgba32float' } };
  const arrayOutput = { storageTexture: { ...output.storageTexture, viewDimension: '2d-array' } };
  async function kernel(entryPoint, bindings) {
    const entries = Object.entries(bindings).map(([binding, type]) => ({
      binding: Number(binding), visibility: GPUShaderStage.COMPUTE, ...type,
    }));
    const group = device.createBindGroupLayout({ entries });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [group] });
    return device.createComputePipelineAsync({ label: entryPoint, layout, compute: { module, entryPoint } });
  }
  const [encode, spatial, area, recurrent, reflection, fused, admitted, head, sr, inspect] = await Promise.all([
    kernel('encode_golden', { 0: uniform, 1: read, 2: uniform, 4: image, 5: image, 6: read, 7: storage, 8: storage, 18: image, 19: storage }),
    kernel('encode_spatial', { 0: uniform, 1: read, 7: storage, 8: storage }),
    kernel('reconstruct_area', { 0: uniform, 1: read, 7: storage, 9: array, 10: arrayOutput }),
    kernel('recurrent', { 0: uniform, 1: read, 7: storage, 11: image, 12: image, 13: image, 14: storage }),
    kernel('reflection_golden', { 0: uniform, 1: read, 12: image, 14: storage, 15: output }),
    kernel('recurrent_reflection', { 0: uniform, 1: read, 7: storage, 11: image, 12: image, 13: image, 14: storage, 15: output }),
    kernel('reconstruct_reflection', { 0: uniform, 1: read, 12: image, 13: image, 14: storage, 15: output }),
    kernel('sr_head', { 0: uniform, 1: read, 14: storage, 16: storage }),
    kernel('resolve_display', { 0: uniform, 2: uniform, 11: image, 16: storage, 17: output }),
    kernel('inspect_golden', { 0: uniform, 7: storage, 19: storage }),
  ]);
  const B = GPUBufferUsage, T = GPUTextureUsage;
  function buffer(size, usage, data) {
    const result = device.createBuffer({ size, usage: usage | (data ? B.COPY_DST : 0) });
    if (data) device.queue.writeBuffer(result, 0, data);
    return result;
  }
  function texture([width, height], data, viewDimension = '2d') {
    const size = { width, height, depthOrArrayLayers: 1 };
    const object = device.createTexture({ size, format: 'rgba32float',
      usage: data ? T.TEXTURE_BINDING | T.COPY_DST : T.STORAGE_BINDING | T.COPY_SRC });
    if (data) device.queue.writeTexture({ texture: object }, data, { bytesPerRow: width * 16 }, size);
    return { texture: object, view: object.createView({ dimension: viewDimension }) };
  }
  function grid(values, channels, fn) {
    return Float32Array.from({ length: values * channels }, (_, i) => fn(Math.floor(i / channels), i % channels));
  }
  const weights = buffer(weightBytes.byteLength, B.STORAGE, weightBytes);
  let prior, previousJitter = [0, 0];
  const vectors = [];
  for (const vector of golden.vectors) {
    const { id, sourceSize, displaySize, jitterPx, reset, inputs, outputs } = vector;
    const [width, height] = sourceSize;
    const count = width * height;
    const from = name => inputs[name].data ?? inputs[name].values;
    const extent = (name, channels) => texture(sourceSize, grid(count, 4, (i, c) => c < channels ? from(name)[i * channels + c] : 0));
    if (reset || !prior) { prior = buffer(count * 64, B.STORAGE); previousJitter = jitterPx; }
    const normal = extent('normalRoughness', 4), motion = extent('motion', 3), hdr = extent('hdr', 3);
    const fixture = texture(sourceSize, grid(count, 4, (i, c) => (
      c === 0 ? from('depth')[i] : c === 1 ? Number(!reset) * from('historyValid')[i] : c === 2 ? from('reactive')[i] : 0)));
    const incident = texture(sourceSize, grid(count, 4, (i, c) => from('reflection')[i * 6 + c]));
    const meta = texture(sourceSize, grid(count, 4, (i, c) => [from('reflection')[i * 6 + 4], from('reflection')[i * 6 + 3], 0, from('reflection')[i * 6 + 5]][c]));
    const pairs = texture(sourceSize, grid(count, 4, (i, c) => from('areaPairs')[i * 2 + c % 2]), '2d-array');
    const areaOut = texture(sourceSize, null, '2d-array'), reflOut = texture(sourceSize), srOut = texture(displaySize);
    const fusedOut = texture(sourceSize), admittedOut = texture(sourceSize);
    const guides = buffer(count * 128, B.STORAGE), stem = buffer(count * 48, B.STORAGE);
    const next = buffer(count * 64, B.STORAGE | B.COPY_SRC), coeff = buffer(count * 32, B.STORAGE | B.COPY_SRC);
    const diag = buffer(count * 104, B.STORAGE | B.COPY_SRC);
    const fusedNext = buffer(count * 64, B.STORAGE | B.COPY_SRC);
    const control = buffer(32, B.UNIFORM, new Uint32Array([width, height, 0, 0, 1, 1, 3, 0]));
    const frameWords = new Float32Array(192);
    frameWords.set([jitterPx[0] * 2 / width, jitterPx[1] * -2 / height,
      previousJitter[0] * 2 / width, previousJitter[1] * -2 / height], 160);
    const camera = buffer(768, B.UNIFORM, frameWords);
    const encoder = device.createCommandEncoder({ label: id });
    const run = (pipeline, bindings, size = sourceSize) => {
      const entries = Object.entries(bindings).map(([binding, object]) => ({
        binding: Number(binding), resource: object instanceof GPUBuffer ? { buffer: object } : object.view,
      }));
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(size[0] / 8), Math.ceil(size[1] / 8));
      pass.end();
    };
    run(encode, { 0: control, 1: weights, 2: camera, 4: normal, 5: motion, 6: prior, 7: guides, 8: stem, 18: fixture, 19: diag });
    run(spatial, { 0: control, 1: weights, 7: guides, 8: stem });
    run(area, { 0: control, 1: weights, 7: guides, 9: pairs, 10: areaOut });
    run(recurrent, { 0: control, 1: weights, 7: guides, 11: hdr, 12: incident, 13: meta, 14: next });
    run(reflection, { 0: control, 1: weights, 12: incident, 14: next, 15: reflOut });
    run(admitted, { 0: control, 1: weights, 12: incident, 13: meta, 14: next, 15: admittedOut });
    run(fused, { 0: control, 1: weights, 7: guides, 11: hdr, 12: incident, 13: meta, 14: fusedNext, 15: fusedOut });
    run(head, { 0: control, 1: weights, 14: next, 16: coeff });
    run(sr, { 0: control, 2: camera, 11: hdr, 16: coeff, 17: srOut }, displaySize);
    run(inspect, { 0: control, 7: guides, 19: diag });
    function copy({ texture }, [w, h]) {
      const bytesPerRow = Math.ceil(w * 16 / 256) * 256;
      const mapped = buffer(bytesPerRow * h, B.COPY_DST | B.MAP_READ);
      encoder.copyTextureToBuffer({ texture }, { buffer: mapped, bytesPerRow }, [w, h]);
      return { mapped, stride: bytesPerRow / 4, width: w };
    }
    function copyBuffer(source) {
      const mapped = buffer(source.size, B.COPY_DST | B.MAP_READ);
      encoder.copyBufferToBuffer(source, 0, mapped, 0, source.size);
      return { mapped };
    }
    const copies = [copyBuffer(diag), copyBuffer(next), copyBuffer(coeff),
      copy(areaOut, sourceSize), copy(reflOut, sourceSize), copy(srOut, displaySize),
      copyBuffer(fusedNext), copy(fusedOut, sourceSize), copy(admittedOut, sourceSize)];
    device.queue.submit([encoder.finish()]);
    const [guide, state, coefficients, areaPairs, reflect, display, fusedState, fusedRgb, admittedRgb] = await Promise.all(copies.map(async ({ mapped, stride, width }) => {
      await mapped.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(mapped.getMappedRange()).slice();
      mapped.unmap();
      mapped.destroy();
      return stride ? (i, c) => data[Math.floor(i / width) * stride + i % width * 4 + c]
        : (i, c, channels) => data[i * channels + c];
    }));
    const values = {
      encoded: (i, c) => guide(i, c, 26),
      warped: (i, c) => guide(i, c + 12, 26),
      historyGate: i => guide(i, 24, 26),
      historyValid: i => guide(i, 25, 26),
      state: (i, c) => state(i, c, 16),
      coefficients: (i, c) => coefficients(i, c, 8),
      reflection: reflect,
      display,
      areaPairs,
      areaVisibility: i => areaPairs(i, 1) > 1e-8 ? Math.min(1, Math.max(0, areaPairs(i, 0) / areaPairs(i, 1))) : 1,
    };
    const tensors = {};
    for (const [name, tensor] of Object.entries(outputs)) {
      const expected = tensor.data ?? tensor.values;
      const channels = tensor.shape.at(-1);
      let maxAbs = 0, outOfTolerance = 0, firstFailure;
      expected.forEach((target, n) => {
        const actual = values[name](Math.floor(n / channels), n % channels);
        const delta = Math.abs(actual - target);
        maxAbs = Math.max(maxAbs, delta);
        if (!Number.isFinite(delta) || delta > golden.f32Absolute + golden.f32Relative * Math.abs(target)) {
          outOfTolerance++;
          firstFailure ??= { index: n, actual, target };
        }
      });
      tensors[name] = { maxAbs, checked: expected.length, outOfTolerance, ...(firstFailure && { firstFailure }) };
    }
    const compare = (channels, actual, expected) => {
      let maxAbs = 0, outOfTolerance = 0;
      for (let p = 0; p < count; p++) {
        for (let c = 0; c < channels; c++) {
          const a = actual(p, c, channels), b = expected(p, c, channels);
          maxAbs = Math.max(maxAbs, Math.abs(a - b));
          outOfTolerance += Number(!Number.isFinite(a) || !Number.isFinite(b) || a !== b);
        }
      }
      return { maxAbs, checked: count * channels, outOfTolerance };
    };
    tensors.areaPackedSlots = compare(2, areaPairs, (p, c) => areaPairs(p, c + 2));
    tensors.fusedState = compare(16, fusedState, state);
    tensors.fusedReflection = compare(4, fusedRgb, admittedRgb);
    tensors.screenAdmission = compare(4, fusedRgb, (p, c) => {
      const src = from('reflection');
      const accepted = src[p * 6 + 3] > 0 && src[p * 6 + 4] === 1;
      return accepted && c < 3 ? reflect(p, c) : src[p * 6 + c];
    });
    fusedNext.destroy();
    vectors.push({ id, tensors });
    prior.destroy();
    prior = next;
    previousJitter = jitterPx;
  }
  const error = await device.popErrorScope();
  if (error) throw error;
  device.destroy();
  const failures = vectors.flatMap(({ id, tensors }) => Object.entries(tensors)
    .filter(([, result]) => result.outOfTolerance).map(([tensor, result]) => ({ id, tensor, ...result })));
  return resultFor(adapter, manifest, golden, buildId, vectors, failures);
}

if (typeof window === 'undefined') await stage();
else window.nmcGpu = { ops: { prediction: verify } };
