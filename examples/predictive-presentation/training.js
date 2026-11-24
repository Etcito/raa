function decoded(bytes, half) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const step = half ? 2 : 4;
  return Float32Array.from({ length: bytes.byteLength / step }, (_, i) => {
    if (!half) return view.getFloat32(i * 4, true);
    const bits = view.getUint16(i * 2, true), exponent = (bits >>> 10) & 31, mantissa = bits & 1023;
    if (exponent === 31) throw new Error('Nonfinite retained HDR');
    return (bits & 32768 ? -1 : 1) * (exponent ? (1024 + mantissa) * 2 ** (exponent - 25) : mantissa * 2 ** -24);
  });
}

export function predictionInputs(source, guides, width, height) {
  const n = width * height;
  const sourceHdr = decoded(source.hdr, true), sourceDepth = decoded(source.depth, false);
  if (sourceHdr.length !== n * 4 || sourceDepth.length !== n) throw new Error('Unexpected retained source extent');
  const warp = new Float32Array(n * 4), fresh = new Float32Array(n * 4);
  const f32 = Math.fround;
  for (let p = 0; p < n; p++) {
    const lane = p * 4;
    const u = f32(f32((p % width + 0.5) / width) - guides.motion[lane]);
    const v = f32(f32((Math.floor(p / width) + 0.5) / height) - guides.motion[lane + 1]);
    const expected = guides.motion[lane + 3], depth = guides.depth[p];
    if (guides.motion[lane + 2] < 0.5 && u >= 0 && u < 1 && v >= 0 && v < 1
        && depth >= 0 && depth <= 1 && expected >= 0 && expected <= 1 && !guides.changed[p]) {
      const src = Math.floor(f32(u * width)) + Math.floor(f32(v * height)) * width;
      const actual = sourceDepth[src], color = sourceHdr.subarray(src * 4, src * 4 + 3);
      if ((depth === 1) === (actual === 1) && (actual === 1) === (expected === 1)
          && Math.abs(f32(actual - expected)) <= f32(0.0005)
          && color.every(value => Math.abs(value) <= 65504)) {
        warp.set(color, lane);
        warp[lane + 3] = 1;
        continue;
      }
    }
    if (guides.repair[lane + 3] !== 1 || !guides.repair.subarray(lane, lane + 3).every(value => Math.abs(value) <= 65504)) {
      throw new Error(`Causal warp rejected pixel ${p} without a valid fresh sample`);
    }
    fresh.set(guides.repair.subarray(lane, lane + 4), lane);
  }
  return { sourceHdr, sourceDepth, warp, fresh, targetDepth: guides.depth,
    motion: guides.motion, lightingChanged: Float32Array.from(guides.changed) };
}

export function predictionLabel(taps) {
  const rgba = decoded(taps.hdr, true);
  return Float32Array.from({ length: rgba.length / 4 * 3 },
    (_, i) => rgba[Math.floor(i / 3) * 4 + i % 3]);
}

export function scorePrediction(inputs, label) {
  let warpedPixels = 0, freshPixels = 0, warpedSquaredError = 0, freshMaxError = 0;
  for (let p = 0; p < label.length / 3; p++) {
    const valid = inputs.warp[p * 4 + 3] === 1;
    if (valid) warpedPixels++;
    else freshPixels++;
    const plane = valid ? inputs.warp : inputs.fresh;
    for (let c = 0; c < 3; c++) {
      const actual = plane[p * 4 + c], target = label[p * 3 + c];
      if (valid) warpedSquaredError += (Math.log1p(Math.max(0, actual)) - Math.log1p(Math.max(0, target))) ** 2;
      else freshMaxError = Math.max(freshMaxError, Math.abs(actual - target));
    }
  }
  const warpedLogRmse = Math.sqrt(warpedSquaredError / (3 * warpedPixels));
  if (!(freshPixels > 0 && freshMaxError <= 0.001 && warpedLogRmse > 0)) {
    throw new Error(`Causal fixture needs trustworthy fresh samples and a nonzero learnable residual: ${freshPixels}, ${freshMaxError}, ${warpedLogRmse}`);
  }
  return { warpedPixels, freshPixels, warpedLogRmse, freshMaxError };
}

export async function predictionPlanes(out, prefix, arrays, width, height) {
  const records = {};
  for (const [name, values] of Object.entries(arrays)) {
    const file = `${prefix}-${name.toLowerCase()}.bin`;
    const bytes = new Uint8Array(values.byteLength), view = new DataView(bytes.buffer);
    values.forEach((value, i) => view.setFloat32(4 * i, value, true));
    const hash = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
    records[name] = { file, dtype: 'f32le', shape: [height, width, values.length / (width * height)],
      byteLength: bytes.length, sha256: Array.from(hash, x => x.toString(16).padStart(2, '0')).join('') };
    out.push({ name: file, blob: new Blob([bytes], { type: 'application/octet-stream' }) });
  }
  return records;
}
