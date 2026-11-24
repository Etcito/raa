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
  const warp = new Float32Array(n * 4);
  for (let p = 0; p < n; p++) {
    const x = Math.round(p % width - guides.motion[p * 4] * width);
    const y = Math.round(Math.floor(p / width) - guides.motion[p * 4 + 1] * height);
    if (x < 0 || x >= width || y < 0 || y >= height) continue;
    const src = x + y * width;
    if (Math.abs(sourceDepth[src] - guides.motion[p * 4 + 3]) > 0.001) continue;
    warp.set(sourceHdr.subarray(src * 4, src * 4 + 3), p * 4);
    warp[p * 4 + 3] = 1;
  }
  return { sourceHdr, sourceDepth, warp, fresh: guides.repair, targetDepth: guides.depth,
    motion: guides.motion, lightingChanged: Float32Array.from(guides.changed) };
}

export function predictionLabel(taps) {
  const rgba = decoded(taps.hdr, true);
  return Float32Array.from({ length: rgba.length / 4 * 3 },
    (_, i) => rgba[Math.floor(i / 3) * 4 + i % 3]);
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
