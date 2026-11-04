import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mapMaterial } from '../src/map-material.js';
import { SdfmError } from '../src/errors.js';
import {
  checkNormalVolume, orthoTangent, volumeInstanceParams, volumeMaterial, volumeStandardMaterial, volumeSurfaceAttributes,
  volumeTexture,
} from '../src/normal-volume.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}.json`, import.meta.url), 'utf8'));

/** A well-formed one-brick volume (MAMA normal-volume.ts layout: 25-wide tiles, 10 rows per tile). */
function volume(over = {}) {
  const width = 25, height = 1 + 10;
  return {
    format: 'sdfm.normal-volume', version: 1, origin: [-1, -1, -1], spacing: 0.5, bricks: [1, 1, 1], count: 1,
    width, height, tilesPerRow: 1, indexRows: 1, texels: new Uint8Array(width * height * 4), ...over,
  };
}

test('volumeMaterial renames the base material_main and defines one wrapper', () => {
  const base = mapMaterial(fixture('compiled-unmappable'));
  const w = volumeMaterial(base.functionName, base.wgsl);
  assert.equal(w.functionName, 'sdfv.' + base.functionName);
  assert.equal(w.wgsl.match(/\bfn\s+material_main\s*\(/g).length, 1);
  assert.equal(w.wgsl.match(/\bfn\s+sdfv_base_main\s*\(/g).length, 1);
  assert.ok(w.wgsl.includes('o.normal = n;'));
  assert.ok(w.wgsl.includes('b.color = vec4<f32>(1.0);'), 'the base never sees the rest position as a colour');
  assert.ok(!/\bfn\s+raa_/.test(w.wgsl), 'S11: no identifier starting raa_ is defined');
  assert.ok(new TextEncoder().encode(w.wgsl).length <= 65536);
});

test('volumeMaterial refuses a base without exactly one material_main, or one already wrapped', () => {
  const refuse = (wgsl) => assert.throws(() => volumeMaterial('x', wgsl), (e) => e instanceof SdfmError && e.code === 'sdfm.material_unmappable');
  refuse('fn other(in: MaterialInput) -> MaterialOutput { return raa_default_output(in); }');
  const once = volumeStandardMaterial().wgsl;
  refuse(once);
  refuse(once + '\nfn material_main(in: MaterialInput) -> MaterialOutput { return raa_default_output(in); }');
  assert.equal(volumeStandardMaterial().functionName, 'sdfv.standard');
});

test('volumeSurfaceAttributes: rest position in the colour, a rest frame the WGSL rebuilds', () => {
  const positions = new Float32Array([0.5, -2, 3, 1, 1, 1, -4, 0, 0.25]);
  const raw = [[0, 1, 0], [0.3, -0.2, -0.9], [0, 0, -1]];
  const normals = new Float32Array(raw.flat());
  const { colors, uvs, tangents } = volumeSurfaceAttributes(positions, normals);
  for (let v = 0; v < 3; v++) {
    assert.deepEqual(Array.from(colors.subarray(v * 4, v * 4 + 4)), [...positions.subarray(v * 3, v * 3 + 3), 1]);
    // octahedral decode (sdfv_oct_decode) of the uv gives the unit rest normal back
    const [x, y] = uvs.subarray(v * 2, v * 2 + 2);
    let n = [x, y, 1 - Math.abs(x) - Math.abs(y)];
    if (n[2] < 0) n = [(1 - Math.abs(n[1])) * (n[0] >= 0 ? 1 : -1), (1 - Math.abs(n[0])) * (n[1] >= 0 ? 1 : -1), n[2]];
    const l = Math.hypot(...n);
    n = n.map((c) => c / l);
    const r = raw[v], rl = Math.hypot(...r);
    assert.ok(n.every((c, i) => Math.abs(c - r[i] / rl) < 1e-6), `normal ${v}`);
    const t = Array.from(tangents.subarray(v * 4, v * 4 + 3));
    assert.ok(Math.abs(Math.hypot(...t) - 1) < 1e-6, 'unit tangent');
    assert.ok(Math.abs(t[0] * n[0] + t[1] * n[1] + t[2] * n[2]) < 1e-6, 'tangent across the normal');
    assert.equal(tangents[v * 4 + 3], 1);
    assert.deepEqual(t.map((c) => Math.fround(c)), orthoTangent(n).map((c) => Math.fround(c)));
  }
});

test('checkNormalVolume accepts the MAMA layout and names the first bad field', () => {
  const v = volume();
  assert.equal(checkNormalVolume(v), v);
  const bad = (over, word) => assert.throws(() => checkNormalVolume(volume(over)),
    (e) => e instanceof SdfmError && e.code === 'sdfm.schema' && e.message.includes(word));
  bad({ format: 'other' }, 'format');
  bad({ version: 2 }, 'version');
  bad({ origin: [0, 0] }, 'origin');
  bad({ spacing: 0 }, 'spacing');
  bad({ bricks: [1, 0, 1] }, 'bricks');
  bad({ width: 24 }, 'width');
  bad({ bricks: [26, 1, 1] }, 'index rows');
  bad({ count: 2 }, 'every brick');
  bad({ texels: new Uint8Array(4) }, 'texels');
});

test('volumeInstanceParams and volumeTexture follow the WGSL params layout', () => {
  const v = volume();
  assert.deepEqual(volumeInstanceParams(v, 7), [-1, -1, -1, 0.5, 1, 1, 1, 7, 25, 11, 1, 1, 1, 0, 0, 0]);
  assert.deepEqual(volumeTexture(v), { width: 25, height: 11, data: v.texels });
});
