import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mapMaterial } from '../src/map-material.js';
import { SdfmError } from '../src/errors.js';
import {
  REST_TANGENT_AXIS, checkNormalVolume, restTangent, volumeInstanceParams, volumeMaterial, volumeStandardMaterial, volumeSurfaceAttributes,
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
    assert.deepEqual(Array.from(colors.subarray(v * 4, v * 4 + 3)), Array.from(positions.subarray(v * 3, v * 3 + 3)));
    // (uv, colour w) is the unit rest normal (sdfv_shading_normal reads it so)
    const n = [uvs[v * 2], uvs[v * 2 + 1], colors[v * 4 + 3]];
    const r = raw[v], rl = Math.hypot(...r);
    assert.ok(n.every((c, i) => Math.abs(c - r[i] / rl) < 1e-6), `normal ${v}`);
    const t = Array.from(tangents.subarray(v * 4, v * 4 + 3));
    assert.ok(Math.abs(t[0] * n[0] + t[1] * n[1] + t[2] * n[2]) < 1e-6, 'tangent across the normal');
    assert.equal(tangents[v * 4 + 3], 1);
    assert.deepEqual(t.map((c) => Math.fround(c)), restTangent(n).map((c) => Math.fround(c)));
  }
});

test('volumeSurfaceAttributes: the frame rebuilt from interpolated streams is the interpolated frame (any hemisphere)', () => {
  // sdfv_shading_normal in JS, at rest (posed = rest): it must hand the rest gradient back unchanged inside every
  // triangle, also across the octahedral fold and the z = 0 seam that broke the former encoding (file 34, 2026-11-04).
  const norm = (a) => { const l = Math.hypot(...a); return a.map((c) => c / l); };
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const tris = [[[0.1, -0.1, -1], [-0.1, 0.05, -1], [0.02, 0.1, -1]], [[0.3, 0.2, 0.05], [0.3, 0.1, -0.05], [0.25, 0.3, 0]],
    [[-0.5, -0.5, -0.7], [0.5, -0.5, -0.7], [0, 0.6, -0.8]]];
  let worst = 0;
  for (const tri of tris) {
    const ns = tri.map(norm);
    const { colors, uvs, tangents } = volumeSurfaceAttributes(new Float32Array(9), new Float32Array(ns.flat()));
    for (const [u, w] of [[1 / 3, 1 / 3], [0.5, 0.25], [0.1, 0.8], [0.45, 0.45]]) {
      const bw = [1 - u - w, u, w], lerp = (f) => [0, 1, 2].map((k) => bw.reduce((s, b, j) => s + b * f(j)[k], 0));
      const nr0 = lerp((j) => [uvs[j * 2], uvs[j * 2 + 1], colors[j * 4 + 3]]);
      const tan = lerp((j) => Array.from(tangents.subarray(j * 4, j * 4 + 3)));
      const n = norm(lerp((j) => ns[j])), g = norm(lerp((j) => ns[j]).map((c, k) => c + [0.2, -0.1, 0.3][k]));
      const nr = norm(nr0), tr0 = cross(REST_TANGENT_AXIS, nr0), tr = norm(tr0.map((c, k) => c - nr[k] * dot(nr, tr0)));
      const t = norm(tan.map((c, k) => c - n[k] * dot(n, tan))), br = cross(nr, tr), bw2 = cross(n, t);
      const m = norm([0, 1, 2].map((k) => dot(g, tr) * t[k] + dot(g, br) * bw2[k] + dot(g, nr) * n[k]));
      worst = Math.max(worst, Math.acos(Math.min(1, dot(m, g))));
    }
  }
  assert.ok(worst < 1e-4, `rebuilt frame off by ${(worst * 180 / Math.PI).toFixed(3)} degrees`);
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
