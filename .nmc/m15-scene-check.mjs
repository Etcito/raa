// CPU-only build/animation audit; this is not render or judge evidence.
import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import scene from '../examples/lookdev/scenes/m15-waddle.js';

export async function traceScene(module, id) {
  let handle = 0;
  const calls = [];
  const engine = new Proxy({}, { get: (_, method) => (...args) => {
    calls.push({ method, args });
    return ++handle;
  } });
  const variant = module.variants.find(v => v.id === id);
  const { rigs } = await module.build(engine, {}, variant);
  module.frame(engine, variant.timeMs, variant);
  assert.equal(rigs.length, 4);
  return calls;
}

const ids = process.argv.slice(2);
for (const id of ids.length ? ids : scene.variants.map(v => v.id)) {
  const calls = await traceScene(scene, id);
  const last = method => calls.filter(c => c.method === method).at(-1)?.args[0];
  const settings = Object.fromEntries(['setCamera', 'setFog', 'setGrading',
    'setToneMapping', 'setLight', 'setShadows'].map(m => [m, last(m)]));
  const passes = Object.fromEntries(calls.filter(c => c.method === 'setPassEnabled').map(c => c.args));
  const materials = calls.filter(c => c.method === 'addSurfaceMaterial').map(c => c.args[0]);
  if (id !== 'f00' && id !== 'tints') {
    for (const pass of ['fog', 'sunshafts', 'bloom', 'dof']) assert.equal(passes[pass], false);
    assert.equal(settings.setFog.density, 0);
    assert.equal(settings.setFog.aerialStrength, 0);
    assert.deepEqual(settings.setGrading.lift, [0, 0, 0]);
    assert.ok(settings.setCamera.eye[2] < 0);
    assert.equal(passes['shadow.csm'], true);
    assert.ok(materials.some(m => m.furShells === 64 && m.furLength === 0.018 && m.furDensity === 48));
    assert.ok(materials.some(m => m.function === 'raa.plush' && m.params.color.every(c => c === 1)));
  }
  if (id === 'f06') {
    mkdirSync('target/lookdev', { recursive: true });
    // Capture's standard input contains only args, not scene settings. This extra
    // diagnostic records actual public API calls locally; capture regenerates args.
    writeFileSync('target/lookdev/m15-waddle--f06.input.json', JSON.stringify({
      args: { scene: scene.id, variant: id, width: 1280, height: 720 },
      localSceneAudit: { settings, passes, materials },
    }, null, 2) + '\n');
  }
  console.log(`PASS ${scene.id}--${id}: CPU build and fixed-time animation`);
}
