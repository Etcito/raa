// node --test "examples/lookdev/test/*.test.mjs": the m13 mud set builds against a recording
// fake engine, adds white spray emitters and glossy mesh crowns at every pad, and its bank
// reaches the horizon. Proves counts and parameters only, not the look (GPU card judges it).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as geometry from '../lib/geometry.js';
import * as objects from '../lib/objects.js';
import scene, { mudSet } from '../scenes/m13-liquids-particles.js';

function fakeEngine() {
  const rec = { emitters: [], materials: [], meshes: [], instances: [], lights: [], orbit: null };
  const engine = new Proxy({}, {
    get(_, name) {
      if (name === 'addEmitter') return (e) => rec.emitters.push(e) - 1;
      if (name === 'addSurfaceMaterial') return (m) => rec.materials.push(m) - 1;
      if (name === 'addMesh') return (m) => rec.meshes.push(m) - 1;
      if (name === 'addMeshInstance') return (node, mesh) => rec.instances.push(mesh) - 1;
      if (name === 'addNode') return () => 0;
      if (name === 'setLight') return (l) => rec.lights.push(l);
      if (name === 'setOrbit') return (...a) => { rec.orbit = a; };
      return () => {};
    },
  });
  return { engine, rec };
}

const mud = scene.variants.find((v) => v.id === 'mud');

test('mud set spawns white spray at every pad', async () => {
  const { engine, rec } = fakeEngine();
  await scene.build(engine, { geometry, objects }, mud);
  const spray = rec.emitters.filter((e) => e.kind === 'splash');
  assert.ok(spray.length >= mudSet.pads.length * 2, `spray emitters: ${spray.length}`);
  for (const e of spray) {
    assert.ok(e.rate > 0 && e.maxParticles > 0);
    assert.ok(e.opacity > 0.5, `spray opacity ${e.opacity}`);
    for (const c of e.colors) assert.ok(Math.min(...c) > 0.85, `spray colour ${c}`);
  }
  for (const [x, z, r] of mudSet.pads) {
    const near = spray.filter((e) => Math.hypot(e.position[0] - x, e.position[2] - z) < r + 0.45);
    assert.ok(near.length > 0, `no spray at pad ${x}, ${z}`);
  }
});

test('mud set builds a glossy chocolate crown mesh at every pad', async () => {
  const { engine, rec } = fakeEngine();
  await scene.build(engine, { geometry, objects }, mud);
  const crown = rec.materials.findIndex((m) => m.color === mudSet.crownStyle.color);
  assert.ok(crown >= 0, 'no crown material');
  assert.ok(rec.materials[crown].roughness < 0.2);
  const crowns = rec.meshes.filter((m) => m.subMeshes[0].material === crown);
  assert.equal(crowns.length, mudSet.pads.length);
  for (const m of crowns) {
    let top = -Infinity;
    for (let i = 1; i < m.positions.length; i += 3) top = Math.max(top, m.positions[i]);
    assert.ok(top > 0.15, `crown top ${top}`);
    assert.ok(m.positions.every(Number.isFinite) && m.normals.every(Number.isFinite));
  }
});

test('mud bank reaches the horizon', async () => {
  const { engine, rec } = fakeEngine();
  await scene.build(engine, { geometry, objects }, mud);
  const span = Math.max(...rec.meshes.map((m) => Math.max(...m.positions.filter((_, i) => i % 3 === 0))));
  assert.ok(span >= 30, `bank half width ${span}`);
});

test('water and slime sets still build', async () => {
  for (const id of ['water', 'slime', 'splash']) {
    const { engine, rec } = fakeEngine();
    await scene.build(engine, { geometry, objects }, scene.variants.find((v) => v.id === id));
    assert.ok(rec.meshes.length > 0);
  }
});

test('mud sun lights the camera-facing side of the crowns', async () => {
  const { engine, rec } = fakeEngine();
  await scene.build(engine, { geometry, objects }, mud);
  // The last light set wins; the eye sits at target + (cos p sin y, sin p, cos p cos y).
  const d = rec.lights.at(-1).direction;
  const [, , yaw, pitch] = rec.orbit;
  const eye = [Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch), Math.cos(pitch) * Math.cos(yaw)];
  // Light travels along d, so a crown side facing the eye is lit when dot(eye, -d) > 0
  // (a backlit crown rendered grey-black, card 5064).
  const toSun = d.map((v) => -v);
  const facing = eye[0] * toSun[0] + eye[2] * toSun[2];
  assert.ok(facing > 0.2, `sun lights the camera side: ${facing}`);
  assert.ok(toSun[1] > 0.5, `sun elevation: ${toSun[1]}`);
});

test('mud crowns are flared sheets, not spheres', async () => {
  // Card 5070's judge read lathe tongues with droplet spheres as "blobby spheres": the crown
  // is one sheet whose lobes flare outward from the pad wall as they rise.
  for (const [i, pad] of mudSet.pads.entries()) {
    const m = mudSet.crown(geometry, pad, i, 1);
    const [, , r] = pad;
    let flared = 0;
    let top = -Infinity;
    for (let k = 0; k < m.positions.length; k += 3) {
      const y = m.positions[k + 1];
      top = Math.max(top, y);
      if (y > 0.18 && Math.hypot(m.positions[k], m.positions[k + 2]) > r + 0.1) flared++;
    }
    assert.ok(top > 0.25, `crown top ${top}`);
    assert.ok(flared > 50, `flared lobe vertices: ${flared}`);
    const lobes = Array.from({ length: mudSet.crownStyle.lobes }, (_, k) => mudSet.lobe(i, k));
    assert.ok(lobes.length >= 7 && lobes.every((l) => l.h > 0.2));
  }
});

test('mud pads are low wide discs with a rounded rim', () => {
  for (const pad of mudSet.pads) {
    const [, , r, h] = pad;
    assert.ok(h < 0.25 * 2 * r, `pad height ${h} vs diameter ${2 * r}`);
    const prof = mudSet.padProfile(r, h);
    const top = Math.max(...prof.map(([, y]) => y));
    assert.ok(top < 0.5 * r, `pad top ${top}`);
    // The rim is rounded: several profile points between the wall top and the top face.
    const shoulder = prof.filter(([rho, y]) => y > 0.6 * h && y < h && rho > 0.7 * r);
    assert.ok(shoulder.length >= 4, `rim points ${shoulder.length}`);
    const m = mudSet.pad(geometry, pad, mudSet.padPose(pad, 1), 1);
    assert.ok(m.positions.every(Number.isFinite) && m.normals.every(Number.isFinite));
  }
});

test('mud pads float tilted on the swell, top clear and underside in the mud', () => {
  for (const [x, z, r, h] of mudSet.pads) {
    const pose = mudSet.padPose([x, z, r], 1);
    assert.ok(Math.abs(Math.hypot(...pose.q) - 1) < 1e-9);
    for (let a = 0; a < 6.28; a += 0.25) {
      const rim = (y) => {
        const w = mudSet.rotate(pose.q, [r * Math.cos(a), y, r * Math.sin(a)]);
        return { y: pose.base + w[1], mud: mudSet.height(x + w[0], z + w[2], 1) };
      };
      const top = rim(h);
      const keel = rim(-0.12);
      assert.ok(top.y - top.mud > 0.08, `pad top clears the mud at ${x}, ${z}: ${top.y - top.mud}`);
      assert.ok(keel.mud - keel.y > 0.05, `pad underside in the mud at ${x}, ${z}`);
    }
  }
});

test('mud surface is one low swell with a narrow white froth band', () => {
  const M = mudSet.material;
  const waves = [M.wave0, M.wave1, M.wave2, M.wave3];
  // The longest wave carries the swell; the short ones are faint ripples, not bumps.
  assert.ok(waves[0][1] >= 3 && waves[0][0] <= 0.1);
  for (const [a, l, , q] of waves.slice(2)) {
    assert.ok(a / l < 0.015, `short wave slope ${a / l}`);
    assert.ok(q <= 0.3, `short wave steepness ${q}`);
  }
  assert.ok(M.foamWidth <= 0.12 && M.foamAmount <= 0.3, 'froth band narrow');
  assert.ok(M.detailStrength <= 0.03);
  // Warm milk chocolate (sRGB) as bright as ref-3's mud: the dark (0.36, 0.2, 0.12)
  // rendered maroon and purple under the sky reflection (card 5077).
  assert.ok(M.color[0] >= 0.45 && M.color[0] <= 0.65, `red ${M.color[0]}`);
  assert.ok(M.color[1] / M.color[0] >= 0.5 && M.color[2] / M.color[0] <= 0.35, `hue ${M.color}`);
});

test('mud crowns are large and spray is thin droplets arcing from the lobe tips', () => {
  for (const [i, pad] of mudSet.pads.entries()) {
    const lobes = Array.from({ length: mudSet.crownStyle.lobes }, (_, k) => mudSet.lobe(i, k));
    assert.ok(Math.max(...lobes.map((l) => l.h)) >= 0.45, 'tallest lobe');
    assert.ok(lobes.every((l) => l.flare >= 0.2), 'lobes flare out');
    const m = mudSet.spray(geometry, pad, i, 1);
    const verts = m.positions.length / 3;
    const perDrop = verts / (lobes.length * mudSet.drops.count);
    assert.ok(Number.isInteger(perDrop) && perDrop > 0);
    assert.ok(m.positions.every(Number.isFinite) && m.normals.every(Number.isFinite));
    // Each drop is stretched: its extent along its longest axis is about stretch x width.
    for (let d = 0; d < 3; d++) {
      const pts = [];
      for (let v = d * perDrop; v < (d + 1) * perDrop; v++) pts.push([0, 1, 2].map((c) => m.positions[3 * v + c]));
      let far = 0;
      for (const p of pts) for (const q of pts) far = Math.max(far, Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]));
      assert.ok(far > 2 * 2 * mudSet.drops.radius[0], `drop length ${far}`);
    }
  }
});

test('mud bank is a grass material over a ridge around the pit', async () => {
  const { engine, rec } = fakeEngine();
  const fns = [];
  const eng = new Proxy(engine, {
    get(target, name) {
      if (name === 'addMaterialFunction') return (n, w) => fns.push([n, w]);
      return target[name];
    },
  });
  await scene.build(eng, { geometry, objects }, mud);
  const grass = fns.find(([n]) => n === 'demo.grass');
  assert.ok(grass && /fn material_main\(in: MaterialInput\) -> MaterialOutput/.test(grass[1]));
  assert.ok(!/@group|@binding|fn raa_/.test(grass[1]));
  assert.ok(rec.materials.some((m) => m.function === 'demo.grass'));
  const ridge = mudSet.ground(3.3, 0);
  assert.ok(ridge > mudSet.ground(6, 0) + 0.08 && ridge > 0.3, `ridge ${ridge}`);
});

test('mud spray drops are white glowing beads, not thin needles', () => {
  const D = mudSet.drops;
  assert.ok(D.stretch <= 2.5, `stretch ${D.stretch}`);
  assert.ok(D.radius[0] >= 0.015, `radius ${D.radius}`);
  assert.ok(Math.min(...D.color) > 0.9 && Math.min(...D.emissive) > 0.3);
});

test('mud bank rises in a rounded mound well above the meadow', () => {
  assert.ok(mudSet.ground(3.6, 0) - mudSet.ground(8, 0) > 0.35, 'mound height');
  for (const [x, z, r] of mudSet.pads) {
    // The pads and their crowns stay inside the pit, clear of the bank.
    const a = Math.atan2(z, x);
    assert.ok(mudSet.ground(x + r * Math.cos(a), z + r * Math.sin(a)) < -0.05, `pad ${x}, ${z} on the bank`);
  }
});

test('mud pads have a raised rim tube around a recessed top', () => {
  for (const [, , r, h] of mudSet.pads) {
    const prof = mudSet.padProfile(r, h);
    const centre = prof.at(-1)[1];
    const crown = Math.max(...prof.map(([, y]) => y));
    assert.ok(crown - centre > 0.015, `rim stands ${crown - centre} m above the top`);
  }
});
