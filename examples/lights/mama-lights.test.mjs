import assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { join, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = fileURLToPath(new URL('../../', import.meta.url));
const mamaRoot = resolve(root, '../MAMA');
const adapterRoot = join(root, 'packages/raa-sdfm');
const git = (cwd, ...args) => execFileSync('git', args, { cwd, encoding: 'utf8', stdio: 'pipe' }).trim();
const requireValue = (ok, message) => { if (!ok) throw new Error(message); };
let evaluator;

async function producer(directory = mamaRoot) {
  const revision = git(directory, 'rev-parse', '--verify', 'HEAD');
  requireValue(/^[0-9a-f]{40}$/.test(revision), 'MAMA producer commit provenance is missing');
  const dist = join(directory, 'dist');
  for (const file of ['node/procgen.js', 'procgen/mama_procgen.wasm', 'procgen/scene-build.js',
    'procgen/scene-file.js', 'procgen/runtime.js']) {
    await fs.access(join(dist, file)).catch((cause) => {
      throw new Error(`Missing built MAMA dependency: ${file}; run linked MAMA npm run build`, { cause });
    });
  }
  const module = (file) => import(pathToFileURL(join(dist, file)).href);
  const node = await module('node/procgen.js');
  evaluator ??= node.loadProcgenNode(); // One evaluator per process; no independent workers.
  return {
    revision, dist, procgen: await evaluator,
    ...await module('procgen/scene-build.js'),
    ...await module('procgen/scene-file.js'),
    ...await module('procgen/runtime.js'),
  };
}

function input(innerAngleDeg = 24, runtime = false) {
  const link = (from, to, port, output = 'out') => ({
    from: { node: from, port: output }, to: { node: to, port },
  });
  return {
    version: 1, timeTicks: 0, paramValues: {}, sources: {},
    graph: {
      format: 'sdfm.procedural-graph', version: 1, seed: 7,
      nodes: [
        { id: 'placement', type: 'input_primitive', params: { collide: false } },
        { id: 'pose', type: 'op_transform', params: { translate: [0, 3, 0], rotateDeg: [-90, 0, 0] } },
        { id: 'spot', type: 'game_light', params: {
          kind: 'spot', intensity: 24, range: 6, angleDeg: 60,
          ...(innerAngleDeg === null ? {} : { innerAngleDeg }),
        } },
        { id: 'grid', type: 'array_grid', params: { counts: [1, 1, 1] } },
        { id: 'none', type: 'points_select', params: { mode: 'fraction', fraction: 0 } },
        { id: 'profile', type: 'input_curve', params: {
          points: [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], closed: true, smooth: 'polyline',
        } },
        { id: 'empty', type: 'element_primitive' },
        { id: 'collect', type: 'game_collect' },
        { id: 'output', type: 'output' },
      ],
      // The placement prototype is never published. Extruding a profile at zero points
      // produces real empty geometry for game_collect's required geometry input.
      links: [
        link('placement', 'pose', 'geometry'), link('pose', 'spot', 'instances'),
        link('grid', 'none', 'points'), link('none', 'empty', 'points', 'kept'),
        link('profile', 'empty', 'profile'), link('empty', 'collect', 'geometry'),
        link('spot', 'collect', 'markers'), link('collect', 'output', 'in'),
      ],
      params: runtime ? [{ name: 'power', label: 'Power', type: 'number', default: 24,
        binds: [{ node: 'spot', param: 'intensity' }] }] : [],
    },
  };
}

function evaluated(mama, source = input(), variant = 'sdf-refs') {
  const output = mama.procgen.evaluate(source);
  requireValue(output.ok === true, `MAMA evaluation failed: ${JSON.stringify(output.issues)}`);
  requireValue(output.sources.length === 0 && output.instances.length === 0, 'Expected light-only output');
  requireValue(output.lights?.length === 1, 'Expected one evaluated light');
  const plan = mama.planProceduralScene({
    name: 'mama-lights', game: 'lights', documentName: 'mama-lights', node: 'output',
    output, input: source, lookupAsset: () => null,
  }, variant);
  requireValue(plan.assets.length + plan.meshes.length + plan.reuseMeshes.length === 0, 'Unexpected geometry plan');
  return { output, scene: plan.scene, text: mama.serializeProceduralScene(plan.scene) };
}

/**
 * M3_LIGHT_FIXTURE staging handoff: the render-chain owner calls this once after the
 * ordinary page build, or uses the intact --stage CLI below. pageDir must already be an
 * ignored built page. Only canonical scene data, module trees and actual producer/build
 * provenance cross this boundary; the browser adapter owns all handles and registries.
 */
export async function stageMamaLights(pageDir) {
  await stage(pageDir, producer, fs);
}

async function stage(pageDir, load, io) {
  const page = resolve(pageDir);
  requireValue((await io.stat(page)).isDirectory(), 'pageDir must be an existing built page directory');
  git(root, 'check-ignore', '--', relative(root, page));
  const record = JSON.parse(await io.readFile(join(page, 'build.json'), 'utf8'));
  requireValue(record && typeof record.buildId === 'string' && record.buildId.length > 0
    && /^[0-9a-f]{40}$/.test(record.commit) && Number.isFinite(Date.parse(record.builtAt))
    && record.source === 'scripts/build-lights.sh' && typeof record.worktree === 'string'
    && typeof record.dirty === 'boolean', 'Missing page build provenance');
  const stagedAt = new Date().toISOString();
  const mama = await load();
  const { text, scene } = evaluated(mama);
  requireValue(scene.runtime === null, 'Staged scene must be static');
  await io.cp(mama.dist, join(page, 'mama'), { recursive: true, dereference: true });
  await io.cp(adapterRoot, join(page, 'raa-sdfm'), { recursive: true, dereference: true });
  await io.writeFile(join(page, 'mama-lights.scene.json'), text);
  await io.writeFile(join(page, 'build.json'), JSON.stringify({
    ...record, mamaProducerRevision: mama.revision, mamaStagedAt: stagedAt,
  }, null, 2) + '\n');
}

const direct = process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url;
if (direct) {
  if (process.argv.includes('--stage')) {
    requireValue(process.argv.length === 4 && process.argv[2] === '--stage', 'usage: --stage <built pageDir>');
    await stageMamaLights(process.argv[3]);
  } else {
    requireValue(process.argv.length === 2, 'usage: --stage <built pageDir>');
    await tests();
  }
}

// Recording-engine convention from spot.test.mjs; these are transport assertions,
// not rendered evidence. The real adapter is responsible for every recorded light.
function recordingEngine() {
  let next = 1;
  const lights = new Map();
  const meshes = [];
  const nodes = [];
  const instances = [];
  const calls = [];
  const settings = {};
  return {
    lights, meshes, nodes, instances, calls, settings,
    setCamera(value) { settings.camera = value; },
    setBackground(value) { settings.background = value; },
    setLight(value) { settings.sun = value; },
    setAmbient(value) { settings.ambient = value; },
    addMaterial: (desc) => desc,
    addMesh(desc) { meshes.push(desc); return meshes.length - 1; },
    addNode(parent, position, rotation, scale) {
      nodes.push({ parent, position, rotation, scale });
      return nodes.length - 1;
    },
    removeNode(node) { calls.push(['removeNode', node]); },
    addMeshInstance(node, mesh) { instances.push({ node, mesh }); },
    addLight(desc) {
      const handle = next++;
      calls.push(['addLight', handle, new Error().stack, structuredClone(desc)]);
      lights.set(handle, structuredClone(desc));
      return handle;
    },
    updateLight(handle, changes) {
      assert.ok(lights.has(handle));
      calls.push(['updateLight', handle]);
      lights.set(handle, structuredClone({ ...lights.get(handle), ...changes }));
    },
    removeLight(handle) {
      assert.ok(lights.delete(handle));
      calls.push(['removeLight', handle]);
    },
  };
}

async function tests() {
  const { test, after } = await import('node:test');
  const { inView, variantProblems } = await import('../../web/common/lights-job.mjs');
  const geometry = await import('../lookdev/lib/geometry.js');
  const preset = (await import('./scenes/mama-lights.js')).default;
  const { loadProceduralScene } = await import('../../packages/raa-sdfm/src/procedural.js');
  const lib = { geometry, pbr: (engine, color, opts) => engine.addMaterial({ ...opts, color, model: 'pbr' }) };
  let loading;
  const mama = () => loading ??= producer();
  await fs.mkdir(join(root, 'out'), { recursive: true });
  const scratch = await fs.mkdtemp(join(root, 'out/mama-lights-test-'));
  after(() => fs.rm(scratch, { recursive: true, force: true }));
  const buildRecord = {
    buildId: 'fixture-page-build', commit: git(root, 'rev-parse', 'HEAD'),
    builtAt: new Date().toISOString(), source: 'scripts/build-lights.sh', worktree: root, dirty: true,
    retained: { field: 'preserved' },
  };
  async function page() {
    const directory = await fs.mkdtemp(join(scratch, 'page-'));
    await fs.writeFile(join(directory, 'build.json'), JSON.stringify(buildRecord));
    return directory;
  }

  test('source metadata discovers on/off and pins the authored evidence and camera', async () => {
    assert.equal(preset.id, 'mama-lights');
    assert.equal(preset.milestone, 'm03');
    assert.deepEqual(preset.variants.map((v) => v.id), ['on', 'off']);
    const expected = [
      ['mama-lights-on-soft-spot-illumination.png',
        'A visible pool of spot illumination with a soft cone edge on a matte floor, produced by the loaded MAMA scene. Local lights cast no shadows yet.'],
      ['mama-lights-off-valid-floor-without-spot.png',
        'The same valid matte-floor scene under its ambient and sun lighting, without the demonstrated spot illumination. Local lights cast no shadows yet.'],
    ];
    preset.variants.forEach((variant, i) => {
      assert.deepEqual(variantProblems(variant), []);
      assert.deepEqual(variant.evidence, { filename: expected[i][0], label: expected[i][1] });
      assert.equal(variant.description, variant.evidence.label);
      assert.equal(variant.judge, true);
      assert.equal(variant.timeMs, 1000);
      assert.deepEqual(variant.selection, {
        camera: { eye: [0, 6, 9], target: [0, 0.8, 0], fovYDeg: 45, near: 0.1, far: 100 },
        subjects: [{ min: [-6, 0, -4], max: [6, 0, 4] }],
      });
    });
    const source = await fs.readFile(new URL('./scenes/mama-lights.js', import.meta.url), 'utf8');
    assert.doesNotMatch(source, /engine\.addLight\(/);
    assert.match(source, /import\('\.\.\/mama\/procgen\/scene-file\.js'\)/);
    assert.match(source, /import\('\.\.\/raa-sdfm\/src\/procedural\.js'\)/);
    const index = await fs.readdir(new URL('./scenes/', import.meta.url));
    assert.ok(index.includes(`${preset.id}.js`));
  });

  test('real evaluation and both export plans preserve identity, placement and 60/24 full angles', async () => {
    const m = await mama();
    for (const variant of ['sdf-refs', 'mesh-refs']) {
      const { output, scene, text } = evaluated(m, input(), variant);
      const parsed = m.parseProceduralScene(text);
      assert.deepEqual(scene.lights, output.lights);
      assert.equal(parsed.lights.length, output.lights.length);
      const original = output.lights[0];
      // Scene text uses the producer's existing 1e-6 numeric serialization grid.
      for (const field of ['t', 'r', 's', 'color']) {
        parsed.lights[0][field].forEach((v, i) => assert.ok(Math.abs(v - original[field][i]) <= 1e-6));
      }
      for (const field of ['id', 'node', 'key', 'kind', 'intensity', 'range', 'shadows', 'angleDeg', 'innerAngleDeg']) {
        assert.equal(parsed.lights[0][field], original[field]);
      }
      assert.equal(parsed.variant, variant);
      assert.equal(parsed.runtime, null);
      for (const section of ['sources', 'instances', 'groups', 'colliders', 'visual']) {
        assert.deepEqual(parsed[section], []);
      }
      const [light] = parsed.lights;
      assert.equal(light.node, 'spot');
      assert.equal(typeof light.id, 'string');
      assert.ok(Number.isInteger(light.key));
      assert.deepEqual(light.t, [0, 3, 0]);
      assert.equal(light.intensity, 24);
      assert.equal(light.range, 6);
      assert.equal(light.angleDeg, 60);
      assert.equal(light.innerAngleDeg, 24);
    }
  });

  test('a separately evaluated omitted inner angle stays absent through both exports and adapter creation', async () => {
    const m = await mama();
    for (const variant of ['sdf-refs', 'mesh-refs']) {
      const { output, text } = evaluated(m, input(null), variant);
      assert.equal(Object.hasOwn(output.lights[0], 'innerAngleDeg'), false);
      assert.equal(Object.hasOwn(m.parseProceduralScene(text).lights[0], 'innerAngleDeg'), false);
      const engine = recordingEngine();
      const handle = await loadProceduralScene(engine, null, text, { mama: async () => m });
      const [light] = engine.lights.values();
      assert.equal(light.angleDeg, 60);
      const created = engine.calls.find(([name]) => name === 'addLight')[3];
      assert.equal(Object.hasOwn(created, 'innerAngleDeg'), false);
      // The owner's complete-snapshot update explicitly resets omitted angles.
      assert.equal(light.innerAngleDeg, undefined);
      handle.dispose();
      assert.equal(engine.lights.size, 0);
    }
  });

  test('real Runtime.step publishes and applies a light-only change with stable handles and cleanup', async () => {
    const m = await mama();
    const { text } = evaluated(m, input(24, true));
    const engine = recordingEngine();
    const handle = await loadProceduralScene(engine, null, text, { mama: async () => m });
    const runtime = handle.runtime(m.procgen);
    try {
      assert.equal(engine.lights.size, 1);
      const [id] = engine.lights.keys();
      assert.equal((await runtime.step(0)).ok, true);
      assert.equal((await runtime.step(1)).delta.output, undefined);
      runtime.set({ power: 12 });
      const result = await runtime.step(2);
      assert.equal(result.ok, true);
      assert.equal(result.delta.output.lights[0].intensity, 12);
      for (const field of ['added', 'removed', 'moved', 'remesh']) assert.deepEqual(result.delta[field], []);
      assert.deepEqual([...engine.lights.keys()], [id]);
      assert.equal(engine.lights.get(id).intensity, 12);
      assert.equal(engine.lights.get(id).innerAngleDeg, 24);
    } finally {
      runtime.dispose();
      runtime.dispose();
      handle.dispose();
      handle.dispose();
    }
    assert.equal(engine.lights.size, 0);
    assert.equal(engine.calls.filter(([name]) => name === 'addLight').length, 1);
    assert.equal(engine.calls.filter(([name]) => name === 'removeLight').length, 1);
  });

  test('staging rejects missing page/provenance/dependencies without fabricating a producer', async () => {
    await assert.rejects(stageMamaLights(join(scratch, 'absent')), /ENOENT/);
    const directory = await page();
    await fs.rm(join(directory, 'build.json'));
    await assert.rejects(stageMamaLights(directory), /ENOENT/);
    await fs.writeFile(join(directory, 'build.json'), '{}');
    await assert.rejects(stageMamaLights(directory), /Missing page build provenance/);
    await fs.writeFile(join(directory, 'build.json'), JSON.stringify(buildRecord));
    await assert.rejects(stage(directory, async () => { throw new Error('Missing built MAMA dependency'); }, fs),
      /Missing built MAMA dependency/);
    await assert.rejects(fs.access(join(directory, 'mama-lights.scene.json')), /ENOENT/);
    const missing = await fs.mkdtemp('/tmp/mama-lights-missing-');
    try {
      await assert.rejects(producer(missing), /rev-parse/);
      // A real repository with no dist must fail its dependency gate before importing.
      await assert.rejects(producer(root), /Missing built MAMA dependency/);
    } finally {
      await fs.rm(missing, { recursive: true, force: true });
    }
  });

  test('evaluation, export, copy and write failures reject staging and preserve build provenance', async () => {
    const m = await mama();
    const failure = new Error('staging failure');
    for (const [load, io, expected] of [
      [async () => ({ ...m, procgen: { evaluate: () => ({ ok: false, issues: ['refused'] }) } }), fs, /evaluation failed/],
      [async () => ({ ...m, planProceduralScene() { throw failure; } }), fs, /staging failure/],
      [async () => ({ ...m, serializeProceduralScene() { throw failure; } }), fs, /staging failure/],
      [async () => m, { ...fs, cp: async () => { throw failure; } }, /staging failure/],
      [async () => m, { ...fs, writeFile: async () => { throw failure; } }, /staging failure/],
    ]) {
      const directory = await page();
      await assert.rejects(stage(directory, load, io), expected);
      assert.deepEqual(JSON.parse(await fs.readFile(join(directory, 'build.json'))), buildRecord);
    }
  });

  let staged;
  async function browserScene() {
    staged ??= (async () => {
      const directory = await page();
      await stageMamaLights(directory);
      await fs.mkdir(join(directory, 'scenes'));
      await fs.copyFile(new URL('./scenes/mama-lights.js', import.meta.url), join(directory, 'scenes/mama-lights.js'));
      // Resolve the copied producer's bare dependencies from its linked checkout in
      // Node only. The downstream browser build bundles them at the declared URLs.
      const { registerHooks } = await import('node:module');
      const hooks = registerHooks({ resolve(specifier, context, nextResolve) {
        if (context.parentURL?.startsWith(pathToFileURL(join(directory, 'mama/')).href)
          && !specifier.startsWith('.') && !specifier.includes(':') && !specifier.startsWith('/')) {
          return nextResolve(specifier, { ...context, parentURL: pathToFileURL(join(mamaRoot, 'dist/index.js')).href });
        }
        return nextResolve(specifier, context);
      } });
      after(() => hooks.deregister());
      return { directory, scene: (await import(pathToFileURL(join(directory, 'scenes/mama-lights.js')).href)).default };
    })();
    return staged;
  }

  test('stage transfers canonical real output and module trees and preserves actual build provenance', async () => {
    const m = await mama();
    const { directory } = await browserScene();
    const record = JSON.parse(await fs.readFile(join(directory, 'build.json')));
    const { mamaProducerRevision, mamaStagedAt, ...retained } = record;
    assert.deepEqual(retained, buildRecord);
    assert.equal(mamaProducerRevision, git(mamaRoot, 'rev-parse', 'HEAD'));
    assert.equal(new Date(mamaStagedAt).toISOString(), mamaStagedAt);
    const actual = await fs.readFile(join(directory, 'mama-lights.scene.json'), 'utf8');
    assert.equal(actual, evaluated(m).text);
    assert.equal(m.parseProceduralScene(actual).runtime, null);
    for (const file of ['mama/node/procgen.js', 'mama/procgen/mama_procgen.wasm',
      'mama/procgen/scene-file.js', 'raa-sdfm/src/procedural.js', 'raa-sdfm/src/mama-light-owner.js']) {
      await fs.access(join(directory, file));
    }
  });

  test('staged browser build awaits preparation and calls the actual adapter; both variants retain a valid floor', async (t) => {
    const { directory, scene } = await browserScene();
    const text = await fs.readFile(join(directory, 'mama-lights.scene.json'), 'utf8');
    let preparation;
    t.mock.method(globalThis, 'fetch', async (url) => {
      assert.equal(url.href, pathToFileURL(join(directory, 'mama-lights.scene.json')).href);
      return { ok: true, text: () => preparation.promise };
    });
    const engines = [];
    for (const variant of scene.variants) {
      preparation = Promise.withResolvers();
      const engine = recordingEngine();
      engines.push(engine);
      let complete = false;
      const building = scene.build(engine, lib, variant).then((result) => { complete = true; return result; });
      await new Promise(setImmediate);
      assert.equal(complete, false);
      assert.equal(engine.lights.size, 0);
      preparation.resolve(text);
      assert.equal(await building, undefined);
      assert.equal(engine.lights.size, 1);
      const [light] = engine.lights.values();
      assert.equal(light.intensity, variant.id === 'on' ? 24 : 0);
      assert.equal(light.angleDeg, 60);
      assert.equal(light.innerAngleDeg, 24);
      assert.deepEqual(light.position, [0, 3, 0]);
      light.direction.forEach((v, i) => assert.ok(Math.abs(v - [0, -1, 0][i]) < 1e-6));
      assert.match(engine.calls.find(([name]) => name === 'addLight')[2], /mama-light-owner\.js/);
      assert.equal(engine.instances.length, 1);
      const { node, mesh } = engine.instances[0];
      assert.deepEqual(engine.nodes[node], {
        parent: null, position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1],
      });
      const floor = engine.meshes[mesh];
      const axes = [0, 1, 2].map((axis) => Array.from(floor.positions).filter((_, i) => i % 3 === axis));
      const bounds = { min: axes.map((a) => Math.min(...a)), max: axes.map((a) => Math.max(...a)) };
      assert.deepEqual(bounds, variant.selection.subjects[0]);
      assert.ok(inView(variant.selection.camera, bounds));
      assert.ok([...floor.positions, ...floor.normals].every(Number.isFinite));
      assert.ok(floor.indices.length > 0 && floor.indices.every((i) => i < floor.positions.length / 3));
      assert.equal(floor.subMeshes[0].material.roughness, 0.85);
      assert.equal(floor.subMeshes[0].material.model, 'pbr');
    }
    assert.deepEqual(engines[0].settings, engines[1].settings);
    assert.deepEqual(engines[0].meshes, engines[1].meshes);
    assert.deepEqual(engines[0].instances, engines[1].instances);
  });

  test('staged browser rejects fetch, HTTP, body preparation, parser and adapter failures', async (t) => {
    const { directory, scene } = await browserScene();
    const text = await fs.readFile(join(directory, 'mama-lights.scene.json'), 'utf8');
    let response;
    t.mock.method(globalThis, 'fetch', () => response());
    for (const [read, message] of [
      [async () => { throw new Error('fetch failed'); }, /fetch failed/],
      [async () => ({ ok: false, status: 503 }), /HTTP 503/],
      [async () => ({ ok: true, text: async () => { throw new Error('preparation rejected'); } }), /preparation rejected/],
      [async () => ({ ok: true, text: async () => '{}' }), /PSCENE_/],
    ]) {
      response = read;
      const engine = recordingEngine();
      await assert.rejects(scene.build(engine, lib, scene.variants[0]), message);
      assert.equal(engine.lights.size, 0);
    }
    response = async () => ({ ok: true, text: async () => text });
    const engine = recordingEngine();
    engine.addLight = () => { throw new Error('adapter creation failed'); };
    await assert.rejects(scene.build(engine, lib, scene.variants[0]), /adapter creation failed/);
    assert.equal(engine.lights.size, 0);
  });

  test('import is inert; --stage dispatches once without tests and fails on staging errors', async () => {
    const file = fileURLToPath(import.meta.url);
    const imported = execFileSync(process.execPath, ['--input-type=module', '-e', `await import(${JSON.stringify(import.meta.url)})`],
      { encoding: 'utf8', timeout: 60000 });
    assert.equal(imported, '');
    assert.throws(() => execFileSync(process.execPath, [file, '--stage', join(scratch, 'absent')],
      { encoding: 'utf8', timeout: 60000, stdio: 'pipe' }), /ENOENT/);
    const directory = await page();
    const output = execFileSync(process.execPath, [file, '--stage', directory],
      { encoding: 'utf8', timeout: 60000, stdio: 'pipe' });
    assert.equal(output, '');
    const record = JSON.parse(await fs.readFile(join(directory, 'build.json')));
    assert.equal(record.mamaProducerRevision, git(mamaRoot, 'rev-parse', 'HEAD'));
    assert.equal(typeof record.mamaStagedAt, 'string');
    assert.equal((await fs.readdir(directory)).filter((f) => f === 'mama-lights.scene.json').length, 1);
  });
}
