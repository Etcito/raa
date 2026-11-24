import init, { Engine } from './raa/raa.js';
import { loadEngine } from './ege/loader.js';
import { RollbackBinding } from './ege/rollback.js';

const canvas = document.querySelector('canvas');
const hud = document.querySelector('output');
const status = document.querySelector('[role=status]');
const button = document.querySelector('button');
const width = canvas.width;
const height = canvas.height;
const rotation = [0, 0, 0, 1];
const unit = [1, 1, 1];
const base = [0.03, 0.11, 0.5];
const actor = [0.65, 0.05, 0.025];
const stripe = [0.25, 0.75, 0.025];
const flash = [0.9, 0.8, 0.18];
const firstX = 20;
const nextX = 27;
const actorWidth = 12;
const actorHeight = 12;
const bandTop = 18;
const actorX = snapshot => snapshot.inputs.inputIds[0] === 'first-0' ? firstX : nextX;

function require(condition, message) {
  if (!condition) throw new Error(message);
}

function targetGuides(from, to) {
  const count = width * height;
  const depth = new Float32Array(count);
  const motion = new Float32Array(count * 4);
  const changed = new Uint32Array(count);
  const repair = new Float32Array(count * 4);
  let holes = 0;
  let lightPixels = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pixel = y * width + x;
      const row = y >= bandTop && y < bandTop + actorHeight;
      const visible = row && x >= to && x < to + actorWidth;
      const oldActor = row && x >= from && x < from + actorWidth;
      const hole = oldActor && !visible;
      // Independent target change inside the actor. The fixture supplies this sample
      // even though its geometric correspondence is valid.
      const lighting = visible && x === to;
      depth[pixel] = ((visible ? 4 : 5) - 0.02 - 1) / 8;
      motion[4 * pixel] = visible ? (to - from) / width : 0;
      motion[4 * pixel + 3] = depth[pixel];
      if (hole || lighting) {
        repair.set([...(lighting ? stripe : base), 1], 4 * pixel);
        changed[pixel] = Number(lighting);
        holes += Number(hole);
        lightPixels += Number(lighting);
      }
    }
  }
  return { depth, motion, changed, repair, holes, lightPixels };
}

const opportunity = () => new Promise(resolve => requestAnimationFrame(resolve));
const asset = async name => {
  const result = await fetch(new URL(name, import.meta.url));
  if (!result.ok) throw new Error(`${name}: HTTP ${result.status}`);
  return result.text();
};

let running = false;
async function run({ buildId } = {}) {
  if (running) throw new Error('Prediction fixture is already running');
  require(/Windows/.test(navigator.userAgent), 'Prediction verification requires Windows Chrome');
  running = true;
  button.disabled = true;
  let engine;
  try {
    status.textContent = 'Running prediction fixture';
    await init();
    const simEngine = await loadEngine({ wasmUrl: new URL('./ege.wasm', import.meta.url) });
    simEngine.setSchema(await asset('./schema.json'));
    const world = simEngine.createWorld(await asset('./scene.json'));
    const { simulation } = new RollbackBinding(simEngine, world, {
      playerCount: 2, peers: [{ peer: 0, players: [0] }, { peer: 1, players: [1] }],
    });
    function snapshot(event, discontinuities = []) {
      const tick = simEngine.tick(world);
      const receivedAtMs = performance.now();
      simulation.stepCodes([0, 0]);
      return simulation.capturePrediction(1, performance.now(), [0, 1].map(player => ({
        player, tick, id: `${event}-${player}`, code: 0, receivedAtMs,
      })), { lights: [], discontinuities });
    }

    engine = await Engine.create(canvas, { allowSoftware: false, msaa: 1, toneMapper: 'linear' });
    const { adapter, hardware } = engine.stats();
    require(hardware, 'A hardware WebGPU adapter is required');
    engine.setCamera({ eye: [0, 0, 5], target: [0, 0, 0], projection: 'orthographic', orthoHeight: 1, near: 1, far: 9 });
    engine.setToneMapping({ dither: false });
    engine.setAutoExposure({ enabled: false });
    for (const pass of engine.passes()) {
      if (pass.enabled && !pass.required) engine.setPassEnabled(pass.name, false);
    }
    engine.setCameraJitter(false);
    engine.setRenderScale(1);
    const create = (size, color) => {
      const node = engine.addNode(null);
      const material = engine.addMaterial({ color: [0, 0, 0], emissive: color, model: 'flat' });
      engine.addInstance(node, engine.primitiveMesh('box', { halfExtents: size.map(value => value / 2) }), material);
      return { node, material };
    };
    create([3, 3, 0.04], base);
    const strip = create([1 / height, actorHeight / height, 0.04], actor);
    const block = create([(actorWidth - 1) / height, actorHeight / height, 0.04], actor);
    function move(x) {
      engine.setTransform(strip.node, [(x + 0.5 - width / 2) / height, 0, 1], rotation, unit);
      engine.setTransform(block.node, [(x + (actorWidth + 1) / 2 - width / 2) / height, 0, 1], rotation, unit);
    }
    require(engine.preparePrediction(), 'The canvas does not support split presentation');

    const pixels = document.createElement('canvas');
    pixels.width = width;
    pixels.height = height;
    const read = pixels.getContext('2d', { willReadFrequently: true });
    function present(target) {
      const selected = engine.presentReady(target);
      const mode = selected?.predicted ? 'prediction' : selected ? 'real' : 'hold';
      hud.textContent = `Tick ${target.target.tick} · ${mode}`;
      return selected;
    }
    function sampleCanvas() {
      read.drawImage(canvas, 0, 0);
      return read.getImageData(0, 0, width, height).data;
    }
    async function waitFor(target, predicate) {
      for (let attempt = 0; attempt < 100; attempt++) {
        await opportunity();
        const frame = present(target);
        if (frame && predicate(frame)) return { frame, pixels: sampleCanvas(), hud: hud.textContent };
      }
      throw new Error(`Expected frame was never selected at tick ${target.target.tick}: ${hud.textContent}`);
    }
    const first = snapshot('first');
    move(actorX(first));
    const source = engine.produceReal(performance.now(), first);
    require(source !== undefined, 'First real frame did not submit');
    const start = await waitFor(first, frame => frame.realFrameId === source);
    const target = snapshot('move');
    require(target.inputs.confirmedMask.every(Boolean), 'Target snapshot must contain cutoff input receipts');
    const guides = targetGuides(actorX(first), actorX(target));
    const submit = (snapshot, limit) => engine.submitPrediction(snapshot, source, performance.now() + 1500,
      20, 0.0005, limit, guides.depth, guides.motion, guides.changed, guides.repair);
    require(!engine.submitPrediction(target, source, performance.now() - 1, 20, 0.0005,
      guides.holes + guides.lightPixels, guides.depth, guides.motion, guides.changed, guides.repair),
    'An expired deadline admitted prediction');
    const missed = present(target);
    require(missed?.realFrameId === source && !missed.predicted, 'A missed deadline did not present the causal real image');
    const needed = guides.holes + guides.lightPixels;
    require(submit(target, needed - 1), 'Budget-overflow fixture did not submit');
    let admitted = false;
    for (let attempt = 0; attempt < 100 && !admitted; attempt++) {
      await opportunity();
      require(!present(target)?.predicted, 'An incomplete repair reached the display');
      admitted = submit(target, needed);
    }
    require(admitted, 'Valid prediction was not admitted after overflow fallback');
    const predicted = await waitFor(target, frame => frame.predicted);
    require(engine.stats().frames === 1, 'Prediction advanced real history');
    require(predicted.hud === `Tick ${target.target.tick} · prediction`, 'Prediction HUD did not use the current snapshot');

    // Render the same target only after selecting prediction. These pixels cannot feed the candidate.
    move(actorX(target));
    engine.setMaterial(strip.material, { color: [0, 0, 0], emissive: stripe, model: 'flat' });
    const targetReal = engine.produceReal(performance.now(), target);
    require(targetReal !== undefined, 'Target real frame did not submit');
    const reference = await waitFor(target, frame => !frame.predicted && frame.realFrameId === targetReal);
    let maximumError = 0;
    let movingPixels = 0;
    let worstPixel = 0;
    for (let index = 0; index < reference.pixels.length; index++) {
      const error = Math.abs(predicted.pixels[index] - reference.pixels[index]);
      if (error > maximumError) {
        maximumError = error;
        worstPixel = Math.floor(index / 4);
      }
      if (index % 4 === 0 && Math.abs(start.pixels[index] - reference.pixels[index]) > 16) movingPixels++;
    }
    require(movingPixels >= guides.holes * 2, `Fixture did not move visibly: ${movingPixels}`);
    const rgba = frame => Array.from(frame.pixels.slice(worstPixel * 4, worstPixel * 4 + 4));
    require(maximumError <= 3, `Predicted pixels differ from the full target: max ${maximumError} at ${worstPixel % width},${Math.floor(worstPixel / width)}. Predicted ${rgba(predicted)}; real ${rgba(reference)}; source ${rgba(start)}`);

    const eventTick = simEngine.tick(world);
    const flashTarget = snapshot('flash', [{ kind: 'light-change', tick: eventTick }]);
    require(flashTarget.discontinuities.length === 1, 'Flash discontinuity was not captured');
    require(!engine.submitPrediction(flashTarget, targetReal, performance.now() + 1500, 20, 0.0005,
      0, guides.depth, guides.motion, guides.changed, guides.repair), 'Flash was admitted as early prediction');
    require(present(flashTarget)?.realFrameId === targetReal, 'Flash fallback did not preserve the latest real frame');
    require(sampleCanvas().every((value, index) => value === reference.pixels[index]),
      'Flash changed visible pixels before the real frame');
    engine.setMaterial(block.material, { color: [0, 0, 0], emissive: flash, model: 'flat' });
    const flashReal = engine.produceReal(performance.now(), flashTarget);
    require(flashReal !== undefined, 'Flash real frame did not submit');
    const flashed = await waitFor(flashTarget, frame => !frame.predicted && frame.realFrameId === flashReal);
    require(flashed.pixels.some((value, index) => Math.abs(value - reference.pixels[index]) > 16),
      'The real flash did not reach the display');
    require(flashed.hud === `Tick ${flashTarget.target.tick} · real`, 'Flash HUD did not use the current snapshot');
    status.textContent = 'Prediction fixture passed';
    return { json: { buildId, facts: { adapter, hardware, browser: navigator.userAgent },
      size: [width, height], sourceFrameId: source, targetReal, flashReal, maximumError, movingPixels, holes: guides.holes,
      refreshedLightingPixels: guides.lightPixels, predictionHud: predicted.hud, flashHud: hud.textContent,
      realSubmissions: engine.stats().frames }, files: [] };
  } catch (error) {
    status.textContent = error.message;
    throw error;
  } finally {
    engine?.dispose();
    running = false;
    button.disabled = false;
  }
}

window.nmcGpu = { ops: { prediction: input => run(input?.args ?? input) } };
button.addEventListener('click', () => run().catch(console.error));
