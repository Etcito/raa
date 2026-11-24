import init, { Engine } from './raa/raa.js';
import { loadEngine } from './ege/loader.js';
import { RollbackBinding } from './ege/rollback.js';
import { predictionInputs, predictionLabel, predictionPlanes, scorePrediction } from './training.js';

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
      const center = x + 0.5;
      const visible = row && center >= to && center < to + actorWidth;
      const oldActor = row && center >= from && center < from + actorWidth;
      const hole = oldActor && !visible;
      // Independent target change inside the actor. The fixture supplies this sample
      // even though its geometric correspondence is valid.
      const lighting = visible && center < to + 1;
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
async function run({ buildId, trajectory, model } = {}) {
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
    let cutoffMs;
    function snapshot(event, discontinuities = [], prepare) {
      const tick = simEngine.tick(world);
      const receivedAtMs = performance.now();
      simulation.stepCodes([0, 0]);
      prepare?.();
      cutoffMs = performance.now();
      return simulation.capturePrediction(1, cutoffMs, [0, 1].map(player => ({
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
    function gradientBlock(stops) {
      const hw = (actorWidth - 1) / (2 * height), hh = actorHeight / (2 * height);
      const node = engine.addNode(null);
      const material = engine.addMaterial({ color: [1, 1, 1], model: 'flat' });
      const mesh = engine.addMesh({
        positions: new Float32Array([-hw, -hh, 0.02, hw, -hh, 0.02, hw, hh, 0.02, -hw, hh, 0.02]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
        colors: new Float32Array([...stops[0], 1, ...stops[1], 1, ...stops[1], 1, ...stops[0], 1]),
      });
      engine.addInstance(node, mesh, material);
      return { node, material };
    }
    const block = trajectory ? gradientBlock(trajectory.stops)
      : create([(actorWidth - 1) / height, actorHeight / height, 0.04], actor);
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
    const from = trajectory?.from ?? firstX, to = trajectory?.to ?? nextX;
    move(from);
    const source = engine.produceReal(performance.now(), first);
    require(source !== undefined, 'First real frame did not submit');
    const start = await waitFor(first, frame => frame.realFrameId === source);
    const sourceTaps = trajectory && await engine.captureTrainingTaps(source);
    const sourceAvailableMs = performance.now();
    let guides, inputs, guidesAvailableMs;
    const target = snapshot('move', [], () => {
      guides = targetGuides(from, to);
      if (trajectory) inputs = predictionInputs(sourceTaps, guides, width, height);
      guidesAvailableMs = performance.now();
    });
    const targetCutoffMs = cutoffMs;
    require(target.inputs.confirmedMask.every(Boolean), 'Target snapshot must contain cutoff input receipts');
    const submit = (snapshot, limit) => engine.submitPrediction(snapshot, source, performance.now() + 1500,
      20, 0.0005, limit, guides.depth, guides.motion, guides.changed, guides.repair);
    require(!engine.submitPrediction(target, source, performance.now() - 1, 20, 0.0005,
      guides.holes + guides.lightPixels, guides.depth, guides.motion, guides.changed, guides.repair),
    'An expired deadline admitted prediction');
    const missed = present(target);
    require(missed?.realFrameId === source && !missed.predicted, 'A missed deadline did not present the causal real image');
    const needed = guides.holes + guides.lightPixels;
    async function checkOverflow() {
      require(submit(target, needed - 1), 'Budget-overflow fixture did not submit');
      for (let attempt = 0; attempt < 100; attempt++) {
        await opportunity();
        require(!present(target)?.predicted, 'An incomplete repair reached the display');
        if (submit(target, needed)) return;
      }
      throw new Error('Valid prediction was not admitted after overflow fallback');
    }
    await checkOverflow();
    const predicted = await waitFor(target, frame => frame.predicted);
    require(engine.stats().frames === 1, 'Prediction advanced real history');
    require(predicted.hud === `Tick ${target.target.tick} · prediction`, 'Prediction HUD did not use the current snapshot');

    let learnedProof;
    if (model) {
      const registration = engine.setGeneratedFrameModel(model.manifest, model.weights);
      require(registration.weightsSha256 === model.digest && registration.kind === 'windows-capture-fit',
        'Generated model receipt mismatch');
      await checkOverflow();
      const learned = await waitFor(target, frame => frame.predicted);
      let warpedChanged = 0, freshPreserved = 0, maxDisplayDelta = 0;
      for (let p = 0; p < width * height; p++) {
        const lane = p * 4;
        const delta = Math.max(...[0, 1, 2, 3].map(c => Math.abs(predicted.pixels[lane + c] - learned.pixels[lane + c])));
        maxDisplayDelta = Math.max(maxDisplayDelta, delta);
        if (inputs.fresh[lane + 3] === 1) {
          require(delta === 0, `Learned display changed fresh pixel ${p}`);
          freshPreserved++;
        } else if (delta > 0) {
          warpedChanged++;
        }
      }
      require(warpedChanged > 0 && freshPreserved === needed, 'Learned GPU A/B produced no eligible change or skipped repair');
      require(engine.stats().frames === 1, 'Learned prediction advanced real history');
      engine.clearGeneratedFrameModel();
      require(submit(target, needed), 'Analytic prediction was refused after disabling the model');
      const disabled = await waitFor(target, frame => frame.predicted);
      require(disabled.pixels.every((value, i) => value === predicted.pixels[i]), 'Analytic prediction changed after disabling the model');
      learnedProof = { ...registration, warpedChanged, freshPreserved, maxDisplayDelta,
        analyticRestored: true, incompleteRepairBypassed: true, generatedAdvancedRealHistory: false };
    }

    // Render the same target only after selecting prediction. These pixels cannot feed the candidate.
    const targetRenderStartedMs = performance.now();
    move(to);
    engine.setMaterial(strip.material, { color: [0, 0, 0], emissive: stripe, model: 'flat' });
    const targetReal = engine.produceReal(performance.now(), target);
    require(targetReal !== undefined, 'Target real frame did not submit');
    const reference = await waitFor(target, frame => !frame.predicted && frame.realFrameId === targetReal);
    const label = trajectory && predictionLabel(await engine.captureTrainingTaps(targetReal));
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
    if (!trajectory) {
      require(maximumError <= 3, `Predicted pixels differ from the full target: max ${maximumError} at ${worstPixel % width},${Math.floor(worstPixel / width)}. Predicted ${rgba(predicted)}; real ${rgba(reference)}; source ${rgba(start)}`);
    }

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
    const files = [];
    const packet = trajectory && !model && {
      id: trajectory.id, sourceRealFrameId: source, sourceAgeRealFrames: targetReal - source,
      sourceAvailableMs, guidesAvailableMs, cutoffMs: targetCutoffMs, targetRenderStartedMs,
      targetRealFrameId: targetReal, analytic: scorePrediction(inputs, label), sourceSnapshot: first, acceptedTarget: target,
      sourcePositionPx: from, targetPositionPx: to, actorStops: trajectory.stops,
      planes: await predictionPlanes(files, trajectory.id, { ...inputs, label }, width, height),
    };
    return { json: { buildId, facts: { adapter, hardware, browser: navigator.userAgent },
      size: [width, height], sourceFrameId: source, targetReal, flashReal, maximumError, movingPixels, holes: guides.holes,
      refreshedLightingPixels: guides.lightPixels, predictionHud: predicted.hud, flashHud: hud.textContent,
      realSubmissions: engine.stats().frames, ...(learnedProof && { learnedProof }), ...(packet && { packet }) }, files };
  } catch (error) {
    status.textContent = error.message;
    throw error;
  } finally {
    engine?.dispose();
    running = false;
    button.disabled = false;
  }
}

async function captureDataset({ buildId }) {
  const schedule = [
    { id: 'short-right', split: 'train', from: 16, to: 21.375, stops: [[0.12, 0.02, 0.07], [0.82, 0.22, 0.03]] },
    { id: 'wide-right', split: 'train', from: 20.125, to: 29.5, stops: [[0.08, 0.45, 0.24], [0.49, 0.04, 0.08]] },
    { id: 'short-left', split: 'test', from: 38.25, to: 31, stops: [[0.55, 0.14, 0.02], [0.02, 0.21, 0.47]] },
    { id: 'wide-left', split: 'test', from: 42, to: 32.375, stops: [[0.06, 0.18, 0.45], [0.42, 0.06, 0.62]] },
  ];
  const sequences = [], files = [];
  let facts;
  for (const trajectory of schedule) {
    const result = await run({ buildId, trajectory });
    const { packet, ...proof } = result.json;
    facts = proof.facts;
    require(packet.sourceAvailableMs <= packet.cutoffMs && packet.guidesAvailableMs <= packet.cutoffMs
      && packet.cutoffMs < packet.targetRenderStartedMs, 'Predictive evidence crossed its input cutoff');
    sequences.push({ id: trajectory.id, split: trajectory.split, groupId: trajectory.id, frames: [packet], proof });
    files.push(...result.files);
  }
  return { json: { format: 'raa.generated-frame-capture/1', buildId, facts, size: [width, height],
    target: 'separate fully shaded linear HDR submission after the accepted cutoff',
    warp: 'causal fixture warp from retained HDR and admitted geometry; comparison against actual GPU presentation is in proof',
    sequences }, files };
}

async function compareLearned({ buildId }) {
  const manifest = await asset('./model/manifest.json');
  const { weightsSha256: digest } = JSON.parse(manifest);
  const response = await fetch(new URL('./model/weights.fp32le', import.meta.url));
  if (!response.ok) throw new Error(`Generated model weights: HTTP ${response.status}`);
  const weights = new Uint8Array(await response.arrayBuffer());
  const trajectory = { id: 'wide-left', from: 42, to: 32.375, stops: [[0.06, 0.18, 0.45], [0.42, 0.06, 0.62]] };
  return run({ buildId, trajectory, model: { manifest, weights, digest } });
}

window.nmcGpu = { ops: { prediction: input => {
  const args = input?.args ?? input;
  return args?.dataset ? captureDataset(args) : args?.learned ? compareLearned(args) : run(args);
} } };
button.addEventListener('click', () => run().catch(console.error));
