import { box, enclosedRoomShell, panel, sphere, volumeRoom } from './content.js';

import { loadGalleryFit } from './gallery-path-model.js';
import { lightingOptions } from './gallery-lighting.js';
import { applySpherical, isSphericalOption, sphericalOptions } from './gallery-spherical.js';

const identity = [0, 0, 0, 1];
const unitScale = [1, 1, 1];
const skylightWidth = 2;
const shellThickness = 0.8;
const mirrorCenter = [0.45, 1.5, -7.86];
const mirrorSize = [4.25, 2.25];
const entryExhibit = [2.15, 1.28, -1.15];
const options = [
  { id: 'gi', label: 'Global illumination', enabled: true },
  { id: 'screen-gi', label: 'Screen-space GI detail', enabled: true },
  { id: 'area-lights', label: 'Area lights', enabled: true },
  { id: 'moving-objects', label: 'Moving objects', enabled: true },
  { id: 'geometry-reflections', label: 'Hybrid geometry reflections', enabled: false },
  { id: 'reflection-bounces', label: 'Reflection bounces', value: 2, min: 1, max: 6, step: 1 },
  { id: 'reflection-confidence', label: 'Screen confidence threshold', value: 0.55, min: 0, max: 1, step: 0.01 },
  { id: 'model-assisted-path', label: 'Model-assisted path tracing', enabled: false,
    unavailable: state => state?.pathModelError },
  { id: 'path-reference', label: 'Path-traced reference', enabled: false },
  { id: 'path-bounces', label: 'Reference path bounces', value: 2, min: 0, max: 6, step: 1 },
  ...lightingOptions,
  ...sphericalOptions,
  { id: 'ssr', label: 'SSR on other surfaces', enabled: true },
  { id: 'dof', label: 'Depth of field', enabled: true },
  { id: 'bloom', label: 'Bloom', enabled: true },
];
const variant = {
  id: 'live',
  title: 'Quality · 4× MSAA, medium screen GI',
  engine: { msaa: 4 },
  screenSpaceQuality: 'medium',
  selection: {
    camera: { eye: [0.1, 1.65, -0.35], target: [0, 1.32, -5.8], fovYDeg: 62, near: 0.05, far: 40 },
  },
};

const performanceVariant = {
  ...variant,
  id: 'performance',
  title: 'Performance · probe GI, 1× sampling',
  effects: { 'screen-gi': false, dof: false, ssr: false },
  engine: { msaa: 1 },
  screenSpaceQuality: 'low',
};

const fixtures = [
  { position: [-1.9, 2.62, -3.4], size: [0.65, 1.45], color: [1, 0.79, 0.57], intensity: 18 },
  { position: [1.45, 2.62, -5.65], size: [0.8, 1.7], color: [0.59, 0.79, 1], intensity: 23 },
];

function setGi(engine, state, update = {}) {
  const settings = { ...state.enabled, ...state.values, ...update };
  const enabled = settings.gi;
  // The GI rays sample the authored sky. Keep the unoccluded IBL fill off,
  // including when GI is disabled for a direct-light comparison.
  engine.setEnvironmentLighting({ diffuse: enabled ? 0.65 : 0, specular: 0 });
  engine.setGlobalIllumination({
    enabled,
    screenSpace: enabled && settings['screen-gi'],
    screenSpaceQuality: state.screenSpaceQuality,
    indirect: settings['indirect-solver'],
    reflections: enabled,
    localLightShadows: true,
    environmentDiffuse: 0,
    historyMax: 16,
  });
  applySpherical(engine, state, settings);
}

function fixtureSurface(color, enabled) {
  return {
    model: 'flat',
    color: [0.14, 0.15, 0.17],
    emissive: enabled ? color.map(channel => channel * 2) : [0, 0, 0],
  };
}

function assertForwardOrder(engine) {
  const passes = engine.passes().filter(pass => pass.enabled).map(pass => pass.name);
  const forward = passes.indexOf('forward');
  if (forward < 0
      || ['shadow.csm', 'prepass', 'light.tiles'].some(pass => {
        const index = passes.indexOf(pass);
        return index < 0 || index >= forward;
      })
      || ['ssr', 'taa', 'tonemap'].some(pass => passes.includes(pass) && passes.indexOf(pass) <= forward)) {
    throw new Error('Geometry reflections require sun shadows, depth and light tiles before forward, with post effects afterward.');
  }
}

function setTracing(engine, state, update = {}) {
  const values = { ...state.enabled, ...state.values, ...update };
  const reference = values['path-reference'];
  const pathMode = values['model-assisted-path'] ? 'learned' : 'off';
  const requested = values['lighting-view'];
  const lightingOutput = !reference && requested === 'bounce' ? 'combined' : requested;
  const isolated = lightingOutput !== 'combined';
  const raster = !reference && pathMode === 'off';
  const reflections = values['geometry-reflections'] && !(raster && isolated);
  if (reflections || reference || pathMode === 'learned') assertForwardOrder(engine);
  engine.setMeshTracing({
    reflections, reference, pathMode, lightingOutput,
    lightingDepth: values['bounce-depth'],
    reflectionBounces: values['reflection-bounces'],
    confidenceThreshold: values['reflection-confidence'],
    pathBounces: values['path-bounces'],
  });
  if (isolated !== state.isolated) {
    for (const pass of ['ssr', 'dof', 'bloom']) engine.setPassEnabled(pass, !isolated && values[pass]);
    if (isolated) state.taaEnabled = engine.passes().find(pass => pass.name === 'taa')?.enabled;
    if (state.taaEnabled !== undefined) engine.setPassEnabled('taa', !isolated && state.taaEnabled);
    state.isolated = isolated;
  }
  return lightingOutput;
}

function build(engine, _lib, selected = variant) {
  const room = volumeRoom();
  const skylightBlocked = selected.skylightBlocked === true;
  const state = {
    skylightBlocked,
    enabled: Object.fromEntries(options.filter(option => option.value === undefined)
      .map(option => [option.id, selected.effects?.[option.id] ?? option.enabled])),
    values: Object.fromEntries(options.filter(option => option.value !== undefined)
      .map(option => [option.id, selected.effects?.[option.id] ?? option.value])),
    screenSpaceQuality: selected.screenSpaceQuality,
    isolated: false,
    fixtures: [],
    motionTime: 0,
    previousTime: undefined,
  };
  engine.setSky({ preset: 'day' });
  engine.setWeather({ preset: 'clear' });
  engine.setPassEnabled('sunshafts', false);
  engine.setLight({ ...room.sun, intensity: 5 });
  engine.setAmbient({ color: [0, 0, 0], intensity: 0 });
  engine.setShadows({ distance: 20 });
  engine.setGiQuality({ resolution: 64, levels: 3, probeRays: 8 });
  engine.setAreaLightShading({ mode: 'ltc' });
  engine.setAreaLightShadows({ tier: 'hero', heroCount: fixtures.length });
  engine.setSsr({ intensity: 0.65, maxRoughness: 0.35 });
  for (const pass of ['dof', 'ssr', 'bloom']) engine.setPassEnabled(pass, state.enabled[pass]);
  setGi(engine, state);

  const material = (color, roughness = 0.78, metallic = 0) =>
    engine.addSurfaceMaterial({ color, roughness, metallic });
  function instance(mesh, position = [0, 0, 0]) {
    const node = engine.addNode(null, position, identity, unitScale);
    return { node, id: engine.addMeshInstance(node, engine.addMesh(mesh)) };
  }
  const withSurface = (mesh, surface) => ({
    ...mesh, subMeshes: [{ first: 0, count: mesh.indices.length, material: surface }],
  });
  const solid = (position, half, surface) => instance(withSurface(box([0, 0, 0], half), surface), position);

  const plaster = material(room.colors.white);
  const floor = material([0.55, 0.52, 0.48], 0.28);
  const wood = material([0.37, 0.23, 0.13]);
  const dark = material([0.12, 0.15, 0.19]);
  const ceramic = material([0.88, 0.81, 0.68], 0.25);
  const copper = material([0.82, 0.43, 0.23], 0.31, 0.6);
  const surfaces = [floor, plaster, plaster, material(room.colors.red), material(room.colors.green), plaster];
  // A solid shell keeps the two sides of the walls distinct in the GI clipmap.
  const shell = enclosedRoomShell({
    ...room.room,
    opening: skylightBlocked ? 0 : skylightWidth,
  }, shellThickness);
  // The physical regression can author the same wall colors through the
  // supported vertex stream, without changing the gallery's normal materials.
  if (selected.vertexWallColors) {
    const white = material([1, 1, 1]);
    const colors = new Float32Array(shell.positions.length / 3 * 4).fill(1);
    const linear = value => value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
    for (const [surface, color] of [[3, room.colors.red], [4, room.colors.green]]) {
      const { first, count } = shell.subMeshes[surface];
      const rgba = [...color.map(linear), 1];
      for (let i = first; i < first + count; i++) colors.set(rgba, shell.indices[i] * 4);
      surfaces[surface] = white;
    }
    shell.colors = colors;
  }
  instance({
    ...shell,
    subMeshes: shell.subMeshes.map((part, index) => ({ ...part, material: surfaces[index] })),
  });

  solid([-2.55, 0.55, -4.5], [0.42, 0.55, 1.1], wood);
  solid([2.32, 0.39, -4.9], [0.55, 0.39, 0.85], dark);
  solid([0.7, 0.24, -7.03], [1.9, 0.24, 0.57], wood);
  solid([-1.72, 0.98, -6.3], [0.12, 0.98, 0.85], plaster);
  const [mirrorX, mirrorY, mirrorZ] = mirrorCenter;
  const [width, height] = mirrorSize;
  const frameWidth = 0.065;
  solid([mirrorX, mirrorY, mirrorZ - 0.055], [width / 2, height / 2, 0.045], dark);
  for (const side of [-1, 1]) {
    solid([mirrorX + side * (width / 2 + frameWidth / 2), mirrorY, mirrorZ],
      [frameWidth / 2, height / 2, 0.04], copper);
    solid([mirrorX, mirrorY + side * (height / 2 + frameWidth / 2), mirrorZ],
      [width / 2 + frameWidth, frameWidth / 2, 0.04], copper);
  }
  // The base stays visible when the capture is off. Its roughness keeps SSR
  // below its cutoff on this surface while SSR can still affect the polished floor.
  const mirrorSurface = engine.addSurfaceMaterial({
    color: [0.22, 0.28, 0.33], roughness: 0.42, metallic: 0.72, traceMirror: true,
  });
  const mirrorMesh = withSurface(panel(width, height), mirrorSurface);
  const rearMirror = instance(mirrorMesh, mirrorCenter).id;
  engine.setInstanceGlobalIllumination(rearMirror, false);

  // The entrance mirror faces the rear mirror and gives reflected rays a
  // second real surface to bounce from.
  const entryMirrorCenter = [0.55, 1.55, -0.07];
  solid([entryMirrorCenter[0], entryMirrorCenter[1], -0.025], [1.96, 0.93, 0.025], copper);
  const frontMirror = instance(withSurface(panel(3.8, 1.74), mirrorSurface), entryMirrorCenter);
  engine.setTransform(frontMirror.node, entryMirrorCenter, [0, 1, 0, 0], unitScale);
  engine.setInstanceGlobalIllumination(frontMirror.id, false);

  for (const fixture of fixtures) {
    const { position, size, color } = fixture;
    const surface = engine.addSurfaceMaterial(fixtureSurface(color, state.enabled['area-lights']));
    const housing = solid([position[0], 2.89, position[2]], [size[1] / 2, 0.055, size[0] / 2], surface);
    engine.setInstanceGlobalIllumination(housing.id, false);
    const light = engine.addLight({
      ...fixture, kind: 'area', direction: [0, -1, 0], range: 12,
      intensity: state.enabled['area-lights'] ? fixture.intensity : 0,
    });
    state.fixtures.push({ ...fixture, light, surface });
  }

  const ball = sphere(room.sphere.radius * 0.72, room.sphere.segments);
  state.sphereNode = instance(withSurface(ball, ceramic), [-1.1, 0.56, -3.2]).node;
  state.blockNode = solid([0.85, 0.36, -4.95], [0.36, 0.36, 0.46], copper).node;

  // A moving glazed exhibit beside the entry lies outside the home camera's
  // frustum. The rear mirror can see it above its plinth.
  const glaze = engine.addSurfaceMaterial({
    color: [0.04, 0.65, 0.82], roughness: 0.24, metallic: 0.08,
    emissive: [0.01, 0.28, 0.47],
  });
  solid([entryExhibit[0], 0.42, entryExhibit[2]], [0.46, 0.42, 0.38], dark);
  state.entryNode = instance(withSurface(sphere(0.32, 32), glaze), entryExhibit).node;
  if (state.enabled['geometry-reflections'] || state.enabled['path-reference']
      || state.values['lighting-view'] !== 'combined') {
    state.values['lighting-view'] = setTracing(engine, state);
  }
  return state;
}

async function prepare(engine, state) {
  try {
    const model = await loadGalleryFit();
    if (model) engine.setMeshPathModel(model.bytes);
    else state.pathModelError = 'This build has no fitted Windows gallery model.';
  } catch (error) {
    state.pathModelError = error.message;
  }
  if (state.pathModelError) state.status = state.pathModelError;
}

function setOption(engine, state, id, enabled) {
  if (Object.hasOwn(state.values, id)) {
    const option = options.find(option => option.id === id);
    const valid = option.choices
      ? option.choices.some(([value]) => value === enabled)
      : Number.isFinite(enabled) && enabled >= option.min && enabled <= option.max
        && (option.step !== 1 || Number.isInteger(enabled));
    if (!valid || (id === 'lighting-view' && enabled === 'bounce' && !state.enabled['path-reference'])) {
      throw new TypeError(`Invalid interactive interior value for ${id}`);
    }
    if (state.values[id] !== enabled) {
      if (id === 'indirect-solver') setGi(engine, state, { [id]: enabled });
      else if (isSphericalOption(id)) applySpherical(engine, state, { ...state.enabled, ...state.values, [id]: enabled });
      else if (id !== 'bounce-depth' || state.values['lighting-view'] === 'bounce') {
        setTracing(engine, state, { [id]: enabled });
      }
      state.values[id] = enabled;
    }
    return;
  }
  if (!Object.hasOwn(state.enabled, id) || typeof enabled !== 'boolean') {
    throw new TypeError(`Invalid interactive interior option: ${id}`);
  }
  if (state.enabled[id] === enabled) return;
  if (id === 'model-assisted-path' && enabled && state.pathModelError) throw new Error(state.pathModelError);
  switch (id) {
    case 'gi':
    case 'screen-gi':
      setGi(engine, state, { [id]: enabled });
      break;
    case 'area-lights':
      for (const { light, intensity, surface, color } of state.fixtures) {
        engine.updateLight(light, { intensity: enabled ? intensity : 0 });
        engine.setSurfaceMaterial(surface, fixtureSurface(color, enabled));
      }
      break;
    case 'geometry-reflections':
    case 'model-assisted-path':
    case 'path-reference':
      state.values['lighting-view'] = setTracing(engine, state, { [id]: enabled });
      if (id !== 'geometry-reflections') {
        applySpherical(engine, state, { ...state.enabled, ...state.values, [id]: enabled });
      }
      break;
    case 'ssr':
    case 'dof':
    case 'bloom':
      engine.setPassEnabled(id, enabled && !state.isolated);
      break;
    case 'moving-objects':
      state.previousTime = undefined;
      break;
  }
  state.enabled[id] = enabled;
}

function frame(engine, timeMs, _variant, state) {
  if (!state.enabled['moving-objects']) return;
  const previous = state.previousTime ?? timeMs;
  state.previousTime = timeMs;
  if (timeMs <= previous) return;
  state.motionTime += (timeMs - previous) / 1000;
  const phase = state.motionTime * 1.25;
  engine.setTransform(state.sphereNode,
    [-1.1 + 0.75 * Math.sin(phase), 0.56 + 0.32 * Math.sin(phase) ** 2, -3.2], identity, unitScale);
  const yaw = phase * 0.65;
  engine.setTransform(state.blockNode,
    [0.85 + 0.26 * Math.sin(phase * 0.8), 0.36, -4.95 + 0.7 * Math.sin(phase * 0.7)],
    [0, Math.sin(yaw / 2), 0, Math.cos(yaw / 2)], unitScale);
  engine.setTransform(state.entryNode,
    [entryExhibit[0] + 0.18 * Math.sin(phase),
      entryExhibit[1] + 0.26 * Math.sin(phase), entryExhibit[2]], identity, unitScale);
}

// Add this namespace to the picker map after managed discovery. Pass the state
// from createInteractiveView's onReady callback to setOption on the live engine.
export const interactiveInterior = {
  default: {
    id: 'interactive-interior',
    milestone: 'lighting',
    title: 'Interactive gallery interior',
    variants: [performanceVariant, variant],
    options,
    build,
    prepare,
    frame,
    setOption,
  },
  inventory: [performanceVariant, variant].map(({ id }) => ({ mode: id })),
};
