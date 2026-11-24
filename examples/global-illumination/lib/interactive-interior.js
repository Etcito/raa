import { box, roomShell, sphere, volumeRoom } from './content.js';

const identity = [0, 0, 0, 1];
const unitScale = [1, 1, 1];
const options = [
  { id: 'gi', label: 'Global illumination', enabled: true },
  { id: 'area-lights', label: 'Area lights', enabled: true },
  { id: 'moving-objects', label: 'Moving objects', enabled: true },
  { id: 'ssr', label: 'Screen-space reflections', enabled: true },
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
  title: 'Performance · 1× sampling, low screen GI',
  engine: { msaa: 1 },
  screenSpaceQuality: 'low',
};

const fixtures = [
  { position: [-1.55, 2.62, -3.4], size: [0.65, 1.45], color: [1, 0.79, 0.57], intensity: 18 },
  { position: [1.45, 2.62, -5.65], size: [0.8, 1.7], color: [0.59, 0.79, 1], intensity: 23 },
];

function setGi(engine, enabled, screenSpaceQuality) {
  engine.setGlobalIllumination({
    enabled,
    screenSpace: enabled,
    screenSpaceQuality,
    reflections: enabled,
    localLightShadows: true,
    environmentDiffuse: 0.1,
    historyMax: 16,
  });
}

function fixtureSurface(color, enabled) {
  return {
    model: 'flat',
    color: [0.14, 0.15, 0.17],
    emissive: enabled ? color.map(channel => channel * 2) : [0.025, 0.025, 0.025],
  };
}

function build(engine, _lib, selected = variant) {
  const room = volumeRoom();
  const state = {
    enabled: Object.fromEntries(options.map(option => [option.id, option.enabled])),
    screenSpaceQuality: selected.screenSpaceQuality,
    fixtures: [],
    motionTime: 0,
    previousTime: undefined,
  };
  engine.setSky({ preset: 'day' });
  engine.setWeather({ preset: 'clear' });
  engine.setLight({ ...room.sun, intensity: 1.2 });
  engine.setAmbient({ color: [0.65, 0.69, 0.77], intensity: 0.06 });
  engine.setEnvironmentLighting({ diffuse: 0.18, specular: 0.25 });
  engine.setGiQuality({ resolution: 64, levels: 3, probeRays: 8 });
  engine.setAreaLightShading({ mode: 'ltc' });
  engine.setAreaLightShadows({ tier: 'hero', heroCount: fixtures.length });
  engine.setSsr({ intensity: 0.65, maxRoughness: 0.35 });
  setGi(engine, state.enabled.gi, state.screenSpaceQuality);

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
  const floor = material([0.38, 0.36, 0.33], 0.28);
  const wood = material([0.37, 0.23, 0.13]);
  const dark = material([0.12, 0.15, 0.19]);
  const ceramic = material([0.88, 0.81, 0.68], 0.25);
  const copper = material([0.82, 0.43, 0.23], 0.31, 0.6);
  const surfaces = [floor, plaster, plaster, material(room.colors.red), material(room.colors.green)];
  const shell = roomShell(room.room);
  instance({
    ...shell,
    subMeshes: shell.subMeshes.map((part, index) => ({ ...part, material: surfaces[index] })),
  });

  solid([0, 1.5, 0.2], [3.4, 1.5, 0.2], plaster);
  solid([-2.55, 0.55, -4.5], [0.42, 0.55, 1.1], wood);
  solid([2.32, 0.39, -4.9], [0.55, 0.39, 0.85], dark);
  solid([0.7, 0.24, -7.03], [1.9, 0.24, 0.57], wood);
  solid([-1.72, 0.98, -6.3], [0.12, 0.98, 0.85], plaster);
  solid([0.9, 1.58, -7.89], [1.24, 0.49, 0.06], dark);
  for (const x of [0.08, 0.9, 1.72]) {
    solid([x, 1.58, -7.8], [0.31, 0.39, 0.045], copper);
  }

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
  return state;
}

function setOption(engine, state, id, enabled) {
  if (!Object.hasOwn(state.enabled, id) || typeof enabled !== 'boolean') {
    throw new TypeError(`Invalid interactive interior option: ${id}`);
  }
  if (state.enabled[id] === enabled) return;
  switch (id) {
    case 'gi':
      setGi(engine, enabled, state.screenSpaceQuality);
      break;
    case 'area-lights':
      for (const { light, intensity, surface, color } of state.fixtures) {
        engine.updateLight(light, { intensity: enabled ? intensity : 0 });
        engine.setSurfaceMaterial(surface, fixtureSurface(color, enabled));
      }
      break;
    case 'ssr':
    case 'dof':
    case 'bloom':
      engine.setPassEnabled(id, enabled);
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
    frame,
    setOption,
  },
  inventory: [{ mode: variant.id }],
};
