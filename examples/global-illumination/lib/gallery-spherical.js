const prefix = 'spherical-';

export const sphericalOptions = [
  { id: 'gi-grid', label: 'Indirect grid', value: 'volume', choices: [
    ['volume', 'Volume'], ['spherical', 'Spherical'],
  ] },
  { id: 'spherical-subdivision', label: 'Spherical subdivision', value: 3, min: 0, max: 6, step: 1 },
  { id: 'spherical-exponent', label: 'Radial exponent', value: 0.6, min: 0, max: 2, step: 0.1 },
  { id: 'spherical-clip-start', label: 'Radial clip start (m)', value: 0.05, min: 0, max: 4, step: 0.05 },
  { id: 'spherical-z-max', label: 'Outer radius (m)', value: 12, min: 5, max: 64, step: 0.25 },
  { id: 'spherical-mirror-cache', label: 'Secondary-hit cache', value: 'auto', choices: [
    ['auto', 'Automatic'], ['off', 'Off'],
  ] },
  { id: 'spherical-mirror-lighting', label: 'Secondary-hit lighting groups', value: '7', choices: [
    ['7', 'All'], ['1', 'Sky'], ['2', 'Sun'], ['4', 'Local / other'], ['0', 'None'],
  ] },
  { id: 'spherical-debug', label: 'Spherical debug overlay', value: 'off', choices: [
    ['off', 'Off'], ['boundaries', 'Boundaries'], ['occupancy', 'Occupancy'],
    ['age', 'Age'], ['light-hash', 'Light hash'],
  ] },
];

export function isSphericalOption(id) {
  return id === 'gi-grid' || id.startsWith(prefix);
}

function available(settings, state) {
  return settings.gi && !settings['path-reference'] && (!settings['model-assisted-path'] || state.secondaryCache);
}

function sphericalEnabled(settings) {
  return available(settings) && settings['gi-grid'] === 'spherical';
}

export function usesVolumeGrid(settings) {
  // Keep the raster coverage fallback. The fitted path selects one actual grid.
  return settings.gi && !(settings['model-assisted-path'] && sphericalEnabled(settings));
}

export function applySpherical(engine, state, settings) {
  const enabled = sphericalEnabled(settings);
  if (enabled) {
    engine.setSphericalGi({
      enabled,
      sphereSubdivision: settings['spherical-subdivision'],
      exponentialPower: settings['spherical-exponent'],
      clipStart: settings['spherical-clip-start'],
      zMax: settings['spherical-z-max'],
      ...(state.secondaryCache && {
        mirrorCache: settings['spherical-mirror-cache'],
        mirrorChannelMask: Number(settings['spherical-mirror-lighting']),
      }),
      debug: settings['spherical-debug'],
    });
  } else if (state.sphericalActive) {
    engine.setSphericalGi({ enabled: false });
  }
  state.sphericalActive = enabled;
}

export function syncSphericalControls(controls, state) {
  const applicable = available(state.enabled, state);
  const selected = state.values['gi-grid'] === 'spherical';
  const selector = controls.get('gi-grid');
  selector.disabled = !applicable;
  selector.title = applicable ? '' : 'Requires global illumination outside the path reference.';
  for (const { id } of sphericalOptions.slice(1)) {
    const downstream = id.startsWith('spherical-mirror-');
    const tracing = state.enabled['geometry-reflections'] || state.enabled['model-assisted-path'];
    controls.get(id).disabled = !applicable || !selected || (downstream && (!tracing || !state.secondaryCache))
      || (id === 'spherical-mirror-lighting' && state.values['spherical-mirror-cache'] === 'off');
  }
}
