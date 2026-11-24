const prefix = 'spherical-';

export const sphericalOptions = [
  { id: 'gi-grid', label: 'Indirect grid', value: 'volume', choices: [
    ['volume', 'Volume'], ['spherical', 'Spherical'],
  ] },
  { id: 'spherical-subdivision', label: 'Spherical subdivision', value: 3, min: 0, max: 6, step: 1 },
  { id: 'spherical-exponent', label: 'Radial exponent', value: 0.6, min: 0, max: 2, step: 0.1 },
  { id: 'spherical-clip-start', label: 'Radial clip start (m)', value: 0.05, min: 0, max: 4, step: 0.05 },
  { id: 'spherical-z-max', label: 'Outer radius (m)', value: 12, min: 5, max: 64, step: 0.25 },
  { id: 'spherical-debug', label: 'Spherical debug', value: 'off', choices: [
    ['off', 'Off'], ['boundaries', 'Boundaries'], ['occupancy', 'Occupancy'],
    ['age', 'Age'], ['light-hash', 'Light hash'],
  ] },
];

export function isSphericalOption(id) {
  return id === 'gi-grid' || id.startsWith(prefix);
}

function available(settings) {
  return settings.gi && !settings['path-reference'] && !settings['model-assisted-path'];
}

export function applySpherical(engine, state, settings) {
  const enabled = available(settings) && settings['gi-grid'] === 'spherical';
  if (enabled) {
    engine.setSphericalGi({
      enabled,
      sphereSubdivision: settings['spherical-subdivision'],
      exponentialPower: settings['spherical-exponent'],
      clipStart: settings['spherical-clip-start'],
      zMax: settings['spherical-z-max'],
      debug: settings['spherical-debug'],
    });
  } else if (state.sphericalActive) {
    engine.setSphericalGi({ enabled: false });
  }
  state.sphericalActive = enabled;
}

export function syncSphericalControls(controls, state) {
  const applicable = available(state.enabled);
  const selected = state.values['gi-grid'] === 'spherical';
  const selector = controls.get('gi-grid');
  selector.disabled = !applicable;
  selector.title = applicable ? '' : 'Requires raster shading and global illumination.';
  for (const { id } of sphericalOptions.slice(1)) {
    controls.get(id).disabled = !applicable || !selected;
  }
}
