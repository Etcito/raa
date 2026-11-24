import assert from 'node:assert/strict';
import test from 'node:test';
import { syncSphericalControls } from './lib/gallery-spherical.js';
import { interactiveInterior } from './lib/interactive-interior.js';

test('the fitted gallery switches the live grid and restores raster fallback', () => {
  const { options, setOption } = interactiveInterior.default;
  const state = { enabled: {}, values: {}, screenSpaceQuality: 'low', isolated: false, secondaryCache: true };
  const controls = new Map(options.map(({ id, enabled, value }) => {
    if (value === undefined) state.enabled[id] = enabled;
    else state.values[id] = value;
    return [id, {}];
  }));
  const live = {};
  const engine = {
    setEnvironmentLighting() {},
    setGlobalIllumination(settings) { live.volume = settings.enabled; },
    setSphericalGi(settings) { live.spherical = settings.enabled; },
    setMeshTracing(settings) { live.mode = settings.pathMode; },
    passes: () => ['shadow.csm', 'prepass', 'light.tiles', 'forward'].map(name => ({ name, enabled: true })),
  };
  const select = (id, value) => setOption(engine, state, id, value);
  select('model-assisted-path', true);
  assert.deepEqual(live, { mode: 'learned', volume: true });
  syncSphericalControls(controls, state);
  assert.equal(controls.get('gi-grid').disabled, false);

  select('gi-grid', 'spherical');
  assert.deepEqual(live, { mode: 'learned', volume: false, spherical: true });
  syncSphericalControls(controls, state);
  assert.equal(controls.get('spherical-subdivision').disabled, false);

  select('gi-grid', 'volume');
  assert.deepEqual(live, { mode: 'learned', volume: true, spherical: false });
  select('gi-grid', 'spherical');
  select('model-assisted-path', false);
  assert.deepEqual(live, { mode: 'off', volume: true, spherical: true });

  engine.setGlobalIllumination({ enabled: false });
  select('gi-grid', 'volume');
  assert.deepEqual(live, { mode: 'off', volume: false, spherical: false });
  select('gi-grid', 'spherical');
  assert.deepEqual(live, { mode: 'off', volume: false, spherical: true });

  select('path-reference', true);
  syncSphericalControls(controls, state);
  assert.equal(controls.get('gi-grid').disabled, true);
  assert.deepEqual(live, { mode: 'off', volume: true, spherical: false });
});
