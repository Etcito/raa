import { buildSkySample, sampleSelection } from '../lib/scene.js';

const camera = { eye: [0, 2.6, 9], target: [0, 1.9, -8], fovYDeg: 47, near: .1, far: 300 };
const behavior = 'A readable blue daytime sky fading to a warm pale horizon above a broad neutral ground. '
  + 'A terracotta matte sphere stands on the left and a large smooth chrome sphere on the right '
  + 'reflects the blue sky and horizon. Three distant block forms give depth behind the spheres. '
  + 'Both spheres are fully visible, with clear silhouettes and different matte and reflective surfaces. '
  + 'The status below the view reads ready, with no displayed failure.';

export const factNames = ['sky-radiance-positive', 'sky-passes-enabled', 'sample-submitted'];
export const inventory = [{
  id: 'foundation-sky-ground-and-reflective-sphere', criterionName: 'Foundation sky and material references',
  scenario: 'foundation', mode: 'day', filename: 'foundation-sky-ground-and-reflective-sphere.png',
  expectedVisibleBehavior: behavior,
  requiredFacts: ['snapshot:sky', 'runtime:passes', 'runtime:viewSizes', 'runtime:ready', ...factNames.map(name => `assertion:${name}`)],
  pair: null,
}];

// Read only landed getters. Pixels establish appearance; these assertions establish
// the submitted scene and enabled sky pipeline, without claiming future diagnostics.
export async function observe(engine, variant) {
  const sky = engine.sky();
  const passes = engine.passes();
  const stats = engine.stats();
  return {
    sky, time: null, weather: null, diagnostics: null,
    assertions: [
      { name: 'sky-radiance-positive', passed: Number.isFinite(sky?.intensity) && sky.intensity > 0, detail: `sky().intensity = ${sky?.intensity}` },
      { name: 'sky-passes-enabled', passed: ['sky.env', 'sky.background'].every(name => passes.some(pass => pass.name === name && pass.enabled === true)), detail: JSON.stringify(passes.filter(pass => pass.name.startsWith('sky.'))) },
      { name: 'sample-submitted', passed: Number.isSafeInteger(stats.frames) && stats.frames > 0 && stats.instances === variant.selection.subjects.length && stats.triangles > 0, detail: `Submitted frames ${stats.frames}; instances ${stats.instances}; triangles ${stats.triangles}` },
    ],
    edits: [],
  };
}

const controls = [
  { id: 'preset', label: 'Apply sky preset', kind: 'select', unit: '',
    choices: [{ value: '', label: 'Choose preset…' }, ...['day', 'snow', 'jungle', 'night'].map(value => ({ value, label: value }))],
    // The landed getter has no preset identity. This is an action chooser; the
    // numeric fields below display the resulting accepted settings.
    read: () => '',
    apply: (engine, value) => engine.setSky({ preset: value }) },
  { id: 'intensity', label: 'Sky intensity', kind: 'number', unit: '×', choices: [],
    read: engine => engine.sky().intensity, apply: (engine, value) => engine.setSky({ intensity: value }) },
  { id: 'coverage', label: 'Cloud coverage', kind: 'number', unit: '0–1', choices: [],
    read: engine => engine.sky().clouds.coverage, apply: (engine, value) => engine.setSky({ clouds: { coverage: value } }) },
  ...['sky.background', 'sky.env'].map(name => ({
    id: name === 'sky.env' ? 'ibl' : 'background', label: name === 'sky.env' ? 'Sky lighting / reflections' : 'Sky background',
    kind: 'checkbox', unit: '', choices: [],
    read: engine => engine.passes().find(pass => pass.name === name)?.enabled === true,
    apply: (engine, value) => engine.setPassEnabled(name, value),
  })),
];

export default {
  id: 'foundation', milestone: 'foundation', title: 'Sky, ground and material references',
  engine: { msaa: 4, toneMapper: 'agx', exposureEv: 0 },
  variants: [{ id: 'day', description: behavior, references: [], judge: true, timeMs: 1000, selection: sampleSelection(camera) }],
  controls,
  async build(engine, lib, variant) {
    engine.setBackground([.05, .06, .08]);
    engine.setLight({ direction: [-.35, -.6, -.55], color: [1, .93, .82], intensity: 3 });
    engine.setAmbient({ color: [.55, .62, .75], intensity: .25 });
    engine.setSky({ preset: 'day', zenith: [.025, .16, .65], horizon: [.85, .65, .42], intensity: 1.2, horizonSoftness: 2, clouds: { coverage: .25, density: 1 } });
    engine.setEnvironmentLighting({ diffuse: 1, specular: 1 });
    await buildSkySample(engine, lib, variant.selection.camera);
  },
};
