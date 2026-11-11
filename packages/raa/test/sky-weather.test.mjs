// Pure `.weather` text parsing and formatting (packages/raa/sky-weather.js). These tests need no
// Engine, GPU or wasm module: they import the data functions directly.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { formatWeatherText, parseWeatherText } from '../sky-weather.js';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, '..', '..', '..');

const FAIR = Object.freeze({
  coverage: 0.3,
  type: 0.9,
  base: 1200,
  thickness: 1600,
  density: 1,
  cirrus: 0.15,
  wind: { speed: 7, from: 250 },
  evolution: 1,
  rain: 0,
});

function withFields(fields) {
  return { ...FAIR, wind: { ...FAIR.wind }, ...fields };
}

function assertRefused(text, pattern) {
  assert.throws(
    () => parseWeatherText(text),
    (error) => {
      assert.equal(error.name, 'RaaError');
      assert.equal(error.code, 'sky.invalid');
      assert.match(error.message, /^sky\.invalid: /);
      assert.match(error.message, pattern);
      return true;
    },
  );
}

function assertFormatRefused(weather, pattern) {
  assert.throws(
    () => formatWeatherText(weather),
    (error) => {
      assert.equal(error.code, 'sky.invalid');
      assert.match(error.message, pattern);
      return true;
    },
  );
}

test('empty and comment-only text parse to the fair defaults', () => {
  assert.deepEqual(parseWeatherText(''), FAIR);
  assert.deepEqual(parseWeatherText('\n\n'), FAIR);
  assert.deepEqual(parseWeatherText('# only a comment\n   # indented\n'), FAIR);
});

test('the parser returns a fresh object on every call', () => {
  const first = parseWeatherText('');
  first.wind.speed = 99;
  assert.equal(parseWeatherText('').wind.speed, 7);
});

test('comments, whitespace and CRLF line endings are normalized', () => {
  const text = '  coverage 0.5 # half covered\r\n\n\t type\t0.25\r\n# evolution 9\r\nrain 0.5#x\r\n';
  assert.deepEqual(
    parseWeatherText(text),
    withFields({ coverage: 0.5, type: 0.25, rain: 0.5 }),
  );
});

test('the last repeated setting wins', () => {
  const text = 'coverage 0.2\ncoverage 0.7\nwind speed=3\nwind speed=4 from=90\nthickness 500\nthickness 300';
  assert.deepEqual(
    parseWeatherText(text),
    withFields({ coverage: 0.7, thickness: 300, wind: { speed: 4, from: 90 } }),
  );
});

test('wind accepts omitted components and keeps the other from the defaults', () => {
  assert.deepEqual(parseWeatherText('wind speed=3').wind, { speed: 3, from: 250 });
  assert.deepEqual(parseWeatherText('wind from=90').wind, { speed: 7, from: 90 });
  assert.deepEqual(parseWeatherText('wind').wind, { speed: 7, from: 250 });
  assert.deepEqual(parseWeatherText('wind from=-45').wind, { speed: 7, from: -45 });
});

test('the parser clamps values as the source file format does', () => {
  const parsed = parseWeatherText([
    'coverage 2', 'type -1', 'base -5', 'thickness 10', 'density -1', 'cirrus 7',
    'wind speed=-3 from=400', 'evolution -2', 'rain 4',
  ].join('\n'));
  assert.deepEqual(parsed, {
    coverage: 1,
    type: 0,
    base: 0,
    thickness: 100,
    density: 0,
    cirrus: 1,
    wind: { speed: 0, from: 400 },
    evolution: 0,
    rain: 1,
  });
});

test('unknown top-level and wind keys are refused with a one-based line', () => {
  assertRefused('coverage 0.1\nhumidity 0.5\n', /line 2: .*humidity/);
  assertRefused('\n# note\nwind speed=2 gust=4\n', /line 3: unknown wind setting gust/);
});

test('a wind word without key=value is refused', () => {
  assertRefused('wind 5', /line 1: expected key=value, found "5"/);
});

test('single-value settings take exactly one number', () => {
  assertRefused('coverage', /line 1: coverage takes one number/);
  assertRefused('coverage 0.1 0.2', /line 1: coverage takes one number/);
});

test('malformed numbers are refused with a one-based line', () => {
  for (const value of ['abc', '0x10', '1,5', '1.2.3', '+', '.', '--1', '1e']) {
    assertRefused(`base ${value}`, /line 1: .* is not a number/);
  }
  assertRefused('base', /line 1: base takes one number/);
  assertRefused('base 1 e5', /line 1: base takes one number/);
  assertRefused('cirrus 0.1\n\ncoverage 1e\n', /line 3: "1e" is not a number/);
  assertRefused('wind speed=', /line 1: "" is not a number/);
  assertRefused('wind from=ten', /line 1: "ten" is not a number/);
});

test('nonfinite numbers are refused', () => {
  for (const value of ['inf', 'Infinity', '-inf', 'NaN', 'nan']) {
    assertRefused(`rain ${value}`, /line 1: .* is not a number/);
    assertRefused(`wind from=${value}`, /line 1: .* is not a number/);
  }
});

test('values that overflow a single-precision float are refused, not clamped', () => {
  assertRefused('coverage 1e39', /line 1: "1e39" is not a number/);
  assertRefused('density 1e39', /line 1: "1e39" is not a number/);
  assertRefused('wind from=3.5e38', /line 1: "3.5e38" is not a number/);
  assertRefused('wind speed=-1e39', /line 1: "-1e39" is not a number/);
  assertRefused('evolution 1e400', /line 1: "1e400" is not a number/);
});

test('base plus thickness must stay finite as an f32', () => {
  assertRefused('base 3.4e38\nthickness 3.4e38', /base \+ thickness must be finite/);
  assert.equal(parseWeatherText('base 3.4e38').base, 3.4e38);
});

test('the text reports unknown setting names in its message', () => {
  assertRefused('clouds 3', /line 1: unknown setting "clouds"/);
});

test('non-string input is refused', () => {
  assert.throws(() => parseWeatherText(undefined), /sky\.invalid: weather text must be a string/);
  assert.throws(() => parseWeatherText(null), /sky\.invalid: weather text must be a string/);
});

test('formatting writes the authored order with a final LF', () => {
  assert.equal(
    formatWeatherText(FAIR),
    'coverage 0.3\ntype 0.9\nbase 1200\nthickness 1600\ndensity 1\ncirrus 0.15\n'
      + 'wind speed=7 from=250\nevolution 1\nrain 0\n',
  );
});

test('formatting refuses out-of-domain and malformed structured values without clamping', () => {
  assertFormatRefused(withFields({ coverage: 1.5 }), /weather\.coverage must be in \[0, 1\]/);
  assertFormatRefused(withFields({ type: -0.1 }), /weather\.type must be in \[0, 1\]/);
  assertFormatRefused(withFields({ cirrus: 2 }), /weather\.cirrus must be in \[0, 1\]/);
  assertFormatRefused(withFields({ rain: Number.NaN }), /weather\.rain must be a finite number/);
  assertFormatRefused(withFields({ base: -1 }), /weather\.base must be at least 0/);
  assertFormatRefused(withFields({ thickness: 99 }), /weather\.thickness must be at least 100/);
  assertFormatRefused(withFields({ density: -1 }), /weather\.density must be at least 0/);
  assertFormatRefused(withFields({ evolution: Infinity }), /weather\.evolution must be a finite number/);
  assertFormatRefused(withFields({ wind: { speed: -1, from: 0 } }), /weather\.wind\.speed must be at least 0/);
  assertFormatRefused(withFields({ wind: { speed: 1, from: Infinity } }), /weather\.wind\.from must be a finite number/);
  assertFormatRefused(withFields({ coverage: '0.5' }), /weather\.coverage must be a finite number/);
  assertFormatRefused(withFields({ coverage: null }), /weather\.coverage must be a finite number/);
  assertFormatRefused(withFields({ base: 1e39 }), /weather\.base must be a finite number/);
});

test('formatting refuses missing, unknown and mistyped structure', () => {
  const { rain, ...missingRain } = FAIR;
  assertFormatRefused(missingRain, /weather\.rain is missing/);
  assertFormatRefused({ ...FAIR, clouds: 1 }, /weather\.clouds is not a weather field/);
  assertFormatRefused({ ...FAIR, wind: 5 }, /weather\.wind must be an object/);
  assertFormatRefused({ ...FAIR, wind: { speed: 1 } }, /weather\.wind\.from is missing/);
  assertFormatRefused({ ...FAIR, wind: { speed: 1, from: 0, gust: 2 } }, /weather\.wind\.gust is not a weather field/);
  assertFormatRefused(null, /weather must be an object/);
  assertFormatRefused([], /weather must be an object/);
});

test('formatting refuses base plus thickness that overflows an f32', () => {
  assertFormatRefused(
    withFields({ base: 3.4e38, thickness: 3.4e38 }),
    /weather\.base \+ weather\.thickness must be finite/,
  );
});

test('a parsed text formats back to the same text, and formatted text parses to the same state', () => {
  const text = 'coverage 0.62\ntype 0.95\nbase 1000\nthickness 2600\ndensity 1.15\ncirrus 0.3\n'
    + 'wind speed=10 from=240\nevolution 1.3\nrain 0\n';
  const state = parseWeatherText(text);
  assert.equal(formatWeatherText(state), text);
  assert.deepEqual(parseWeatherText(formatWeatherText(state)), state);
});

test('round trips hold for extreme but valid states', () => {
  const states = [
    withFields({ coverage: 0, type: 1, base: 0, thickness: 100, density: 0, cirrus: 1, rain: 1 }),
    withFields({ base: 1e-7, thickness: 1e-7 + 100, evolution: 1e-30, wind: { speed: 1e21, from: -720.5 } }),
    withFields({ base: 3.4e38, thickness: 100, density: 3.4e38, wind: { speed: 0.1 + 0.2, from: 359.999 } }),
    withFields({ wind: { speed: 0, from: -0 } }),
  ];
  for (const state of states) {
    const text = formatWeatherText(state);
    assert.ok(text.endsWith('\n') && !text.endsWith('\n\n'));
    assert.deepEqual(parseWeatherText(text), JSON.parse(JSON.stringify(state)));
  }
});

test('the six delivered presets keep their values from the Solstice source files', async () => {
  const presets = {
    clear: { coverage: 0, cirrus: 0.05, wind: { speed: 5, from: 250 } },
    fair: { coverage: 0.3, type: 0.9, base: 1200, thickness: 1600, density: 1, cirrus: 0.15, wind: { speed: 7, from: 250 }, evolution: 1 },
    broken: { coverage: 0.62, type: 0.95, base: 1000, thickness: 2600, density: 1.15, cirrus: 0.3, wind: { speed: 10, from: 240 }, evolution: 1.3 },
    overcast: { coverage: 0.97, type: 0.12, base: 600, thickness: 900, density: 1.3, cirrus: 0, wind: { speed: 8, from: 220 }, evolution: 0.6 },
    storm: { coverage: 0.85, type: 1, base: 800, thickness: 7000, density: 2, cirrus: 0.6, wind: { speed: 16, from: 230 }, evolution: 2, rain: 1 },
    cirrus: { coverage: 0.04, type: 0.9, cirrus: 0.75, wind: { speed: 20, from: 260 } },
  };
  for (const [name, fields] of Object.entries(presets)) {
    const source = await readFile(join(repoRoot, 'crates', 'raa-sky', 'weather', `${name}.weather`), 'utf8');
    const parsed = parseWeatherText(source);
    assert.deepEqual(parsed, withFields(fields), name);
    assert.deepEqual(parseWeatherText(formatWeatherText(parsed)), parsed, name);
  }
});
