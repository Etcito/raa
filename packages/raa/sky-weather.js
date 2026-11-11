// Pure `.weather` text parsing and formatting for @etcito/raa (m08 `parseWeatherText` and
// `formatWeatherText`). No Engine, GPU or service is needed: these functions only read and write
// the authored text form of a weather state. The Rust parser and formatter they match are
// `Weather::parse` and `Weather::to_text` in crates/raa-sky/src/weather.rs.
//
// Text and values are adapted from Solstice's crates/solstice-render/src/clouds/weather.rs and
// weather/*.weather at revision 7e465b599c13da165948ebaa44df824c36c3ae07.
/*
MIT License

Copyright (c) 2026 SamG-Coder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// Fair weather: the base every text file starts from (weather/fair.weather:2-9).
// Distances are metres, wind speed is m/s and wind.from is compass degrees FROM north.
function fairDefaults() {
  return {
    coverage: 0.3,
    type: 0.9,
    base: 1200,
    thickness: 1600,
    density: 1,
    cirrus: 0.15,
    wind: { speed: 7, from: 250 },
    evolution: 1,
    rain: 0,
  };
}

// Decimal text that Rust's f32 parser also accepts: sign, digits with an optional point,
// optional exponent. Words such as inf, NaN, hex and empty text are refused here.
const DECIMAL = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;

// Every value is a single-precision number on the GPU: finite as an f32 is the bound.
function isF32Finite(value) {
  return Number.isFinite(value) && Number.isFinite(Math.fround(value));
}

function refuse(message) {
  const error = new Error(`sky.invalid: ${message}`);
  error.name = 'RaaError';
  error.code = 'sky.invalid';
  return error;
}

function clamp01(value) {
  return Math.min(1, Math.max(0, value));
}

function parseNumber(text, at) {
  const value = DECIMAL.test(text) ? Number(text) : Number.NaN;
  if (!isF32Finite(value)) throw at(`${JSON.stringify(text)} is not a number`);
  return value;
}

/**
 * Reads `.weather` text into a complete weather state (pure).
 *
 * Starts from fair defaults, strips `#` comments and whitespace, accepts empty or comment-only
 * text, keeps the last repeated setting, accepts `wind` with either component omitted and clamps
 * each value as the file format does. Unknown keys and malformed, nonfinite or f32-overflowing
 * numbers are refused with a one-based `line N:` message.
 *
 * @param {string} text
 * @returns {{coverage:number,type:number,base:number,thickness:number,density:number,cirrus:number,wind:{speed:number,from:number},evolution:number,rain:number}}
 * @throws RaaError `sky.invalid` with a one-based line message.
 */
export function parseWeatherText(text) {
  if (typeof text !== 'string') throw refuse('weather text must be a string');
  const weather = fairDefaults();
  text.split('\n').forEach((raw, index) => {
    const words = raw.split('#')[0].split(/\s+/).filter(Boolean);
    if (words.length === 0) return;
    const [keyword, ...rest] = words;
    const at = (message) => refuse(`line ${index + 1}: ${message}`);
    const single = () => {
      if (rest.length !== 1) throw at(`${keyword} takes one number`);
      return parseNumber(rest[0], at);
    };
    switch (keyword) {
      case 'coverage': weather.coverage = clamp01(single()); break;
      case 'type': weather.type = clamp01(single()); break;
      case 'base': weather.base = Math.max(0, single()); break;
      case 'thickness': weather.thickness = Math.max(100, single()); break;
      case 'density': weather.density = Math.max(0, single()); break;
      case 'cirrus': weather.cirrus = clamp01(single()); break;
      case 'evolution': weather.evolution = Math.max(0, single()); break;
      case 'rain': weather.rain = clamp01(single()); break;
      case 'wind':
        for (const word of rest) {
          const equals = word.indexOf('=');
          if (equals < 0) throw at(`expected key=value, found ${JSON.stringify(word)}`);
          const key = word.slice(0, equals);
          const value = word.slice(equals + 1);
          if (key === 'speed') weather.wind.speed = Math.max(0, parseNumber(value, at));
          else if (key === 'from') weather.wind.from = parseNumber(value, at);
          else throw at(`unknown wind setting ${key}`);
        }
        break;
      default:
        throw at(`unknown setting ${JSON.stringify(keyword)}`);
    }
  });
  if (!isF32Finite(Math.fround(weather.base + weather.thickness))) {
    throw refuse('base + thickness must be finite');
  }
  return weather;
}

const FIELDS = ['coverage', 'type', 'base', 'thickness', 'density', 'cirrus', 'evolution', 'rain'];

// Strict structured validation of a weather state before it is written as text. Unlike the
// parser it never clamps: an out-of-domain or nonfinite field is refused with its field path.
function checkNumber(path, value, minimum, maximum) {
  if (typeof value !== 'number' || !isF32Finite(value)) {
    throw refuse(`${path} must be a finite number`);
  }
  if (value < minimum || (maximum !== undefined && value > maximum)) {
    const range = maximum === undefined ? `at least ${minimum}` : `in [${minimum}, ${maximum}]`;
    throw refuse(`${path} must be ${range}`);
  }
  return value;
}

function checkKeys(path, value, keys) {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw refuse(`${path} must be an object`);
  }
  for (const key of Object.keys(value)) {
    if (!keys.includes(key)) throw refuse(`${path}.${key} is not a weather field`);
  }
  for (const key of keys) {
    if (!(key in value)) throw refuse(`${path}.${key} is missing`);
  }
}

function validateWeather(weather) {
  checkKeys('weather', weather, [...FIELDS, 'wind']);
  checkKeys('weather.wind', weather.wind, ['speed', 'from']);
  const valid = {
    coverage: checkNumber('weather.coverage', weather.coverage, 0, 1),
    type: checkNumber('weather.type', weather.type, 0, 1),
    base: checkNumber('weather.base', weather.base, 0),
    thickness: checkNumber('weather.thickness', weather.thickness, 100),
    density: checkNumber('weather.density', weather.density, 0),
    cirrus: checkNumber('weather.cirrus', weather.cirrus, 0, 1),
    speed: checkNumber('weather.wind.speed', weather.wind.speed, 0),
    from: checkNumber('weather.wind.from', weather.wind.from, Number.NEGATIVE_INFINITY),
    evolution: checkNumber('weather.evolution', weather.evolution, 0),
    rain: checkNumber('weather.rain', weather.rain, 0, 1),
  };
  if (!isF32Finite(Math.fround(valid.base + valid.thickness))) {
    throw refuse('weather.base + weather.thickness must be finite');
  }
  return valid;
}

/**
 * Formats a weather state as `.weather` text (pure): coverage, type, base, thickness, density,
 * cirrus, `wind speed=… from=…`, evolution and rain, one setting per line, with a final LF.
 * The state is validated strictly first; out-of-domain values are refused, not clamped.
 *
 * @param {{coverage:number,type:number,base:number,thickness:number,density:number,cirrus:number,wind:{speed:number,from:number},evolution:number,rain:number}} weather
 * @returns {string}
 * @throws RaaError `sky.invalid` naming the offending field path.
 */
export function formatWeatherText(weather) {
  const w = validateWeather(weather);
  return [
    `coverage ${w.coverage}`,
    `type ${w.type}`,
    `base ${w.base}`,
    `thickness ${w.thickness}`,
    `density ${w.density}`,
    `cirrus ${w.cirrus}`,
    `wind speed=${w.speed} from=${w.from}`,
    `evolution ${w.evolution}`,
    `rain ${w.rain}`,
    '',
  ].join('\n');
}
