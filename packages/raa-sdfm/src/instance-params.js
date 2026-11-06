// Procgen attributes -> the 16 floats of engine.setInstanceParams (InstanceExtra.params). The field shading multiplies the
// base colour by floats 0..2 when float 3 is set, so per-instance variation never multiplies materials or pipelines.

export const INSTANCE_PARAM_SLOTS = Object.freeze({ tint: 0, wind: 4, scale: 5, free: 6 });

const PARAM_COUNT = 16;
const FREE_COUNT = PARAM_COUNT - INSTANCE_PARAM_SLOTS.free;
const TINT_KEYS = ['tint.r', 'tint.g', 'tint.b'];

const srgbToLinear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

const isTint = (key) => key.startsWith('tint.');
const isNamed = (key) => isTint(key) || key === 'wind' || key === 'scale';

const encoder = new TextEncoder();
function compareBytes(a, b) {
  const x = encoder.encode(a);
  const y = encoder.encode(b);
  const n = Math.min(x.length, y.length);
  for (let i = 0; i < n; i++) if (x[i] !== y[i]) return x[i] - y[i];
  return x.length - y.length;
}

/**
 * The 16 floats of one instance's parameters, or null when `attrs` is absent or has no keys. A missing tint channel is
 * 1 (sRGB converted to linear); `tint.*` present sets float 3 to 1. The other attributes fill the free slots in
 * ascending key byte order and the rest are dropped.
 *
 * @param {Record<string, number> | null | undefined} attrs
 * @returns {Float32Array | null}
 */
export function instanceParams(attrs) {
  if (!attrs) return null;
  const keys = Object.keys(attrs);
  if (keys.length === 0) return null;
  const out = new Float32Array(PARAM_COUNT);
  const tint = keys.some(isTint);
  if (tint) {
    TINT_KEYS.forEach((key, i) => {
      out[INSTANCE_PARAM_SLOTS.tint + i] = key in attrs ? srgbToLinear(attrs[key]) : 1;
    });
    out[INSTANCE_PARAM_SLOTS.tint + 3] = 1;
  }
  out[INSTANCE_PARAM_SLOTS.wind] = attrs.wind ?? 0;
  out[INSTANCE_PARAM_SLOTS.scale] = attrs.scale ?? 0;
  keys.filter((key) => !isNamed(key)).sort(compareBytes).slice(0, FREE_COUNT)
    .forEach((key, i) => { out[INSTANCE_PARAM_SLOTS.free + i] = attrs[key]; });
  return out;
}
