// Shared Gerstner wave and ripple-ring vertex stage of the liquid material functions
// (S11 `material_vertex`). Every liquid function includes this file, so the GPU height is
// the CPU twin of src/waves.rs: the same packing of `record.p`, the same degenerate-wave
// guards, the same horizontal displacement and the same analytic normal.
//
// Ripple rings come from up to four impacts stored in `instance_params[i]` as
// `(x, z, start time s, strength)`. With age `a = time - start` and `r` the distance to
// `(x, z)`, one ring adds
// `strength exp(-1.6 a) sin(14 (r - 0.9 a)) (1 - smoothstep(0, 0.12, r - 0.9 a)) / (1 + 6 r)`
// to the height; its normal comes from the analytic radial derivative of that expression.

const LIQUID_TAU: f32 = 6.2831855;
// Number of waves packed in `record.p` (src/waves.rs WAVE_COUNT), as f32 for the Qi guard.
const LIQUID_WAVE_COUNT: f32 = 4.0;

// True for a finite f32 (naga's WGSL front-end has no isNan/isInf builtin; `v - v` is NaN
// for both NaN and infinity, and every comparison against NaN is false).
fn liquid_finite(v: f32) -> bool {
    return (v - v) == 0.0;
}

// One Gerstner wave: rest height, horizontal displacement and the two sums the analytic
// normal needs (`slope` is `(sum k A D.x c, sum k A D.y c)`, `steepness` the Jacobian sum
// `sum Qi k A s`).
struct LiquidWaveSample {
    height: f32,
    displacement: vec2<f32>,
    slope: vec2<f32>,
    steepness: f32,
}

fn liquid_wave_sample(wave: vec4<f32>, speed: f32, xz: vec2<f32>, t: f32) -> LiquidWaveSample {
    var s: LiquidWaveSample;
    s.height = 0.0;
    s.displacement = vec2<f32>(0.0);
    s.slope = vec2<f32>(0.0);
    s.steepness = 0.0;
    let a = wave.x;
    let l = wave.y;
    let theta = wave.z;
    let q = wave.w;
    // Degenerate waves contribute 0, exactly as wave_height() does.
    if (!liquid_finite(a)
        || !liquid_finite(l)
        || !liquid_finite(theta)
        || !liquid_finite(q)
        || !liquid_finite(speed)
        || !liquid_finite(xz.x)
        || !liquid_finite(xz.y)
        || !liquid_finite(t)) {
        return s;
    }
    if (l <= 0.0 || a == 0.0) {
        return s;
    }
    let k = LIQUID_TAU / l;
    if (!liquid_finite(k)) {
        return s;
    }
    let dir = vec2<f32>(cos(theta), sin(theta));
    let phase = k * (dot(dir, xz) - speed * t);
    if (!liquid_finite(phase)) {
        return s;
    }
    let sin_phase = sin(phase);
    let cos_phase = cos(phase);
    let qi = clamp(q / (k * a * LIQUID_WAVE_COUNT), 0.0, 1.0);
    s.height = a * sin_phase;
    s.displacement = dir * (qi * a * cos_phase);
    s.slope = dir * (k * a * cos_phase);
    s.steepness = qi * k * a * sin_phase;
    return s;
}

fn liquid_wave_total(p: array<vec4<f32>, 8>, xz: vec2<f32>, t: f32) -> LiquidWaveSample {
    let w0 = liquid_wave_sample(p[0], p[4].x, xz, t);
    let w1 = liquid_wave_sample(p[1], p[4].y, xz, t);
    let w2 = liquid_wave_sample(p[2], p[4].z, xz, t);
    let w3 = liquid_wave_sample(p[3], p[4].w, xz, t);
    var s: LiquidWaveSample;
    s.height = w0.height + w1.height + w2.height + w3.height;
    s.displacement = w0.displacement + w1.displacement + w2.displacement + w3.displacement;
    s.slope = w0.slope + w1.slope + w2.slope + w3.slope;
    s.steepness = w0.steepness + w1.steepness + w2.steepness + w3.steepness;
    return s;
}

// One ripple ring: height and the radial derivative mapped onto the XZ gradient.
struct LiquidRingSample {
    height: f32,
    gradient: vec2<f32>,
}

fn liquid_ring_sample(impact: vec4<f32>, xz: vec2<f32>, t: f32) -> LiquidRingSample {
    var s: LiquidRingSample;
    s.height = 0.0;
    s.gradient = vec2<f32>(0.0);
    let strength = impact.w;
    if (!liquid_finite(strength)
        || !liquid_finite(impact.x)
        || !liquid_finite(impact.y)
        || !liquid_finite(impact.z)
        || !liquid_finite(t)) {
        return s;
    }
    let age = t - impact.z;
    if (!(strength > 0.0) || !(age >= 0.0)) {
        return s;
    }
    let delta = xz - impact.xy;
    let r = length(delta);
    if (!liquid_finite(r)) {
        return s;
    }
    // h(r) = strength exp(-1.6 a) sin(14 q) (1 - smoothstep(0, 0.12, q)) / (1 + 6 r)
    // with q = r - 0.9 a. Its derivative uses the exact smoothstep slope.
    let q = r - 0.9 * age;
    let u = clamp(q / 0.12, 0.0, 1.0);
    let envelope = 1.0 - u * u * (3.0 - 2.0 * u);
    let denvelope = -6.0 * u * (1.0 - u) / 0.12;
    let damp = 1.0 / (1.0 + 6.0 * r);
    let ddamp = -6.0 * damp * damp;
    let wave = sin(14.0 * q);
    let dwave = 14.0 * cos(14.0 * q);
    let amp = strength * exp(-1.6 * age);
    s.height = amp * wave * envelope * damp;
    let dh_dr = amp * (dwave * envelope * damp + wave * denvelope * damp + wave * envelope * ddamp);
    let dir = select(delta / max(r, 1e-5), vec2<f32>(0.0), r < 1e-5);
    s.gradient = dir * dh_dr;
    return s;
}

fn material_vertex(in: VertexMaterialInput) -> VertexMaterialOutput {
    let xz = in.local_position.xz;
    let t = in.time;
    let waves = liquid_wave_total(in.record.p, xz, t);
    let r0 = liquid_ring_sample(in.instance_params[0], xz, t);
    let r1 = liquid_ring_sample(in.instance_params[1], xz, t);
    let r2 = liquid_ring_sample(in.instance_params[2], xz, t);
    let r3 = liquid_ring_sample(in.instance_params[3], xz, t);
    let ripple_height = r0.height + r1.height + r2.height + r3.height;
    let ripple_gradient = r0.gradient + r1.gradient + r2.gradient + r3.gradient;
    let slope = waves.slope + ripple_gradient;
    var out: VertexMaterialOutput;
    out.local_position = vec3<f32>(
        in.local_position.x + waves.displacement.x,
        waves.height + ripple_height,
        in.local_position.z + waves.displacement.y,
    );
    // GPU Gems 1 ch. 1 normal plus the ripple gradient; both height fields tilt the +Y
    // rest normal the same way.
    out.normal = normalize(vec3<f32>(-slope.x, 1.0 - waves.steepness, -slope.y));
    return out;
}
