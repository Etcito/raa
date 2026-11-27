// Shared water surface (caustics-design.md, "une seule fonction de surface"). One still water body
// per scene: a horizontal surface at `level` over the footprint [lo, hi] (world XZ), its ripple a
// fixed sum of RAA_WATER_WAVES directional sines whose analytic gradient gives the normal. The
// water's material shader, the forward's sun of submerged receivers (raa_water_sun, its caustics
// raa_water_caustic) and the DDGI update's sun evaluate these functions, so the caustics follow
// the visible ripple. `time` is the animation phase of a
// later step (0 = static). CPU twin: raa_shader::water (same constants, same order).
//
// The body is carried by FrameGpu.reserved[0..4] (zero = no water), unpacked by raa_water_unpack:
//   [0] level, rms slope, frequency (1/m), time (s)
//   [1] footprint lo.x, lo.z, hi.x, hi.z
//   [2] ior, absorption sigma r, g, b (1/m, Beer-Lambert)
//   [3] caustics on (0/1), sun bend (1 = physical refraction), unused, water present (0/1)

struct RaaWater {
    level: f32,
    slope: f32,
    frequency: f32,
    time: f32,
    lo: vec2<f32>,
    hi: vec2<f32>,
    ior: f32,
    sigma: vec3<f32>,
    caustics: bool,
    sun_bend: f32,
    present: bool,
}

const RAA_WATER_WAVES: u32 = 8u;
// Wavenumber ratio between consecutive waves, golden-angle direction step.
const RAA_WATER_RATIO: f32 = 1.27;
const RAA_WATER_ANGLE_STEP: f32 = 2.3999632;

fn raa_water_unpack(r: array<vec4<f32>, 4>) -> RaaWater {
    return RaaWater(r[0].x, r[0].y, r[0].z, r[0].w, r[1].xy, r[1].zw, r[2].x, r[2].yzw, r[3].x > 0.5,
        r[3].y, r[3].w > 0.5);
}

fn raa_water_inside(w: RaaWater, xz: vec2<f32>) -> bool {
    return all(xz >= w.lo) && all(xz <= w.hi);
}

// Normalisation of the per-wave slope weights 1.27^(-i/2): sqrt(sum w_i^2 / 2).
fn raa_water_norm() -> f32 {
    var s = 0.0;
    for (var i = 0u; i < RAA_WATER_WAVES; i++) {
        s += pow(RAA_WATER_RATIO, -f32(i));
    }
    return sqrt(0.5 * s);
}

// (h, dh/dx, dh/dz) of the ripple at xz. Wave i: direction angle 0.4 + i * golden angle,
// wavenumber pi * frequency * 1.27^i (the longest wavelength is 2 / frequency), maximum slope
// slope * 1.27^(-i/2) / norm (so the RMS of |grad h| is `slope`), phase 2 pi fract(0.618 (i + 1)),
// deep-water angular frequency sqrt(9.81 k) for the time term.
fn raa_water_height_gradient(w: RaaWater, xz: vec2<f32>) -> vec3<f32> {
    var out = vec3<f32>(0.0);
    if w.slope <= 0.0 || w.frequency <= 0.0 { return out; }
    let norm = raa_water_norm();
    for (var i = 0u; i < RAA_WATER_WAVES; i++) {
        let fi = f32(i);
        let angle = 0.4 + RAA_WATER_ANGLE_STEP * fi;
        let d = vec2<f32>(cos(angle), sin(angle));
        let k = 3.14159265 * w.frequency * pow(RAA_WATER_RATIO, fi);
        let s = w.slope * pow(RAA_WATER_RATIO, -0.5 * fi) / norm;
        let phase = 6.28318531 * fract(0.618034 * (fi + 1.0)) - sqrt(9.81 * k) * w.time;
        let x = k * dot(d, xz) + phase;
        out += vec3<f32>((s / k) * sin(x), s * cos(x) * d);
    }
    return out;
}

// Unit upward normal of the surface at xz: normalize(-dh/dx, 1, -dh/dz).
fn raa_water_normal(w: RaaWater, xz: vec2<f32>) -> vec3<f32> {
    let g = raa_water_height_gradient(w, xz);
    return normalize(vec3<f32>(-g.y, 1.0, -g.z));
}

// Unpolarised Fresnel reflectance of a smooth dielectric interface; `cos_i` > 0 on the incident
// side, `eta` = n_incident / n_transmitted. Total internal reflection returns 1.
fn raa_water_fresnel(cos_i: f32, eta: f32) -> f32 {
    let c = clamp(cos_i, 0.0, 1.0);
    let sin_t2 = eta * eta * (1.0 - c * c);
    if sin_t2 >= 1.0 { return 1.0; }
    let cos_t = sqrt(1.0 - sin_t2);
    let rs = (eta * c - cos_t) / (eta * c + cos_t);
    let rp = (c - eta * cos_t) / (c + eta * cos_t);
    return 0.5 * (rs * rs + rp * rp);
}

// Refraction of the unit direction `d` through the normal `n` facing against d (dot(d, n) < 0),
// eta = n_incident / n_transmitted; w = 0 on total internal reflection.
fn raa_water_refract(d: vec3<f32>, n: vec3<f32>, eta: f32) -> vec4<f32> {
    let c = -dot(d, n);
    let k = 1.0 - eta * eta * (1.0 - c * c);
    if k < 0.0 { return vec4<f32>(reflect(d, n), 0.0); }
    return vec4<f32>(normalize(eta * d + (eta * c - sqrt(k)) * n), 1.0);
}

// Where the sun's path to the submerged point `p` crosses the surface (flat refraction): the sun
// visibility of p is tested from there (shadow map, shadow ray), so a shadow edge under the water
// moves as the refracted light does. p itself above the water or outside the footprint.
fn raa_water_sun_entry(w: RaaWater, sun_dir: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
    if !w.present || p.y >= w.level || !raa_water_inside(w, p.xz) { return p; }
    let t = raa_water_refract(sun_dir, vec3<f32>(0.0, 1.0, 0.0), 1.0 / max(w.ior, 1.0));
    if t.y > -1e-3 { return p; }
    return p - t.xyz * ((w.level - p.y) / -t.y);
}

// The sun of a receiver `p` under the water: it arrives along the direction refracted by the
// rippled surface at the point x whose refracted sun lands on p (raa_water_solve, shared with
// the caustics), through `entry` where that path crosses the surface, attenuated by the Fresnel
// transmittance and concentrated by the caustics `k` (1 / |det| at x). So the shadow of a wall
// under water has the ripple's wavy, broken edge and the caustics run right up to it. `sun_bend`
// is artistic: the horizontal run per metre of depth is straight + sun_bend x (refracted -
// straight), so 0 keeps the straight sun, 1 is physical and more bends it further (and the
// ripple's wobble of the edge with it), at most half a straight run past the vertical: the
// strip lit next to a wall on the sun side grows by depth x sun_bend x (tan(incidence) -
// tan(refraction)) on average. Outside the water or with caustics off (setWater caustics: false, the
// unrefracted sun of before): not under, entry p, to_sun -sun_dir, transmit 1.
struct RaaWaterSun {
    under: bool,
    entry: vec3<f32>,
    to_sun: vec3<f32>,
    transmit: f32,
    k: f32,
}

fn raa_water_sun(w: RaaWater, sun_dir: vec3<f32>, p: vec3<f32>) -> RaaWaterSun {
    var o = RaaWaterSun(false, p, -sun_dir, 1.0, 1.0);
    if !w.present || !w.caustics || p.y >= w.level || !raa_water_inside(w, p.xz) { return o; }
    let t = raa_water_refract(sun_dir, vec3<f32>(0.0, 1.0, 0.0), 1.0 / max(w.ior, 1.0));
    if t.y > -1e-3 { return o; }
    let straight = sun_dir.xz / min(sun_dir.y, -1e-3);
    var refracted = t.xz / t.y;
    if w.slope > 0.0 && w.frequency > 0.0 {
        let sv = raa_water_solve(w, sun_dir, p);
        let tw = raa_water_refract(sun_dir, normalize(vec3<f32>(-sv.g.x, 1.0, -sv.g.y)),
            1.0 / max(w.ior, 1.0));
        if tw.y < -1e-3 { refracted = tw.xz / tw.y; }
        o.k = raa_water_caustic_at(w, sun_dir, p, sv);
    }
    var run = mix(straight, refracted, w.sun_bend);
    // Past the vertical by at most half a straight run (along the sun's azimuth).
    let ss = max(dot(straight, straight), 1e-8);
    let along = dot(run, straight) / ss;
    if along < -0.5 { run += (-0.5 - along) * straight; }
    let dir = normalize(vec3<f32>(-run.x, -1.0, -run.y));
    o.under = true;
    o.entry = vec3<f32>(p.x + run.x * (w.level - p.y), w.level, p.z + run.y * (w.level - p.y));
    o.to_sun = -dir;
    o.transmit = raa_water_sun_transmittance(w, sun_dir);
    return o;
}

// Flat-surface transmittance of the sun into the water: 1 - F at the sun's incidence.
fn raa_water_sun_transmittance(w: RaaWater, sun_dir: vec3<f32>) -> f32 {
    return 1.0 - raa_water_fresnel(-sun_dir.y, 1.0 / max(w.ior, 1.0));
}

// Caustics of the sun on a submerged point `p` (Evan Wallace's WebGL Water: the area ratio of
// the refracted surface grid; Shah et al. 2007 caustics mapping), evaluated per receiver with no
// map: the sun refracted at a surface point x lands at x + t.xz * depth / -t.y. Starting from the
// flat-refracted entry, three fixed-point steps find the x that lands on p (without them the
// floor would average 2x too bright under the pool's ripple); one pass over the waves then gives
// the slope g and curvature H at x, the refracted directions at x and at x + e (slope g + e H)
// give the Jacobian of the landing map, and the light concentration is 1 / |det|. 1 for a flat
// surface, above the water, outside it, without a body or with caustics off; about 1 on average
// over the floor (refracted-grid check in water.rs); folds (the bright lines) capped at 1 / 0.15.
// The first-order footprint e also stands for the sun's angular size.
fn raa_water_caustic(w: RaaWater, sun_dir: vec3<f32>, p: vec3<f32>) -> f32 {
    if !w.present || !w.caustics || w.slope <= 0.0 || w.frequency <= 0.0 || p.y >= w.level
        || !raa_water_inside(w, p.xz) { return 1.0; }
    return raa_water_caustic_at(w, sun_dir, p, raa_water_solve(w, sun_dir, p));
}

// The surface point x whose refracted sun lands on the submerged p (three fixed-point steps from
// the flat-refracted entry), with the slope g and curvature (hxx, hxz, hzz) there.
struct RaaWaterSolve {
    x: vec2<f32>,
    g: vec2<f32>,
    h: vec3<f32>,
}

fn raa_water_solve(w: RaaWater, sun_dir: vec3<f32>, p: vec3<f32>) -> RaaWaterSolve {
    let depth = w.level - p.y;
    let eta = 1.0 / max(w.ior, 1.0);
    var x = raa_water_sun_entry(w, sun_dir, p).xz;
    for (var j = 0u; j < 3u; j++) {
        let s = raa_water_height_gradient(w, x).yz;
        let t = raa_water_refract(sun_dir, normalize(vec3<f32>(-s.x, 1.0, -s.y)), eta);
        x = p.xz - t.xz * (depth / max(-t.y, 0.05));
    }
    let norm = raa_water_norm();
    var g = vec2<f32>(0.0);
    var hxx_hxz_hzz = vec3<f32>(0.0);
    for (var i = 0u; i < RAA_WATER_WAVES; i++) {
        let fi = f32(i);
        let angle = 0.4 + RAA_WATER_ANGLE_STEP * fi;
        let d = vec2<f32>(cos(angle), sin(angle));
        let k = 3.14159265 * w.frequency * pow(RAA_WATER_RATIO, fi);
        let s = w.slope * pow(RAA_WATER_RATIO, -0.5 * fi) / norm;
        let phase = 6.28318531 * fract(0.618034 * (fi + 1.0)) - sqrt(9.81 * k) * w.time;
        let arg = k * dot(d, x) + phase;
        g += s * cos(arg) * d;
        hxx_hxz_hzz -= s * k * sin(arg) * vec3<f32>(d.x * d.x, d.x * d.y, d.y * d.y);
    }
    return RaaWaterSolve(x, g, hxx_hxz_hzz);
}

// 1 / |det| of the landing map's first-order Jacobian at the solved point (raa_water_caustic).
fn raa_water_caustic_at(w: RaaWater, sun_dir: vec3<f32>, p: vec3<f32>, sv: RaaWaterSolve) -> f32 {
    let depth = w.level - p.y;
    let eta = 1.0 / max(w.ior, 1.0);
    let e = 0.01;
    let g = sv.g;
    let g1 = g + e * sv.h.xy;
    let g2 = g + e * sv.h.yz;
    let t0 = raa_water_refract(sun_dir, normalize(vec3<f32>(-g.x, 1.0, -g.y)), eta);
    let t1 = raa_water_refract(sun_dir, normalize(vec3<f32>(-g1.x, 1.0, -g1.y)), eta);
    let t2 = raa_water_refract(sun_dir, normalize(vec3<f32>(-g2.x, 1.0, -g2.y)), eta);
    let h0 = t0.xz * (depth / max(-t0.y, 0.05));
    let a = vec2<f32>(e, 0.0) + t1.xz * (depth / max(-t1.y, 0.05)) - h0;
    let b = vec2<f32>(0.0, e) + t2.xz * (depth / max(-t2.y, 0.05)) - h0;
    return 1.0 / max(abs(a.x * b.y - a.y * b.x) / (e * e), 0.15);
}
