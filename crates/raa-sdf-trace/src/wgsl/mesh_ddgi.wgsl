// DDGI probe update: one 64-thread workgroup traces, relocates, classifies and blends one
// scheduled probe. Rays go through the shared world-space mesh BVH.
//
// A probe keeps its lighting as separate LINEAR groups per irradiance texel, so a change of one
// light source never needs the others to be traced again:
//   sun      response to a white unit sun (shadow ray + bounces); composed x frame.sun_color
//   sky      transfer: order-2 SH coefficients (rgb) of the response to the sky radiance, i.e.
//            visibility for escaped rays and albedo x transfer of the volume at hits (bounces);
//            composed as dot(transfer, current sky SH) x sky scale: follows any sky change
//   residual escaped-ray sky radiance minus its SH reconstruction (what order 2 cannot hold: the
//            aureole, clouds), divided by the sky norm (band-0 luma) so a sky that only gets
//            brighter keeps it; refreshed with the sun group; composed x sky norm x sky scale
//   other    local lights (one shadow ray), emission and their bounces
// The composed irradiance tile read by the lookup is rebuilt from the groups at the end of an
// update and by `ddgi_relight` (every probe, no ray) when the sun colour, the sky or its scale
// changed. A job names the groups it refreshes (the scheduler invalidates per group and, for
// local lights, per probe through the light mask each probe records).

struct DdgiParams {
    rotation: array<vec4<f32>, 3>,
    // rays per probe, scheduled probes, frame, flags (1 relocation, 2 classification, 4 bounce,
    // 8 sky transfer)
    counts: vec4<u32>,
    // hysteresis, ray limit, irradiance change threshold, minimum front-face distance (m)
    tuning: vec4<f32>,
    // sky scale, backface fraction threshold, cos of the cone around the sun left out of the sky,
    // smallest light contribution (luma) recorded in a probe's light mask
    extra: vec4<f32>,
}

@group(0) @binding(27) var<storage, read_write> ddgi_probes: array<vec4<u32>>;
@group(0) @binding(28) var<uniform> ddgi_params: DdgiParams;
// Two words per job. Word 0: slot (low 16 bits), geometry age (high 16 bits; 0 = fresh).
// Word 1: sun age (bits 0-7), other age (8-15), groups to refresh (16-18: 1 geometry and sky
// transfer, 2 sun and sky residual, 4 other), bit 24: replace the light mask instead of adding.
@group(0) @binding(29) var<storage, read> ddgi_jobs: array<u32>;

fn raa_ddgi_word(index: u32) -> vec4<u32> {
    return ddgi_probes[index];
}

const DDGI_THREADS: u32 = 64u;
const DDGI_MAX_RAYS: u32 = 256u;
const DDGI_MISS: f32 = 1e27;
const DDGI_RELOCATE: u32 = 1u;
const DDGI_CLASSIFY: u32 = 2u;
const DDGI_BOUNCE: u32 = 4u;
const DDGI_SKY_TRANSFER: u32 = 8u;
const DDGI_GEOMETRY: u32 = 1u;
const DDGI_SUN: u32 = 2u;
const DDGI_OTHER: u32 = 4u;
// Rows of one irradiance texel's groups: 36 f16 (sun 0-2, residual 3-5, other 6-8, transfer 9-35).
const DDGI_GROUP_ROWS: u32 = 5u;
// Rows of one ray's bounced sky transfer in the scratch area (27 f16).
const DDGI_SCRATCH_ROWS: u32 = 4u;
const DDGI_SKY_SAMPLES: u32 = 4096u;

// Per ray, f16: sun rgb, residual rgb, other rg (other b rides in ddgi_direction.w).
var<workgroup> ddgi_ray: array<vec4<u32>, 256>;
var<workgroup> ddgi_direction: array<vec4<f32>, 256>;
var<workgroup> ddgi_distance: array<f32, 256>;
var<workgroup> ddgi_irr: array<vec3<f32>, 64>;
var<workgroup> ddgi_depth_tile: array<vec2<f32>, 256>;
// slot, run (0 skip, 1 trace), fresh, geometry age
var<workgroup> ddgi_job: vec4<u32>;
// sun age, other age, groups, replace the light mask
var<workgroup> ddgi_job_groups: vec4<u32>;
var<workgroup> ddgi_origin: vec4<f32>;
var<workgroup> ddgi_light_bits: array<atomic<u32>, 8>;
var<workgroup> ddgi_sky_partial: array<array<vec3<f32>, 9>, 64>;

// Header row 10: first row of the texel groups, of the sky SH (9 rows, rgb f32), of the ray
// scratch, and the light mask rows per probe.
fn ddgi_group_row() -> u32 { return ddgi_probes[10].x; }
fn ddgi_sky_row() -> u32 { return ddgi_probes[10].y; }
fn ddgi_scratch_row() -> u32 { return ddgi_probes[10].z; }

// The 9 real SH basis functions (order 2), raa-ibl order.
fn ddgi_sh(d: vec3<f32>) -> array<f32, 9> {
    return array<f32, 9>(
        0.282095,
        0.488603 * d.y,
        0.488603 * d.z,
        0.488603 * d.x,
        1.092548 * d.x * d.y,
        1.092548 * d.y * d.z,
        0.315392 * (3.0 * d.z * d.z - 1.0),
        1.092548 * d.x * d.z,
        0.546274 * (d.x * d.x - d.y * d.y),
    );
}

fn ddgi_sky_coefficient(k: u32) -> vec3<f32> {
    return bitcast<vec3<f32>>(ddgi_probes[ddgi_sky_row() + k].xyz);
}

// Magnitude of the frame's sky (mean radiance luma), the unit of the stored residual; row 9 of
// the sky rows. Zero for a black sky.
fn ddgi_sky_norm() -> f32 {
    return bitcast<f32>(ddgi_probes[ddgi_sky_row() + 9u].x);
}

// Sky radiance of the frame (unit sky scale).
fn ddgi_sky(direction: vec3<f32>) -> vec3<f32> {
    if options[0].z == 0u { return frame.ambient.rgb; }
    return textureSampleLevel(sky, raa_linear_clamp, direction, 0.0).rgb;
}

fn ddgi_in_sun_cone(direction: vec3<f32>) -> bool {
    return dot(direction, -frame.sun_dir.xyz) > ddgi_params.extra.z;
}

// What an escaped ray adds to the residual group: the sky radiance the SH transfer does not
// carry. The sun disc belongs to the sun light: its cone is left out of both.
fn ddgi_sky_residual(direction: vec3<f32>) -> vec3<f32> {
    let norm = ddgi_sky_norm();
    if ddgi_in_sun_cone(direction) || norm <= 1e-7 { return vec3<f32>(0.0); }
    var rgb = ddgi_sky(direction);
    if (ddgi_params.counts.w & DDGI_SKY_TRANSFER) != 0u {
        let y = ddgi_sh(direction);
        for (var k = 0u; k < 9u; k++) { rgb -= y[k] * ddgi_sky_coefficient(k); }
    }
    return rgb / norm;
}

// One irradiance texel's groups, 20 f16 pairs over 5 rows: sun rg | sun b, residual r |
// residual gb | other rg | other b, - | then 14 pairs of sky transfer (element 3k+c of
// coefficient k, channel c). The transfer stays in its packed pair order wherever it is only
// blended or summed.
struct DdgiDirect {
    sun: vec3<f32>,
    residual: vec3<f32>,
    other: vec3<f32>,
}

struct DdgiTexel {
    direct: DdgiDirect,
    transfer: array<vec2<f32>, 14>,
}

fn ddgi_direct_unpack(r0: vec4<u32>, r1x: u32) -> DdgiDirect {
    let a = unpack2x16float(r0.y);
    return DdgiDirect(vec3<f32>(unpack2x16float(r0.x), a.x), vec3<f32>(a.y, unpack2x16float(r0.z)),
        vec3<f32>(unpack2x16float(r0.w), unpack2x16float(r1x).x));
}

fn ddgi_texel_load(row: u32) -> DdgiTexel {
    var t: DdgiTexel;
    let r1 = ddgi_probes[row + 1u];
    t.direct = ddgi_direct_unpack(ddgi_probes[row], r1.x);
    t.transfer[0] = unpack2x16float(r1.y);
    t.transfer[1] = unpack2x16float(r1.z);
    t.transfer[2] = unpack2x16float(r1.w);
    for (var i = 0u; i < 3u; i++) {
        let w = ddgi_probes[row + 2u + i];
        t.transfer[3u + 4u * i] = unpack2x16float(w.x);
        t.transfer[4u + 4u * i] = unpack2x16float(w.y);
        t.transfer[5u + 4u * i] = unpack2x16float(w.z);
        if i < 2u { t.transfer[6u + 4u * i] = unpack2x16float(w.w); }
    }
    return t;
}

fn ddgi_f16(v: vec3<f32>) -> vec3<f32> {
    return clamp(v, vec3<f32>(-65504.0), vec3<f32>(65504.0));
}

fn ddgi_pack(v: vec2<f32>) -> u32 {
    return pack2x16float(clamp(v, vec2<f32>(-65504.0), vec2<f32>(65504.0)));
}

fn ddgi_direct_pack(d: DdgiDirect) -> vec4<u32> {
    return vec4<u32>(ddgi_pack(d.sun.xy), ddgi_pack(vec2<f32>(d.sun.z, d.residual.x)),
        ddgi_pack(d.residual.yz), ddgi_pack(d.other.xy));
}

// Stores the direct groups, and the transfer rows too when `transfer` is set.
fn ddgi_texel_store(row: u32, texel: DdgiTexel, transfer: bool) {
    ddgi_probes[row] = ddgi_direct_pack(texel.direct);
    let b = ddgi_pack(vec2<f32>(texel.direct.other.z, 0.0));
    if !transfer {
        let old = ddgi_probes[row + 1u];
        ddgi_probes[row + 1u] = vec4<u32>(b, old.yzw);
        return;
    }
    ddgi_probes[row + 1u] = vec4<u32>(b, ddgi_pack(texel.transfer[0]), ddgi_pack(texel.transfer[1]),
        ddgi_pack(texel.transfer[2]));
    for (var i = 0u; i < 3u; i++) {
        var last = 0u;
        if i < 2u { last = ddgi_pack(texel.transfer[6u + 4u * i]); }
        ddgi_probes[row + 2u + i] = vec4<u32>(ddgi_pack(texel.transfer[3u + 4u * i]),
            ddgi_pack(texel.transfer[4u + 4u * i]), ddgi_pack(texel.transfer[5u + 4u * i]), last);
    }
}

// Irradiance / pi of a texel under the frame's sun colour, sky and sky scale.
fn ddgi_compose(texel: DdgiTexel) -> vec3<f32> {
    var sky_light = texel.direct.residual * ddgi_sky_norm();
    if (ddgi_params.counts.w & DDGI_SKY_TRANSFER) != 0u {
        let row = ddgi_sky_row();
        for (var k = 0u; k < 9u; k++) {
            let e = 3u * k;
            let t = vec3<f32>(texel.transfer[e >> 1u][e & 1u], texel.transfer[(e + 1u) >> 1u][(e + 1u) & 1u],
                texel.transfer[(e + 2u) >> 1u][(e + 2u) & 1u]);
            sky_light += t * bitcast<vec3<f32>>(ddgi_probes[row + k].xyz);
        }
    }
    let rgb = texel.direct.sun * max(frame.sun_color.rgb, vec3<f32>(0.0))
        + ddgi_params.extra.x * sky_light + texel.direct.other;
    return clamp(rgb, vec3<f32>(0.0), vec3<f32>(65504.0));
}

fn ddgi_ray_direction(i: u32, count: u32) -> vec3<f32> {
    let golden = 2.39996323;
    let z = 1.0 - (2.0 * f32(i) + 1.0) / f32(count);
    let r = sqrt(max(0.0, 1.0 - z * z));
    let phi = f32(i) * golden;
    let d = vec3<f32>(r * cos(phi), r * sin(phi), z);
    let m = ddgi_params.rotation;
    return normalize(m[0].xyz * d.x + m[1].xyz * d.y + m[2].xyz * d.z);
}

fn ddgi_slot_coord(v: RaaDdgiVolume, slot: u32) -> vec3<i32> {
    let local = slot - v.offset;
    let s = vec3<u32>(local % v.counts.x, (local / v.counts.x) % v.counts.y,
        local / (v.counts.x * v.counts.y));
    let n = vec3<i32>(v.counts);
    // The unique coordinate of the current volume that maps to this slot.
    let shift = ((vec3<i32>(s) - v.base) % n + n) % n;
    return v.base + shift;
}

// Interior source of a texel of a bordered tile with `n` interior texels (1-based interior).
fn ddgi_border_source(xy: vec2<u32>, n: u32) -> vec2<u32> {
    let last = n + 1u;
    var s = xy;
    let x_edge = xy.x == 0u || xy.x == last;
    let y_edge = xy.y == 0u || xy.y == last;
    if x_edge && y_edge {
        s = vec2<u32>(select(1u, n, xy.x == 0u), select(1u, n, xy.y == 0u));
    } else if y_edge {
        s = vec2<u32>(last - xy.x, select(n, 1u, xy.y == 0u));
    } else if x_edge {
        s = vec2<u32>(select(n, 1u, xy.x == 0u), last - xy.y);
    }
    return s - 1u;
}

// The volume's groups at a ray hit, each as irradiance / pi: the cascaded eight-probe weights of
// `raa_ddgi_sample` (`raa_ddgi_corners`), blended linearly so the groups stay separable, with one
// texel per probe picked in proportion to its bilinear weight (unbiased; the probe hysteresis
// averages it). Default far mode: renormalized over the usable probes; legacy: faded.
fn ddgi_bounce(p: vec3<f32>, n: vec3<f32>, view_dir: vec3<f32>, groups: u32,
               seed: ptr<function, u32>) -> DdgiTexel {
    var out: DdgiTexel;
    let v0 = raa_ddgi_volume();
    let clamp_far = (v0.flags & RAA_DDGI_FAR_AMBIENT) == 0u;
    let cascades = raa_ddgi_cascades();
    let tap = raa_ddgi_tap(n, f32(RAA_DDGI_IRR));
    let group_row = ddgi_group_row();
    var weight_sum = 0.0;
    var remaining = 1.0;
    for (var k = 0u; k < cascades; k++) {
        let v = raa_ddgi_volume_at(k);
        let c = raa_ddgi_corners(v, p, n, view_dir, k + 1u == cascades, clamp_far);
        let confidence = raa_ddgi_confidence(c, clamp_far);
        if confidence <= 0.0 { continue; }
        let share = remaining * confidence;
        for (var i = 0u; i < 8u; i++) {
            if c.weight[i] <= 0.0 { continue; }
            let weight = share * c.weight[i] / c.total;
            let xy = vec2<u32>(select(tap.i0.x, tap.i1.x, random(seed) < tap.f.x),
                select(tap.i0.y, tap.i1.y, random(seed) < tap.f.y));
            let interior = ddgi_border_source(xy, RAA_DDGI_IRR);
            let row = group_row
                + (c.slot[i] * RAA_DDGI_IRR * RAA_DDGI_IRR + interior.y * RAA_DDGI_IRR + interior.x)
                * DDGI_GROUP_ROWS;
            let r1 = ddgi_probes[row + 1u];
            let direct = ddgi_direct_unpack(ddgi_probes[row], r1.x);
            out.direct.sun += weight * direct.sun;
            out.direct.residual += weight * direct.residual;
            out.direct.other += weight * direct.other;
            if (groups & DDGI_GEOMETRY) != 0u {
                out.transfer[0] += weight * unpack2x16float(r1.y);
                out.transfer[1] += weight * unpack2x16float(r1.z);
                out.transfer[2] += weight * unpack2x16float(r1.w);
                let r2 = ddgi_probes[row + 2u];
                let r3 = ddgi_probes[row + 3u];
                let r4 = ddgi_probes[row + 4u];
                out.transfer[3] += weight * unpack2x16float(r2.x);
                out.transfer[4] += weight * unpack2x16float(r2.y);
                out.transfer[5] += weight * unpack2x16float(r2.z);
                out.transfer[6] += weight * unpack2x16float(r2.w);
                out.transfer[7] += weight * unpack2x16float(r3.x);
                out.transfer[8] += weight * unpack2x16float(r3.y);
                out.transfer[9] += weight * unpack2x16float(r3.z);
                out.transfer[10] += weight * unpack2x16float(r3.w);
                out.transfer[11] += weight * unpack2x16float(r4.x);
                out.transfer[12] += weight * unpack2x16float(r4.y);
                out.transfer[13] += weight * unpack2x16float(r4.z);
            }
        }
        weight_sum += share;
        remaining -= share;
        if remaining <= 1e-3 { break; }
    }
    if weight_sum <= 0.0 { return DdgiTexel(); }
    // Legacy keeps the faded confidence (the weights already carry it).
    let scale = v0.intensity * select(1.0, 1.0 / weight_sum, clamp_far);
    out.direct.sun *= scale;
    out.direct.residual *= scale;
    out.direct.other *= scale;
    if (groups & DDGI_GEOMETRY) != 0u {
        for (var j = 0u; j < 14u; j++) { out.transfer[j] *= scale; }
    }
    return out;
}

struct DdgiShaded {
    // The hit's contribution to each group; `transfer` only for a hit with bounces.
    texel: DdgiTexel,
    distance: f32,
    bounced: bool,
}

fn ddgi_shade(origin: vec3<f32>, direction: vec3<f32>, limit: f32, groups: u32,
              seed: ptr<function, u32>) -> DdgiShaded {
    var out: DdgiShaded;
    let hit = closest(Ray(origin, direction), limit, false);
    if hit.object == MISS {
        if (groups & DDGI_SUN) != 0u { out.texel.direct.residual = ddgi_sky_residual(direction); }
        out.distance = DDGI_MISS;
        return out;
    }
    let tri = mesh_triangle(hit.object);
    let geom = normalize(cross(tri.b.position - tri.a.position, tri.c.position - tri.a.position));
    if dot(geom, direction) > 0.0 && (mesh_material(tri.material).flags.x & 2u) == 0u {
        // Back face: the probe may sit inside geometry. Shorten as the reference does.
        out.distance = -0.2 * hit.t;
        return out;
    }
    out.distance = hit.t;
    let material = hit_material(tri, hit.bary);
    let p = origin + direction * hit.t;
    let ng = select(geom, -geom, dot(geom, direction) > 0.0);
    let blended = tri.a.normal * (1.0 - hit.bary.x - hit.bary.y)
        + tri.b.normal * hit.bary.x + tri.c.normal * hit.bary.y;
    var n = ng;
    if dot(blended, blended) > 1e-8 {
        let candidate = normalize(blended);
        n = select(-candidate, candidate, dot(candidate, ng) >= 0.0);
    }
    if (groups & DDGI_OTHER) != 0u { out.texel.direct.other = material.emission_metallic.rgb; }
    if (material.flags.x & 2u) != 0u {
        out.texel.direct.other += material.base_roughness.rgb * f32((groups & DDGI_OTHER) != 0u);
        return out;
    }
    let v = -direction;
    if (groups & DDGI_SUN) != 0u {
        // White unit sun: its colour and intensity are applied when the groups are composed.
        let sun = -frame.sun_dir.xyz;
        let nl = saturate(dot(n, sun));
        if nl > 0.0 && dot(ng, sun) > 0.0
            && closest(Ray(p + ng * (2.0 * EPS), sun), frame.clip.y, true).object == MISS {
            out.texel.direct.sun = direct_brdf(material, n, v, sun, true) * nl;
        }
    }
    if (groups & DDGI_OTHER) != 0u {
        out.texel.direct.other += ddgi_local_lights(p, ng, n, v, material, seed);
    }
    if (ddgi_params.counts.w & DDGI_BOUNCE) != 0u {
        let bounce = ddgi_bounce(p, n, v, groups, seed);
        let albedo = diffuse(material);
        if (groups & DDGI_SUN) != 0u {
            out.texel.direct.sun += albedo * bounce.direct.sun;
            out.texel.direct.residual = albedo * bounce.direct.residual;
        }
        if (groups & DDGI_OTHER) != 0u { out.texel.direct.other += albedo * bounce.direct.other; }
        if (groups & DDGI_GEOMETRY) != 0u {
            // Element 3k+c is channel c: the albedo of a pair alternates with the pair's parity.
            for (var j = 0u; j < 14u; j++) {
                let c = (2u * j) % 3u;
                out.texel.transfer[j] = vec2<f32>(albedo[c], albedo[(c + 1u) % 3u]) * bounce.transfer[j];
            }
            out.bounced = true;
        }
    }
    return out;
}

// Unshadowed diffuse contribution of a point, spot or directional light (rgb) and its
// distance (w, the shadow ray length); zero when it cannot reach the hit.
fn ddgi_light_unshadowed(light: Light, p: vec3<f32>, ng: vec3<f32>, n: vec3<f32>, v: vec3<f32>,
                         mat: Material) -> vec4<f32> {
    if !any(light.radiance.rgb > vec3<f32>(0.0)) { return vec4<f32>(0.0); }
    let travel = light.direction_outer.xyz;
    let delta = light.position_range.xyz - p;
    var distance = length(delta);
    var l = delta / max(distance, EPS);
    var weight = 1.0;
    if light.kind == 2u {
        l = -travel;
        distance = frame.clip.y;
    } else {
        if distance >= light.position_range.w { return vec4<f32>(0.0); }
        let window = saturate(1.0 - pow(distance / light.position_range.w, 4.0));
        weight = window * window / max(distance * distance, 0.04);
        if light.kind == 1u { weight *= smoothstep(light.direction_outer.w, light.inner, dot(travel, -l)); }
    }
    let nl = saturate(dot(n, l));
    if nl <= 0.0 || dot(ng, l) <= 0.0 || weight <= 0.0 { return vec4<f32>(0.0); }
    return vec4<f32>(direct_brdf(mat, n, v, l, true) * (nl * weight * light.radiance.rgb), distance);
}

fn ddgi_luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Local lights at a probe-ray hit with ONE shadow ray: a point, spot or directional light is
// picked in proportion to its unshadowed contribution and the estimate is their unshadowed sum
// times that light's visibility (unbiased; the probe hysteresis averages the noise). Area panels
// keep the shared one-tap estimator with its own shadow ray. Every light whose unshadowed
// contribution is noticeable sets its bit of the probe's light mask.
fn ddgi_mark_light(i: u32, luma: f32) {
    if luma > ddgi_params.extra.w { atomicOr(&ddgi_light_bits[i >> 5u], 1u << (i & 31u)); }
}

fn ddgi_local_lights(p: vec3<f32>, ng: vec3<f32>, n: vec3<f32>, v: vec3<f32>, mat: Material,
                     seed: ptr<function, u32>) -> vec3<f32> {
    let count = min(256u, frame.light_info.x);
    let shadow_origin = p + ng * (2.0 * EPS);
    var panels = vec3<f32>(0.0);
    var total = 0.0;
    for (var i = 0u; i < count; i++) {
        let light = lights[i];
        if light.kind == 3u {
            var light_seed = *seed;
            let panel = mesh_local_at_hit(i, shadow_origin, p, ng, n, v, mat, &light_seed, true, false);
            ddgi_mark_light(i, ddgi_luma(panel));
            panels += panel;
            continue;
        }
        let luma = ddgi_luma(ddgi_light_unshadowed(light, p, ng, n, v, mat).rgb);
        ddgi_mark_light(i, luma);
        total += luma;
    }
    if total <= 0.0 { return panels; }
    let pick = random(seed) * total;
    var running = 0.0;
    for (var i = 0u; i < count; i++) {
        let light = lights[i];
        if light.kind == 3u { continue; }
        let c = ddgi_light_unshadowed(light, p, ng, n, v, mat);
        let luma = ddgi_luma(c.rgb);
        running += luma;
        if luma <= 0.0 || running < pick { continue; }
        var l = -light.direction_outer.xyz;
        if light.kind != 2u { l = normalize(light.position_range.xyz - p); }
        if closest(Ray(shadow_origin, l), c.w - 2.0 * EPS, true).object == MISS {
            return panels + c.rgb * (total / luma);
        }
        return panels;
    }
    return panels;
}

fn ddgi_pack_irr(a: vec3<f32>, b: vec3<f32>) -> vec4<u32> {
    let x = clamp(a, vec3<f32>(0.0), vec3<f32>(65504.0));
    let y = clamp(b, vec3<f32>(0.0), vec3<f32>(65504.0));
    return vec4<u32>(pack2x16float(x.xy), pack2x16float(vec2<f32>(x.z, 0.0)),
        pack2x16float(y.xy), pack2x16float(vec2<f32>(y.z, 0.0)));
}

// Writes the bordered irradiance tile of `slot` from `ddgi_irr`, whole rows per thread.
fn ddgi_write_irr_tile(v: RaaDdgiVolume, slot: u32, lid: u32) {
    let irr_first = slot * RAA_DDGI_IRR_TEXELS;
    for (var t = 2u * lid; t < RAA_DDGI_IRR_TEXELS; t += 2u * DDGI_THREADS) {
        let a = ddgi_border_source(vec2<u32>(t % RAA_DDGI_IRR_TILE, t / RAA_DDGI_IRR_TILE), RAA_DDGI_IRR);
        let u = t + 1u;
        let b = ddgi_border_source(vec2<u32>(u % RAA_DDGI_IRR_TILE, u / RAA_DDGI_IRR_TILE), RAA_DDGI_IRR);
        ddgi_probes[v.irr_row + (irr_first + t) / 2u] =
            ddgi_pack_irr(ddgi_irr[a.x + a.y * RAA_DDGI_IRR], ddgi_irr[b.x + b.y * RAA_DDGI_IRR]);
    }
}

fn ddgi_max3(v: vec3<f32>) -> f32 {
    return max(max(abs(v.x), abs(v.y)), abs(v.z));
}

// Blend weight of the stored value: a young group averages its first updates progressively
// (weight 1/(age+1)) before the configured hysteresis takes over; a large change shortens it.
fn ddgi_hysteresis(age: u32, fresh: bool, change: f32) -> f32 {
    let a = f32(age);
    var h = select(min(ddgi_params.tuning.x, a / (a + 1.0)), 0.0, fresh);
    if change > ddgi_params.tuning.z { h = max(0.0, h - 0.15); }
    return h;
}

@compute @workgroup_size(64, 1, 1)
fn ddgi_update(@builtin(local_invocation_index) lid: u32,
               @builtin(workgroup_id) group: vec3<u32>) {
    // The cascade of the job's slot: its grid, spacing, biases and distance cap.
    let v = raa_ddgi_volume_at(raa_ddgi_cascade_of(ddgi_jobs[2u * group.x] & 0xffffu));
    let spacing_scale = v.spacing / max(raa_ddgi_volume().spacing, 1e-6);
    let rays = min(ddgi_params.counts.x, DDGI_MAX_RAYS);
    if lid == 0u {
        let job = ddgi_jobs[2u * group.x];
        let job_groups = ddgi_jobs[2u * group.x + 1u];
        let slot = job & 0xffffu;
        let g = ddgi_slot_coord(v, slot);
        let row = v.meta_row + RAA_DDGI_META_ROWS * slot;
        var record = ddgi_probes[row];
        var offset = bitcast<vec3<f32>>(ddgi_probes[row + 1u].xyz);
        let fresh = any(bitcast<vec3<i32>>(record.xyz) != g) || (record.w & RAA_DDGI_VALID) == 0u;
        if fresh {
            record = vec4<u32>(bitcast<vec3<u32>>(g), 0u);
            offset = vec3<f32>(0.0);
        }
        let turn = (record.w >> 8u) & 0xffu;
        var run = 1u;
        if !fresh && (record.w & RAA_DDGI_INACTIVE) != 0u && (turn & 7u) != 0u {
            run = 0u;
            ddgi_probes[row] = vec4<u32>(record.xyz, (record.w & 0xffu) | (((turn + 1u) & 0xffu) << 8u));
        }
        ddgi_job = vec4<u32>(slot, run, u32(fresh), job >> 16u);
        // A fresh probe has no group to keep.
        ddgi_job_groups = vec4<u32>(job_groups & 0xffu, (job_groups >> 8u) & 0xffu,
            select((job_groups >> 16u) & 7u, 7u, fresh), select((job_groups >> 24u) & 1u, 1u, fresh));
        ddgi_origin = vec4<f32>(vec3<f32>(g) * v.spacing + offset, 0.0);
    }
    let job = workgroupUniformLoad(&ddgi_job);
    let ages = workgroupUniformLoad(&ddgi_job_groups);
    if job.y == 0u { return; }
    let origin = workgroupUniformLoad(&ddgi_origin).xyz;
    let slot = job.x;
    let fresh = job.z != 0u;
    let groups = ages.z;
    let geometry = (groups & DDGI_GEOMETRY) != 0u;
    let scratch = ddgi_scratch_row() + group.x * rays * DDGI_SCRATCH_ROWS;

    // 1. Trace.
    let limit = ddgi_params.tuning.y;
    for (var r = lid; r < rays; r += DDGI_THREADS) {
        let direction = ddgi_ray_direction(r, rays);
        var seed = (slot * 747796405u + r * 277803737u + ddgi_params.counts.z * 2891336453u) ^ 0x85ebca6bu;
        let shaded = ddgi_shade(origin, direction, limit, groups, &seed);
        let d = shaded.texel.direct;
        ddgi_ray[r] = ddgi_direct_pack(d);
        ddgi_direction[r] = vec4<f32>(direction, clamp(d.other.z, -65504.0, 65504.0));
        ddgi_distance[r] = shaded.distance;
        // Bounced transfer of a hit: 14 pairs, then a flag in the last pair of the 4 scratch rows.
        if geometry {
            let at = scratch + r * DDGI_SCRATCH_ROWS;
            let t = shaded.texel.transfer;
            ddgi_probes[at] = vec4<u32>(ddgi_pack(t[0]), ddgi_pack(t[1]), ddgi_pack(t[2]), ddgi_pack(t[3]));
            ddgi_probes[at + 1u] = vec4<u32>(ddgi_pack(t[4]), ddgi_pack(t[5]), ddgi_pack(t[6]), ddgi_pack(t[7]));
            ddgi_probes[at + 2u] = vec4<u32>(ddgi_pack(t[8]), ddgi_pack(t[9]), ddgi_pack(t[10]), ddgi_pack(t[11]));
            ddgi_probes[at + 3u] = vec4<u32>(ddgi_pack(t[12]), ddgi_pack(t[13]), 0u, u32(shaded.bounced));
        }
    }
    workgroupBarrier();
    storageBarrier();

    // 2. Relocation and classification of this probe, and its light mask (one thread).
    if lid == 0u {
        let row = v.meta_row + RAA_DDGI_META_ROWS * slot;
        if geometry {
            let g = ddgi_slot_coord(v, slot);
            var offset = select(bitcast<vec3<f32>>(ddgi_probes[row + 1u].xyz), vec3<f32>(0.0), fresh);
            var backfaces = 0u;
            var back_distance = DDGI_MISS;
            var back_dir = vec3<f32>(0.0);
            var front_distance = DDGI_MISS;
            var front_dir = vec3<f32>(0.0);
            var near_surface = false;
            for (var r = 0u; r < rays; r++) {
                let d = ddgi_distance[r];
                let dir = ddgi_direction[r].xyz;
                if d < 0.0 {
                    backfaces++;
                    if -5.0 * d < back_distance { back_distance = -5.0 * d; back_dir = dir; }
                } else if d < DDGI_MISS {
                    if d < front_distance { front_distance = d; front_dir = dir; }
                    let reach = abs(dir * d);
                    if all(reach <= vec3<f32>(v.spacing)) { near_surface = true; }
                }
            }
            let backface_fraction = f32(backfaces) / f32(max(rays, 1u));
            let inside = backface_fraction > ddgi_params.extra.y;
            let flags = ddgi_params.counts.w;
            if (flags & DDGI_RELOCATE) != 0u {
                let min_front = ddgi_params.tuning.w * spacing_scale;
                var candidate = offset;
                if inside {
                    candidate = offset + back_dir * (back_distance + 0.5 * min_front);
                } else if front_distance < min_front {
                    candidate = offset - front_dir * (min_front - front_distance);
                }
                if all(abs(candidate) <= vec3<f32>(0.45 * v.spacing)) { offset = candidate; }
            }
            var state = RAA_DDGI_VALID;
            if (flags & DDGI_CLASSIFY) != 0u {
                if inside { state |= RAA_DDGI_INSIDE | RAA_DDGI_INACTIVE; }
                if !near_surface { state |= RAA_DDGI_INACTIVE; }
            }
            let record = ddgi_probes[row];
            let turn = select(((record.w >> 8u) + 1u) & 0xffu, 1u, fresh);
            ddgi_probes[row] = vec4<u32>(bitcast<vec3<u32>>(g), state | (turn << 8u));
            ddgi_probes[row + 1u] = vec4<u32>(bitcast<vec3<u32>>(offset), 0u);
        }
        if (groups & DDGI_OTHER) != 0u {
            var lo = vec4<u32>(atomicLoad(&ddgi_light_bits[0]), atomicLoad(&ddgi_light_bits[1]),
                atomicLoad(&ddgi_light_bits[2]), atomicLoad(&ddgi_light_bits[3]));
            var hi = vec4<u32>(atomicLoad(&ddgi_light_bits[4]), atomicLoad(&ddgi_light_bits[5]),
                atomicLoad(&ddgi_light_bits[6]), atomicLoad(&ddgi_light_bits[7]));
            if ages.w == 0u {
                lo |= ddgi_probes[row + 2u];
                hi |= ddgi_probes[row + 3u];
            }
            ddgi_probes[row + 2u] = lo;
            ddgi_probes[row + 3u] = hi;
        }
    }

    // 3. Blend the groups of one irradiance texel per thread, and four depth moments.
    {
        let xy = vec2<u32>(lid % RAA_DDGI_IRR, lid / RAA_DDGI_IRR);
        let dir = raa_ddgi_oct_decode((vec2<f32>(xy) + 0.5) / f32(RAA_DDGI_IRR) * 2.0 - 1.0);
        var sum: DdgiTexel;
        var escaped: array<f32, 9>;
        var total = 0.0;
        for (var r = 0u; r < rays; r++) {
            let distance = ddgi_distance[r];
            if distance < 0.0 { continue; }
            let ray_dir = ddgi_direction[r];
            let w = dot(dir, ray_dir.xyz);
            if w <= 0.0 { continue; }
            let shaded = ddgi_ray[r];
            let a = unpack2x16float(shaded.y);
            sum.direct.sun += w * vec3<f32>(unpack2x16float(shaded.x), a.x);
            sum.direct.residual += w * vec3<f32>(a.y, unpack2x16float(shaded.z));
            sum.direct.other += w * vec3<f32>(unpack2x16float(shaded.w), ray_dir.w);
            total += w;
            if !geometry { continue; }
            if distance >= DDGI_MISS {
                // An escaped ray sees the sky basis itself, white.
                let y = ddgi_sh(ray_dir.xyz);
                for (var k = 0u; k < 9u; k++) { escaped[k] += w * y[k]; }
                continue;
            }
            let at = scratch + r * DDGI_SCRATCH_ROWS;
            let r3 = ddgi_probes[at + 3u];
            if r3.w == 0u { continue; }
            let r0 = ddgi_probes[at];
            let r1 = ddgi_probes[at + 1u];
            let r2 = ddgi_probes[at + 2u];
            sum.transfer[0] += w * unpack2x16float(r0.x);
            sum.transfer[1] += w * unpack2x16float(r0.y);
            sum.transfer[2] += w * unpack2x16float(r0.z);
            sum.transfer[3] += w * unpack2x16float(r0.w);
            sum.transfer[4] += w * unpack2x16float(r1.x);
            sum.transfer[5] += w * unpack2x16float(r1.y);
            sum.transfer[6] += w * unpack2x16float(r1.z);
            sum.transfer[7] += w * unpack2x16float(r1.w);
            sum.transfer[8] += w * unpack2x16float(r2.x);
            sum.transfer[9] += w * unpack2x16float(r2.y);
            sum.transfer[10] += w * unpack2x16float(r2.z);
            sum.transfer[11] += w * unpack2x16float(r2.w);
            sum.transfer[12] += w * unpack2x16float(r3.x);
            sum.transfer[13] += w * unpack2x16float(r3.y);
        }
        let row = ddgi_group_row() + (slot * RAA_DDGI_IRR * RAA_DDGI_IRR + lid) * DDGI_GROUP_ROWS;
        var texel = ddgi_texel_load(row);
        if fresh { texel = DdgiTexel(); }
        if total > 1e-4 {
            let inv = 1.0 / total;
            if (groups & DDGI_SUN) != 0u {
                let sun = sum.direct.sun * inv;
                let residual = sum.direct.residual * inv;
                let change = max(ddgi_max3((sun - texel.direct.sun) * frame.sun_color.rgb),
                    ddgi_params.extra.x * ddgi_sky_norm() * ddgi_max3(residual - texel.direct.residual));
                let h = ddgi_hysteresis(ages.x, fresh, change);
                texel.direct.sun = mix(sun, texel.direct.sun, h);
                texel.direct.residual = mix(residual, texel.direct.residual, h);
            }
            if (groups & DDGI_OTHER) != 0u {
                let other = sum.direct.other * inv;
                let h = ddgi_hysteresis(ages.y, fresh, ddgi_max3(other - texel.direct.other));
                texel.direct.other = mix(other, texel.direct.other, h);
            }
            if geometry {
                for (var e = 0u; e < 27u; e++) { sum.transfer[e >> 1u][e & 1u] += escaped[e / 3u]; }
                let h = ddgi_hysteresis(job.w, fresh, 0.0);
                for (var j = 0u; j < 14u; j++) {
                    texel.transfer[j] = mix(sum.transfer[j] * inv, texel.transfer[j], h);
                }
            }
        }
        if fresh || total > 1e-4 { ddgi_texel_store(row, texel, geometry); }
        ddgi_irr[lid] = ddgi_compose(texel);
    }
    if geometry {
        let hysteresis = ddgi_hysteresis(job.w, fresh, 0.0);
        let max_depth = v.max_depth;
        for (var k = 0u; k < 4u; k++) {
            let index = lid + k * DDGI_THREADS;
            let xy = vec2<u32>(index % RAA_DDGI_DEPTH, index / RAA_DDGI_DEPTH);
            let texel = raa_ddgi_oct_decode((vec2<f32>(xy) + 0.5) / f32(RAA_DDGI_DEPTH) * 2.0 - 1.0);
            var sum = vec2<f32>(0.0);
            var total = 0.0;
            for (var r = 0u; r < rays; r++) {
                let c = max(0.0, dot(texel, ddgi_direction[r].xyz));
                if c <= 0.0 { continue; }
                let w = pow(c, 50.0);
                let d = min(abs(ddgi_distance[r]), max_depth);
                sum += w * vec2<f32>(d, d * d);
                total += w;
            }
            let old = raa_ddgi_depth_texel(v, slot, xy + 1u);
            var result = old;
            if total > 1e-6 {
                result = mix(sum / total, old, hysteresis);
            }
            ddgi_depth_tile[index] = result;
        }
    }
    workgroupBarrier();

    // 4. Write both bordered tiles, whole rows per thread (two irradiance or four depth texels).
    ddgi_write_irr_tile(v, slot, lid);
    if geometry {
        let depth_first = slot * RAA_DDGI_DEPTH_TEXELS;
        for (var t = 4u * lid; t < RAA_DDGI_DEPTH_TEXELS; t += 4u * DDGI_THREADS) {
            var words = vec4<u32>(0u);
            for (var k = 0u; k < 4u; k++) {
                let u = t + k;
                let s = ddgi_border_source(vec2<u32>(u % RAA_DDGI_DEPTH_TILE, u / RAA_DDGI_DEPTH_TILE), RAA_DDGI_DEPTH);
                words[k] = pack2x16float(ddgi_depth_tile[s.x + s.y * RAA_DDGI_DEPTH]);
            }
            ddgi_probes[v.depth_row + (depth_first + t) / 4u] = words;
        }
    }
}

// Relight: one workgroup rebuilds the composed irradiance tile of one slot from its stored
// groups under the frame's sun colour, sky SH and sky scale. No ray is traced.
@compute @workgroup_size(64, 1, 1)
fn ddgi_relight(@builtin(local_invocation_index) lid: u32,
                @builtin(workgroup_id) group: vec3<u32>) {
    let v = raa_ddgi_volume();
    let slot = group.x;
    let row = ddgi_group_row() + (slot * RAA_DDGI_IRR * RAA_DDGI_IRR + lid) * DDGI_GROUP_ROWS;
    ddgi_irr[lid] = ddgi_compose(ddgi_texel_load(row));
    workgroupBarrier();
    ddgi_write_irr_tile(v, slot, lid);
}

// Sky SH: the frame's sky radiance (unit sky scale, the cone around the sun left out) projected
// on the order-2 SH basis from fixed spherical-Fibonacci directions; written to the header's sky
// rows, where the update and the relight read it. One workgroup.
@compute @workgroup_size(64, 1, 1)
fn ddgi_sky_project(@builtin(local_invocation_index) lid: u32) {
    var acc: array<vec3<f32>, 9>;
    for (var i = lid; i < DDGI_SKY_SAMPLES; i += DDGI_THREADS) {
        let z = 1.0 - (2.0 * f32(i) + 1.0) / f32(DDGI_SKY_SAMPLES);
        let r = sqrt(max(0.0, 1.0 - z * z));
        let phi = f32(i) * 2.39996323;
        let d = vec3<f32>(r * cos(phi), r * sin(phi), z);
        if ddgi_in_sun_cone(d) { continue; }
        let rgb = ddgi_sky(d);
        let y = ddgi_sh(d);
        for (var k = 0u; k < 9u; k++) { acc[k] += rgb * y[k]; }
    }
    ddgi_sky_partial[lid] = acc;
    workgroupBarrier();
    if lid == 0u {
        let row = ddgi_sky_row();
        for (var k = 0u; k < 9u; k++) {
            var sum = vec3<f32>(0.0);
            for (var t = 0u; t < DDGI_THREADS; t++) { sum += ddgi_sky_partial[t][k]; }
            let coefficient = sum * (4.0 * PI / f32(DDGI_SKY_SAMPLES));
            ddgi_probes[row + k] = vec4<u32>(bitcast<vec3<u32>>(coefficient), 0u);
            if k == 0u {
                // Band 0 is the mean radiance x 2 sqrt(pi).
                let norm = ddgi_luma(coefficient) * 0.282095;
                ddgi_probes[row + 9u] = vec4<u32>(bitcast<u32>(max(norm, 0.0)), 0u, 0u, 0u);
            }
        }
    }
}

// Screen resolve: the volume interpolated once per resolve texel (row 9: first row, width, height,
// scale; a texel covers scale x scale pixels and samples the prepass at its first pixel), so the
// forward shader reads up to four rows instead of eight probes. A row holds rgb and the blend
// weight (f16) and the camera distance of the surface it was resolved for (f32); forward keeps
// the texels resolved for its own surface and falls back to the full lookup when none was.
@compute @workgroup_size(8, 8, 1)
fn ddgi_resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    let screen = raa_ddgi_word(9u);
    if id.x >= screen.y || id.y >= screen.z { return; }
    let row = screen.x + id.y * screen.y + id.x;
    let full = textureDimensions(depth);
    let pixel = min(id.xy * screen.w, full - 1u);
    let z = textureLoad(depth, vec2<i32>(pixel), 0);
    let normal_view = textureLoad(normals, vec2<i32>(pixel), 0).xyz;
    if z >= 1.0 || dot(normal_view, normal_view) <= 0.0 {
        ddgi_probes[row] = vec4<u32>(0u);
        return;
    }
    let p = world_at((vec2<f32>(pixel) + 0.5) / vec2<f32>(full), z);
    let n = normalize((frame.inv_view * vec4<f32>(normal_view, 0.0)).xyz);
    let to_eye = frame.camera_pos.xyz - p;
    let distance = length(to_eye);
    var sample = raa_ddgi_sample(p, n, to_eye / max(distance, 1e-6));
    if ((raa_ddgi_word(4u).w >> 8u) & 0xffu) == 4u {
        // Debug "probes": red = cascade 0 weight, green = coarser cascades, blue = hole filled by
        // inactive probes; black = no usable probe (forward then takes the flat ambient term).
        sample = vec4<f32>(raa_ddgi_diag * vec3<f32>(1.0, 1.0, 0.0) + vec3<f32>(0.0, 0.0, raa_ddgi_diag.z), 1.0);
    }
    let rgb = clamp(sample.rgb, vec3<f32>(0.0), vec3<f32>(65504.0));
    ddgi_probes[row] = vec4<u32>(pack2x16float(rgb.xy), pack2x16float(vec2<f32>(rgb.z, sample.w)),
        bitcast<u32>(distance), 1u);
}
