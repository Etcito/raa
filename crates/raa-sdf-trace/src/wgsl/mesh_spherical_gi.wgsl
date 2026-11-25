// One 8×8 workgroup traces one scheduled face through the shared mesh BVH from a
// free-space probe. The packed 48-byte texel also keeps world-space hit position,
// validity and normal. The normal's fourth half is the mirror flag (0 or 1) on a
// front face; a back face stores -0.25 (or 1.25 for a mirror), so `> 0.5` still
// means mirror and `< 0 || > 1.1` means back face.
// hit_position.w is 0 for a miss; a hit stores a float in [1, 2) whose 22 low mantissa
// bits hold the geometric normal (11-bit octahedral pair), so `> 0.5` still means hit.
// The free fourth half of the sky, sun and other words holds the hit's diffuse albedo
// (r, g, b): with the hit position and both normals, the sun and sky groups can be
// re-shaded in place (spherical_reshade) without the primary ray.
struct SphericalFaceWork {
    position: vec4<f32>,
    normal: vec4<f32>,
    ids: vec4<u32>, // slot, face or irradiance index, unused, unused
}

struct SphericalTexel {
    hit_position: vec4<f32>,
    sky: vec2<u32>,
    sun: vec2<u32>,
    other: vec2<u32>,
    hit_normal: vec2<u32>,
}

@group(0) @binding(21) var<storage, read> spherical_work: array<SphericalFaceWork>;
@group(0) @binding(22) var<storage, read_write> spherical_faces: array<SphericalTexel>;
@group(0) @binding(23) var<storage, read_write> spherical_lookup: array<vec4<u32>>;

struct SphericalFaceLighting {
    masks: array<u32, 8>,
    low: vec4<u32>, // sortable float xyz, w = hit count
    high: vec4<u32>, // sortable float xyz, w = sky/sun contribution bits
}

@group(0) @binding(24) var<storage, read_write> spherical_face_lighting: array<SphericalFaceLighting>;

var<workgroup> spherical_light_masks: array<atomic<u32>, 8>;
var<workgroup> spherical_hit_min: array<atomic<u32>, 3>;
var<workgroup> spherical_hit_max: array<atomic<u32>, 3>;
var<workgroup> spherical_hit_count: atomic<u32>;
var<workgroup> spherical_global_sources: atomic<u32>;

fn spherical_ordered_float(value: f32) -> u32 {
    let bits = bitcast<u32>(value);
    return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

// rgb in three halves and a fourth value (the albedo channel) in the last half. Signed:
// the sky residual of the transfer may be negative.
fn spherical_pack(rgb: vec3<f32>, extra: f32) -> vec2<u32> {
    let safe = clamp(rgb, vec3<f32>(-65504.0), vec3<f32>(65504.0));
    return vec2<u32>(pack2x16float(safe.xy), pack2x16float(vec2<f32>(safe.z, extra)));
}

fn spherical_extra(words: vec2<u32>) -> f32 {
    return unpack2x16float(words.y).y;
}

fn spherical_albedo(texel: SphericalTexel) -> vec3<f32> {
    return vec3<f32>(spherical_extra(texel.sky), spherical_extra(texel.sun), spherical_extra(texel.other));
}

fn spherical_hit_word(ng: vec3<f32>) -> f32 {
    let q = vec2<u32>(round((spherical_oct_encode(ng) * 0.5 + 0.5) * 2047.0));
    return bitcast<f32>(0x3f800000u | (min(q.x, 2047u) << 11u) | min(q.y, 2047u));
}

fn spherical_hit_normal(word: f32) -> vec3<f32> {
    let bits = bitcast<u32>(word);
    let e = vec2<f32>(f32((bits >> 11u) & 2047u), f32(bits & 2047u)) / 2047.0 * 2.0 - 1.0;
    var v = vec3<f32>(e, 1.0 - abs(e.x) - abs(e.y));
    if v.z < 0.0 {
        let s = select(vec2<f32>(-1.0), vec2<f32>(1.0), v.xy >= vec2<f32>(0.0));
        v = vec3<f32>((1.0 - abs(v.yx)) * s, v.z);
    }
    return normalize(v);
}

// ---- Sky transfer (handoff/sky-transfer.md, ported from DDGI) ----
// The transfer buffer: header rows 0-8 = sky SH L_k (rgb f32, the frame's environment with
// the cone around the sun left out, written by spherical_sky_project), row 9 x = sky norm
// (the same value as spherical_sky_norm), row 10 = up to four dense light indices of a reach
// test and row 11 x their count (CPU). Then two rows per (slot, patch): nine f16 visibility
// transfer coefficients T_k (sum over the patch's missed texels of Y_k(w) dw) and the
// residual R (rgb f16, sum of (sky - SH sky) / norm dw). The composed sky miss flux of a
// patch, in unit-sky units, is sum_k T_k L_k / norm + R (spherical_relight).
@group(0) @binding(34) var<storage, read_write> spherical_transfer: array<vec4<u32>>;
@group(0) @binding(35) var<storage, read_write> spherical_reach: array<u32>;
const SPHERICAL_TRANSFER_HEADER: u32 = 16u;
const SPHERICAL_TRANSFER_NORM_ROW: u32 = 9u;
const SPHERICAL_REACH_ROW: u32 = 10u;
const SPHERICAL_FLAG_TRANSFER: u32 = 1u;
// Capture rays of a probe inside a solid skip the surfaces met from behind (options[1].y).
const SPHERICAL_FLAG_INSIDE: u32 = 2u;
const SPHERICAL_INSIDE_CROSSINGS: u32 = 8u;
const SPHERICAL_SKY_PROJECT_SAMPLES: u32 = 4096u;
// Keep in sync with spherical_gpu.rs SLOT_ROWS and RAA_SPHERICAL_SLOT_ROWS.
const SPHERICAL_SLOT_ROWS: u32 = 167u;

fn spherical_transfer_on() -> bool {
    return (options[1].y & SPHERICAL_FLAG_TRANSFER) != 0u;
}

// The sun disc belongs to the sun group: a cone around it is left out of the sky.
fn spherical_in_sun_cone(direction: vec3<f32>) -> bool {
    return dot(direction, -frame.sun_dir.xyz) > bitcast<f32>(options[2].z);
}

// The 9 real SH basis functions (order 2), raa-ibl order (as ddgi_sh).
fn spherical_sh(d: vec3<f32>) -> array<f32, 9> {
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

fn spherical_sky_coefficient(k: u32) -> vec3<f32> {
    return bitcast<vec3<f32>>(spherical_transfer[k].xyz);
}

fn spherical_sh_sky(direction: vec3<f32>) -> vec3<f32> {
    let y = spherical_sh(direction);
    var rgb = vec3<f32>(0.0);
    for (var k = 0u; k < 9u; k++) { rgb += y[k] * spherical_sky_coefficient(k); }
    return rgb;
}

fn spherical_transfer_norm() -> f32 {
    return bitcast<f32>(spherical_transfer[SPHERICAL_TRANSFER_NORM_ROW].x);
}

fn spherical_unpack(words: vec2<u32>) -> vec3<f32> {
    return vec3<f32>(unpack2x16float(words.x), unpack2x16float(words.y).x);
}

// Sky norm: mean luminance of 64 fixed Fibonacci directions of the environment cube at
// its last mip, computed from the very cube the capture samples in the same frame (the
// published SH lags it by the IBL readback). Every thread of a 64-thread workgroup
// calls it; 0 without an environment.
var<workgroup> spherical_norm_sum: array<f32, 64>;
fn spherical_sky_norm(px: u32) -> f32 {
    var l = 0.0;
    if options[0].z != 0u {
        let z = 1.0 - (2.0 * f32(px) + 1.0) / 64.0;
        let r = sqrt(max(0.0, 1.0 - z * z));
        let phi = 2.3999632 * f32(px);
        let level = f32(textureNumLevels(sky) - 1u);
        let c = textureSampleLevel(sky, raa_linear_clamp, vec3<f32>(r * cos(phi), z, r * sin(phi)), level).rgb;
        l = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    }
    spherical_norm_sum[px] = l;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if px < stride { spherical_norm_sum[px] += spherical_norm_sum[px + stride]; }
        workgroupBarrier();
    }
    let norm = spherical_norm_sum[0] / 64.0;
    workgroupBarrier();
    return norm;
}

// Unit-intensity sky group: the environment divided by its norm, or a unit ambient.
// The lookup scales it back (header row 69, written by spherical_scales).
fn spherical_sky(direction: vec3<f32>, inv_norm: f32) -> vec3<f32> {
    if options[0].z == 0u { return vec3<f32>(1.0); }
    return inv_norm * textureSampleLevel(sky, raa_linear_clamp, direction, 0.0).rgb;
}

// What a missed texel stores in the sky group: the unit sky, or with the transfer only the
// residual the order-2 SH does not carry (zero in the sun cone).
fn spherical_miss_sky(direction: vec3<f32>, inv_norm: f32) -> vec3<f32> {
    if !spherical_transfer_on() { return spherical_sky(direction, inv_norm); }
    if spherical_in_sun_cone(direction) { return vec3<f32>(0.0); }
    return inv_norm * (textureSampleLevel(sky, raa_linear_clamp, direction, 0.0).rgb
        - spherical_sh_sky(direction));
}

// Sky reaching a hit through a bounce ray (unit sky; the sun disc left out with the transfer).
fn spherical_bounce_sky(direction: vec3<f32>, inv_norm: f32) -> vec3<f32> {
    if spherical_transfer_on() && spherical_in_sun_cone(direction) { return vec3<f32>(0.0); }
    return spherical_sky(direction, inv_norm);
}

// Cosine sky samples at a hit (the seed continues the capture's own sequence).
fn spherical_sky_bounce(p: vec3<f32>, ng: vec3<f32>, n: vec3<f32>, albedo: vec3<f32>,
                        limit: f32, inv_norm: f32, seed: ptr<function, u32>) -> vec3<f32> {
    let sky_samples = options[3].y;
    var sky_sum = vec3<f32>(0.0);
    for (var k = 0u; k < sky_samples; k++) {
        let u = random(seed);
        let phi = 2.0 * PI * random(seed);
        let bounce = basis(n) * vec3<f32>(sqrt(u) * cos(phi), sqrt(u) * sin(phi), sqrt(1.0 - u));
        if dot(ng, bounce) > 0.0 && closest(Ray(p + ng * (2.0 * EPS), bounce), limit, true).object == MISS {
            sky_sum += spherical_bounce_sky(bounce, inv_norm);
        }
    }
    if sky_samples == 0u { return vec3<f32>(0.0); }
    return albedo * sky_sum / f32(sky_samples);
}

// The capture's per-texel random sequence (also replayed by the re-shade).
fn spherical_seed(slot: u32, face: u32, px: u32) -> u32 {
    return (slot * 747796405u + face * 277803737u + px) ^ 0x85ebca6bu;
}

// Unit sun at a stored hit: diffuse albedo / pi x n.l x one shadow ray.
fn spherical_unit_sun_albedo(p: vec3<f32>, ng: vec3<f32>, n: vec3<f32>, albedo: vec3<f32>) -> vec3<f32> {
    let sun = -frame.sun_dir.xyz;
    let nl = saturate(dot(n, sun));
    if nl > 0.0 && dot(ng, sun) > 0.0
        && closest(Ray(p + ng * (2.0 * EPS), sun), frame.clip.y, true).object == MISS {
        return albedo * (nl / PI);
    }
    return vec3<f32>(0.0);
}

// Unit-intensity sun group: the response to a white unit sun (mesh_sun_at_hit without
// frame.sun_color and without its colour gate), scaled at lookup by header row 68.
fn spherical_unit_sun(p: vec3<f32>, ng: vec3<f32>, n: vec3<f32>, v: vec3<f32>,
                      mat: Material) -> vec3<f32> {
    let origin = p + ng * (2.0 * EPS);
    let sun = -frame.sun_dir.xyz;
    let nl = saturate(dot(n, sun));
    if nl > 0.0 && dot(ng, sun) > 0.0
        && closest(Ray(origin, sun), frame.clip.y, true).object == MISS {
        return direct_brdf(mat, n, v, sun, true) * nl;
    }
    return vec3<f32>(0.0);
}

const SPHERICAL_SUN_SCALE_ROW: u32 = 68u;
const SPHERICAL_SKY_SCALE_ROW: u32 = 69u;
// Group scales for the lookup, every frame before the resolve: row 68 = sun radiance,
// row 69 = sky diffuse x sky norm (or the ambient colour without an environment).
@compute @workgroup_size(64, 1, 1)
fn spherical_scales(@builtin(local_invocation_index) px: u32) {
    let norm = spherical_sky_norm(px);
    if px == 0u {
        var sky_scale = frame.ambient.rgb;
        if options[0].z != 0u { sky_scale = vec3<f32>(bitcast<f32>(options[2].x) * norm); }
        spherical_lookup[SPHERICAL_SUN_SCALE_ROW] = bitcast<vec4<u32>>(vec4<f32>(frame.sun_color.rgb, 0.0));
        spherical_lookup[SPHERICAL_SKY_SCALE_ROW] = bitcast<vec4<u32>>(vec4<f32>(sky_scale, 0.0));
    }
}

@compute @workgroup_size(8, 8, 1)
fn spherical_capture(@builtin(global_invocation_id) id: vec3<u32>) {
    let work = spherical_work[id.z];
    let px = id.x + 8u * id.y;
    if px < 3u {
        atomicStore(&spherical_hit_min[px], 0xffffffffu);
    }
    let sky_norm = spherical_sky_norm(px);
    let inv_norm = select(0.0, 1.0 / sky_norm, sky_norm > 0.0);
    let uv = (vec2<f32>(id.xy) + 0.5) * 0.25 - 1.0;
    let direction = normalize(raa_spherical_face_direction(work.ids.y, uv));
    let origin = work.position.xyz + work.normal.xyz * (2.0 * EPS);
    let limit = max(frame.clip.y, bitcast<f32>(options[2].y));
    var hit = closest(Ray(origin, direction), limit, false);
    // Inside a solid (the first surface is met from behind): continue past each exit until
    // the first front face or the sky, like a camera whose near plane is past the solid.
    // `travel` keeps the distance from the probe centre (the depth moments stay consistent).
    var travel = 0.0;
    if (options[1].y & SPHERICAL_FLAG_INSIDE) != 0u {
        for (var k = 0u; k < SPHERICAL_INSIDE_CROSSINGS; k++) {
            if hit.object == MISS { break; }
            let t = mesh_triangle(hit.object);
            if dot(cross(t.b.position - t.a.position, t.c.position - t.a.position), direction) <= 0.0 { break; }
            travel += hit.t + 2.0 * EPS;
            hit = closest(Ray(origin + direction * travel, direction), max(limit - travel, 0.0), false);
        }
    }
    var hit_position = vec4<f32>(0.0);
    var hit_normal = vec2<u32>(0u);
    var sky_rgb = vec3<f32>(0.0);
    var sun_rgb = vec3<f32>(0.0);
    var other_rgb = vec3<f32>(0.0);
    var albedo = vec3<f32>(0.0);
    if hit.object == MISS {
        sky_rgb = spherical_miss_sky(direction, inv_norm);
    } else {
        let tri = mesh_triangle(hit.object);
        let material = hit_material(tri, hit.bary);
        let p = origin + direction * (travel + hit.t);
        let geom = normalize(cross(tri.b.position - tri.a.position, tri.c.position - tri.a.position));
        let back = dot(geom, direction) > 0.0;
        let ng = select(geom, -geom, back);
        let blended = tri.a.normal * (1.0 - hit.bary.x - hit.bary.y)
            + tri.b.normal * hit.bary.x + tri.c.normal * hit.bary.y;
        var n = ng;
        if dot(blended, blended) > 1e-8 {
            let candidate = normalize(blended);
            n = select(-candidate, candidate, dot(candidate, ng) >= 0.0);
        }
        hit_position = vec4<f32>(p, spherical_hit_word(ng));
        atomicAdd(&spherical_hit_count, 1u);
        for (var axis = 0u; axis < 3u; axis++) {
            let ordered = spherical_ordered_float(p[axis]);
            atomicMin(&spherical_hit_min[axis], ordered);
            atomicMax(&spherical_hit_max[axis], ordered);
        }
        let mirror = f32((material.flags.x & 1u) != 0u);
        hit_normal = vec2<u32>(pack2x16float(n.xy),
            pack2x16float(vec2<f32>(n.z, select(mirror, mirror * 1.5 - 0.25, back))));
        other_rgb = material.emission_metallic.rgb;
        if (material.flags.x & 2u) != 0u {
            other_rgb += material.base_roughness.rgb;
        } else if (material.flags.x & 1u) == 0u {
            var seed = spherical_seed(work.ids.x, work.ids.y, px);
            albedo = diffuse(material);
            sun_rgb = spherical_unit_sun(p, ng, n, -direction, material);
            let shadow_origin = p + ng * (2.0 * EPS);
            let light_count = min(256u, frame.light_info.x);
            let light_samples = options[3].z;
            if light_samples == 0u || light_samples >= light_count {
                for (var i = 0u; i < light_count; i++) {
                    // A change to dense light order must not resample unaffected lights.
                    var light_seed = seed;
                    let contribution = mesh_local_at_hit(i, shadow_origin, p, ng, n, -direction,
                        material, &light_seed, true, false);
                    if any(contribution > vec3<f32>(0.0)) {
                        atomicOr(&spherical_light_masks[i >> 5u], 1u << (i & 31u));
                        other_rgb += contribution;
                    }
                }
            } else {
                // Stratified subset: one light per stride from a random start, weighted
                // by the stride. The dependency mask stays conservative: every light
                // whose range reaches the hit, sampled or not.
                let stride = f32(light_count) / f32(light_samples);
                let start = random(&seed);
                for (var j = 0u; j < light_samples; j++) {
                    let i = min(light_count - 1u, u32((f32(j) + start) * stride));
                    var light_seed = seed;
                    other_rgb += stride * mesh_local_at_hit(i, shadow_origin, p, ng, n, -direction,
                        material, &light_seed, true, false);
                }
                for (var i = 0u; i < light_count; i++) {
                    let light = lights[i];
                    let reach = light.kind >= 2u
                        || distance(light.position_range.xyz, p) < light.position_range.w;
                    if reach && any(light.radiance.rgb > vec3<f32>(0.0)) {
                        atomicOr(&spherical_light_masks[i >> 5u], 1u << (i & 31u));
                    }
                }
            }

            // Cosine samples estimate sky reflected by the hit surface.
            sky_rgb += spherical_sky_bounce(p, ng, n, albedo, limit, inv_norm, &seed);
        }
    }
    let face = 6u * work.ids.x + work.ids.y;
    spherical_faces[64u * face + px] =
        SphericalTexel(hit_position, spherical_pack(sky_rgb, albedo.r), spherical_pack(sun_rgb, albedo.g),
            spherical_pack(other_rgb, albedo.b), hit_normal);
    // With the transfer a missed texel holds only the residual, which may be zero or negative.
    let sky_seen = any(sky_rgb > vec3<f32>(0.0))
        || (spherical_transfer_on() && hit.object == MISS);
    let flags = u32(sky_seen) | (u32(any(sun_rgb > vec3<f32>(0.0))) << 1u);
    if flags != 0u { atomicOr(&spherical_global_sources, flags); }
    workgroupBarrier();
    if px == 0u {
        var metadata: SphericalFaceLighting;
        for (var i = 0u; i < 8u; i++) {
            metadata.masks[i] = atomicLoad(&spherical_light_masks[i]);
        }
        for (var i = 0u; i < 3u; i++) {
            metadata.low[i] = atomicLoad(&spherical_hit_min[i]);
            metadata.high[i] = atomicLoad(&spherical_hit_max[i]);
        }
        metadata.low.w = atomicLoad(&spherical_hit_count);
        metadata.high.w = atomicLoad(&spherical_global_sources);
        spherical_face_lighting[face] = metadata;
    }
}

var<workgroup> spherical_sky_sum: array<vec4<f32>, 64>;
var<workgroup> spherical_sun_sum: array<vec3<f32>, 64>;
var<workgroup> spherical_other_sum: array<vec3<f32>, 64>;
var<workgroup> spherical_hit_sum: array<vec2<f32>, 64>;
var<workgroup> spherical_miss_patches: atomic<u32>;

// Slot row layout after the 21 legacy rows (keep in sync with spherical_gpu.rs and
// RAA_SPHERICAL_* in spherical_gi.wgsl): 24 patches x 4 rows, 96 depth-moment bins
// two per row, then one probe information row and one light row (sphere-only direct). Everything is f32 so consumers
// validate without the float16-in-float32 capability.
const SPHERICAL_PATCH_ROW: u32 = 21u;
const SPHERICAL_MOMENT_ROW: u32 = 117u;
const SPHERICAL_INFO_ROW: u32 = 165u;

// Shared-exponent rgb (9-bit mantissas, 5-bit exponent, bias 15), unsigned, clamped to
// its range; read by raa_spherical_rgb9e5.
fn spherical_rgb9e5(rgb: vec3<f32>) -> u32 {
    let c = clamp(rgb, vec3<f32>(0.0), vec3<f32>(65408.0));
    let m = max(c.x, max(c.y, c.z));
    if !(m >= 1.5258789e-5) { return 0u; }
    var e = max(-16, i32(floor(log2(m)))) + 16;
    var scale = exp2(f32(e - 24));
    if u32(floor(m / scale + 0.5)) >= 512u { e += 1; scale *= 2.0; }
    let q = min(vec3<u32>(floor(c / scale + 0.5)), vec3<u32>(511u));
    return q.x | (q.y << 9u) | (q.z << 18u) | (u32(e) << 27u);
}

fn spherical_backface(texel: SphericalTexel) -> bool {
    let flag = unpack2x16float(texel.hit_normal.y).y;
    return texel.hit_position.w > 0.5 && (flag < 0.0 || flag > 1.1);
}

@compute @workgroup_size(8, 8, 1)
fn spherical_project(@builtin(local_invocation_id) id: vec3<u32>,
                     @builtin(workgroup_id) group: vec3<u32>) {
    let work = spherical_work[options[0].y + group.z / 7u];
    let projection = group.z % 7u;
    let px = id.x + 8u * id.y;
    let uv = (vec2<f32>(id.xy) + 0.5) * 0.25 - 1.0;
    let jacobian = pow(1.0 + dot(uv, uv), -1.5);
    var sky_rgb = vec3<f32>(0.0);
    var sun_rgb = vec3<f32>(0.0);
    var other_rgb = vec3<f32>(0.0);
    var total = 0.0;
    var hits = vec2<f32>(0.0);
    var misses = 0u;
    if px == 0u { atomicStore(&spherical_miss_patches, 0u); }
    workgroupBarrier();
    // A free-space probe has no surface normal: projection zero is the uniform
    // spherical average and also counts back-face hits. The other six workgroups
    // each reduce one face into a world-axis descriptor.
    let first = max(projection, 1u) - 1u;
    let last = select(projection, 6u, projection == 0u);
    // With the transfer a missed texel holds only the residual: add the frame's SH sky.
    let transfer = spherical_transfer_on();
    let norm = spherical_transfer_norm();
    let inv_norm = select(0.0, 1.0 / norm, transfer && norm > 1e-7);
    for (var face = first; face < last; face++) {
        let weight = jacobian;
        let texel = spherical_faces[384u * work.ids.x + 64u * face + px];
        sky_rgb += spherical_unpack(texel.sky) * weight;
        if inv_norm > 0.0 && texel.hit_position.w <= 0.5 {
            let direction = normalize(raa_spherical_face_direction(face, uv));
            sky_rgb += spherical_sh_sky(direction) * (inv_norm * weight);
        }
        sun_rgb += spherical_unpack(texel.sun) * weight;
        other_rgb += spherical_unpack(texel.other) * weight;
        total += weight;
        hits += vec2<f32>(f32(texel.hit_position.w > 0.5), f32(spherical_backface(texel)));
        if texel.hit_position.w <= 0.5 {
            misses |= 1u << (4u * face + (id.x >> 2u) + 2u * (id.y >> 2u));
        }
    }
    if misses != 0u { atomicOr(&spherical_miss_patches, misses); }
    spherical_sky_sum[px] = vec4<f32>(sky_rgb, total);
    spherical_sun_sum[px] = sun_rgb;
    spherical_other_sum[px] = other_rgb;
    spherical_hit_sum[px] = hits;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if px < stride {
            spherical_sky_sum[px] += spherical_sky_sum[px + stride];
            spherical_sun_sum[px] += spherical_sun_sum[px + stride];
            spherical_other_sum[px] += spherical_other_sum[px + stride];
            spherical_hit_sum[px] += spherical_hit_sum[px + stride];
        }
        workgroupBarrier();
    }
    if px == 0u {
        let inv = 1.0 / max(spherical_sky_sum[0].w, 1e-8);
        let base = work.ids.y + 3u * projection;
        spherical_lookup[base] = bitcast<vec4<u32>>(vec4<f32>(spherical_sky_sum[0].rgb * inv, 0.0));
        spherical_lookup[base + 1u] = bitcast<vec4<u32>>(vec4<f32>(spherical_sun_sum[0] * inv, 0.0));
        spherical_lookup[base + 2u] = bitcast<vec4<u32>>(vec4<f32>(spherical_other_sum[0] * inv, 0.0));
        if projection == 0u {
            // A probe whose rays mostly hit back faces sits inside closed geometry.
            let rays = 384.0;
            let back = spherical_hit_sum[0].y / rays;
            let validity = 1.0 - smoothstep(0.3, 0.5, back);
            let info = bitcast<vec4<u32>>(vec4<f32>(validity, back, spherical_hit_sum[0].x / rays, 0.0));
            spherical_lookup[work.ids.y + SPHERICAL_INFO_ROW] =
                vec4<u32>(info.xyz, atomicLoad(&spherical_miss_patches));
        }
    }
}

fn spherical_oct_encode(n: vec3<f32>) -> vec2<f32> {
    let p = n.xy / max(abs(n.x) + abs(n.y) + abs(n.z), 1e-8);
    if n.z >= 0.0 { return p; }
    let s = select(vec2<f32>(-1.0), vec2<f32>(1.0), p >= vec2<f32>(0.0));
    return (1.0 - abs(p.yx)) * s;
}

fn spherical_masked(texel: SphericalTexel, channels: u32) -> vec3<f32> {
    var rgb = vec3<f32>(0.0);
    if (channels & 1u) != 0u { rgb += spherical_unpack(texel.sky); }
    if (channels & 2u) != 0u { rgb += spherical_unpack(texel.sun); }
    if (channels & 4u) != 0u { rgb += spherical_unpack(texel.other); }
    return rgb;
}

// Patch accumulators: a * hit (xyz), a; a * normal (xyz), distance; hit flux per light
// group (sun, sky, other); miss flux (sky group).
var<workgroup> spherical_bin_area: array<vec4<f32>, 64>;
var<workgroup> spherical_bin_normal: array<vec4<f32>, 64>;
var<workgroup> spherical_bin_hit: array<vec3<f32>, 64>;
var<workgroup> spherical_bin_hit_sky: array<vec3<f32>, 64>;
var<workgroup> spherical_bin_hit_other: array<vec3<f32>, 64>;
var<workgroup> spherical_bin_miss: array<vec3<f32>, 64>;
// Per texel: the sky visibility transfer of a missed texel, Y_k(w) dw (zero for a hit).
var<workgroup> spherical_bin_transfer: array<array<f32, 9>, 64>;

// One workgroup per (probe, face). It reduces the face into 2x2 parallax patches
// (hit centroid, area, mean normal, intensity normalised so that the patch reproduces
// the probe's own irradiance at the probe) and 4x4 depth moments for visibility.
@compute @workgroup_size(8, 8, 1)
fn spherical_bins(@builtin(local_invocation_id) id: vec3<u32>,
                  @builtin(workgroup_id) group: vec3<u32>) {
    let work = spherical_work[options[0].y + group.z / 6u];
    let face = group.z % 6u;
    let probe = work.position.xyz;
    let channels = options[1].x;
    // Depth moments see distances up to the DDGI maximum depth (1.5 * sqrt(3) * the cell
    // spacing in work.position.w): a miss stored at the outer shell (~57 m) skewed the bins
    // toward far and let light leak through walls (spherical-normal agent).
    let max_depth = select(1.5 * bitcast<f32>(options[2].y), 2.598 * work.position.w, work.position.w > 0.0);
    let miss_distance = max_depth;
    let px = id.x + 8u * id.y;
    let uv = (vec2<f32>(id.xy) + 0.5) * 0.25 - 1.0;
    let direction = normalize(raa_spherical_face_direction(face, uv));
    let solid_angle = 0.0625 * pow(1.0 + dot(uv, uv), -1.5);
    let texel = spherical_faces[384u * work.ids.x + 64u * face + px];
    let sky_radiance = spherical_masked(texel, channels & 1u) * solid_angle;
    var area = vec4<f32>(0.0);
    var normal = vec4<f32>(0.0, 0.0, 0.0, miss_distance);
    var hit = vec3<f32>(0.0);
    var hit_sky = vec3<f32>(0.0);
    var hit_other = vec3<f32>(0.0);
    var miss = vec3<f32>(0.0);
    if texel.hit_position.w > 0.5 {
        let delta = texel.hit_position.xyz - probe;
        let r = length(delta);
        let n = vec3<f32>(unpack2x16float(texel.hit_normal.x), unpack2x16float(texel.hit_normal.y).x);
        let facing = max(dot(n, -direction), 0.2);
        let a = solid_angle * r * r / facing;
        area = vec4<f32>(a * texel.hit_position.xyz, a);
        normal = vec4<f32>(a * n, min(r, max_depth));
        hit = spherical_masked(texel, channels & 2u) * solid_angle;
        hit_sky = sky_radiance;
        hit_other = spherical_masked(texel, channels & 4u) * solid_angle;
    } else {
        miss = sky_radiance;
    }
    var transfer: array<f32, 9>;
    if texel.hit_position.w <= 0.5 {
        let y = spherical_sh(direction);
        for (var k = 0u; k < 9u; k++) { transfer[k] = y[k] * solid_angle; }
    }
    spherical_bin_transfer[px] = transfer;
    spherical_bin_area[px] = area;
    spherical_bin_normal[px] = normal;
    spherical_bin_hit[px] = hit;
    spherical_bin_hit_sky[px] = hit_sky;
    spherical_bin_hit_other[px] = hit_other;
    spherical_bin_miss[px] = miss;
    workgroupBarrier();
    let slot_base = work.ids.y;
    if px < 4u {
        let corner = vec2<u32>(px & 1u, px >> 1u) * 4u;
        var a = vec4<f32>(0.0);
        var n = vec3<f32>(0.0);
        var flux_hit = vec3<f32>(0.0);
        var flux_sky = vec3<f32>(0.0);
        var flux_other = vec3<f32>(0.0);
        var flux_miss = vec3<f32>(0.0);
        var t_k: array<f32, 9>;
        for (var y = 0u; y < 4u; y++) {
            for (var x = 0u; x < 4u; x++) {
                let t = corner.x + x + 8u * (corner.y + y);
                a += spherical_bin_area[t];
                n += spherical_bin_normal[t].xyz;
                flux_hit += spherical_bin_hit[t];
                flux_sky += spherical_bin_hit_sky[t];
                flux_other += spherical_bin_hit_other[t];
                flux_miss += spherical_bin_miss[t];
                for (var k = 0u; k < 9u; k++) { t_k[k] += spherical_bin_transfer[t][k]; }
            }
        }
        if spherical_transfer_on() {
            // flux_miss is the residual here: keep it with the transfer, publish the composite.
            let patch_index = 24u * work.ids.x + 4u * face + px;
            let row = SPHERICAL_TRANSFER_HEADER + 2u * patch_index;
            spherical_transfer[row] = vec4<u32>(pack2x16float(vec2<f32>(t_k[0], t_k[1])),
                pack2x16float(vec2<f32>(t_k[2], t_k[3])), pack2x16float(vec2<f32>(t_k[4], t_k[5])),
                pack2x16float(vec2<f32>(t_k[6], t_k[7])));
            let r = clamp(flux_miss, vec3<f32>(-65504.0), vec3<f32>(65504.0));
            spherical_transfer[row + 1u] = vec4<u32>(pack2x16float(vec2<f32>(t_k[8], r.x)),
                pack2x16float(r.yz), 0u, 0u);
            flux_miss = spherical_patch_sky(patch_index);
        }
        var row0 = vec4<f32>(probe, -1.0);
        var intensity = vec3<f32>(0.0);
        var intensity_sky = vec3<f32>(0.0);
        var intensity_other = vec3<f32>(0.0);
        var coherence = 0.0;
        var mean_normal = vec3<f32>(0.0, 0.0, 1.0);
        if a.w > 0.0 {
            let h = a.xyz / a.w;
            let r2 = a.w / PI;
            let length_n = length(n);
            coherence = clamp(length_n / a.w, 0.0, 1.0);
            if length_n > 1e-8 { mean_normal = n / length_n; }
            let to_probe = probe - h;
            let d2 = dot(to_probe, to_probe);
            let facing = max(dot(mean_normal, to_probe * inverseSqrt(max(d2, 1e-12))), 0.2);
            let g = mix(1.0, facing, coherence) / (d2 + r2);
            intensity = flux_hit / max(g, 1e-8);
            intensity_sky = flux_sky / max(g, 1e-8);
            intensity_other = flux_other / max(g, 1e-8);
            row0 = vec4<f32>(h, r2);
        }
        let patch_row = slot_base + SPHERICAL_PATCH_ROW + 4u * (4u * face + px);
        spherical_lookup[patch_row] = bitcast<vec4<u32>>(row0);
        spherical_lookup[patch_row + 1u] = bitcast<vec4<u32>>(vec4<f32>(intensity, coherence));
        spherical_lookup[patch_row + 2u] = vec4<u32>(bitcast<vec3<u32>>(flux_miss),
            pack2x16snorm(spherical_oct_encode(mean_normal)));
        spherical_lookup[patch_row + 3u] = vec4<u32>(spherical_rgb9e5(intensity_sky),
            spherical_rgb9e5(intensity_other), 0u, 0u);
    } else if px < 12u {
        // Two 2x2 depth-moment bins per row.
        let pair = px - 4u;
        var moments = vec4<f32>(0.0);
        for (var k = 0u; k < 2u; k++) {
            let bin = 2u * pair + k;
            let corner = vec2<u32>(bin & 3u, bin >> 2u) * 2u;
            var m = vec2<f32>(0.0);
            for (var y = 0u; y < 2u; y++) {
                for (var x = 0u; x < 2u; x++) {
                    let r = spherical_bin_normal[corner.x + x + 8u * (corner.y + y)].w;
                    m += vec2<f32>(r, r * r);
                }
            }
            m *= 0.25;
            if k == 0u { moments = vec4<f32>(m, moments.zw); } else { moments = vec4<f32>(moments.xy, m); }
        }
        spherical_lookup[slot_base + SPHERICAL_MOMENT_ROW + 8u * face + pair] = bitcast<vec4<u32>>(moments);
    }
}

// Sphere-only direct lighting (phase 2). One 64-thread workgroup per projected probe
// writes the light row: x sun visibility (16 jittered shadow rays from the probe),
// y two local light indices (low, high 16 bits; 0xffff none), z and w their visibility
// (16 rays toward a 5 cm jittered emitter). Lights are ranked by their unshadowed
// irradiance at the probe; a spot outside its cone keeps a tenth of it so receivers
// inside the cone near the probe are still served.
const SPHERICAL_LIGHT_ROW: u32 = 166u;
const SPHERICAL_NO_LIGHT: u32 = 0xffffu;
var<workgroup> spherical_light_best: array<vec2<f32>, 128>;
var<workgroup> spherical_light_pick: vec2<u32>;
var<workgroup> spherical_light_hits: array<atomic<u32>, 3>;

fn spherical_light_importance(i: u32, p: vec3<f32>) -> f32 {
    let light = lights[i];
    let lum = dot(light.radiance.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    if lum <= 0.0 { return 0.0; }
    if light.kind == 2u { return lum; }
    let delta = light.position_range.xyz - p;
    let d = length(delta);
    if d >= light.position_range.w { return 0.0; }
    let window = saturate(1.0 - pow(d / light.position_range.w, 4.0));
    var w = window * window / max(d * d, 0.04);
    if light.kind == 1u {
        let cone = smoothstep(light.direction_outer.w, light.inner,
            dot(light.direction_outer.xyz, -delta / max(d, EPS)));
        w *= max(cone, 0.1);
    }
    return lum * w;
}

// Offsets on a 4x4 grid in [-1, 1]^2 across the plane normal to `axis`.
fn spherical_light_jitter(lane: u32, axis: vec3<f32>) -> vec3<f32> {
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(axis.y) > 0.9);
    let t = normalize(cross(up, axis));
    let b = cross(axis, t);
    let uv = (vec2<f32>(f32(lane & 3u), f32(lane >> 2u)) + 0.5) * 0.5 - 1.0;
    return t * uv.x + b * uv.y;
}

@compute @workgroup_size(64, 1, 1)
fn spherical_lights(@builtin(local_invocation_index) t: u32,
                    @builtin(workgroup_id) group: vec3<u32>) {
    let work = spherical_work[options[0].y + group.z];
    let p = work.position.xyz;
    if t < 3u { atomicStore(&spherical_light_hits[t], 0u); }
    var b0 = vec2<f32>(0.0, -1.0);
    var b1 = vec2<f32>(0.0, -1.0);
    for (var i = t; i < min(256u, frame.light_info.x); i += 64u) {
        let w = spherical_light_importance(i, p);
        if w > b0.x {
            b1 = b0;
            b0 = vec2<f32>(w, f32(i));
        } else if w > b1.x {
            b1 = vec2<f32>(w, f32(i));
        }
    }
    spherical_light_best[2u * t] = b0;
    spherical_light_best[2u * t + 1u] = b1;
    workgroupBarrier();
    if t == 0u {
        var c0 = vec2<f32>(0.0, -1.0);
        var c1 = vec2<f32>(0.0, -1.0);
        for (var k = 0u; k < 128u; k++) {
            let c = spherical_light_best[k];
            if c.x > c0.x {
                c1 = c0;
                c0 = c;
            } else if c.x > c1.x {
                c1 = c;
            }
        }
        spherical_light_pick = vec2<u32>(select(SPHERICAL_NO_LIGHT, u32(c0.y), c0.y >= 0.0),
            select(SPHERICAL_NO_LIGHT, u32(c1.y), c1.y >= 0.0));
    }
    let pick = workgroupUniformLoad(&spherical_light_pick);
    let lane = t & 15u;
    let which = t >> 4u;
    if which == 0u {
        let sun = -frame.sun_dir.xyz;
        let dir = normalize(sun + 0.02 * spherical_light_jitter(lane, sun));
        if any(frame.sun_color.rgb > vec3<f32>(0.0))
            && closest(Ray(p, dir), frame.clip.y, true).object == MISS {
            atomicAdd(&spherical_light_hits[0], 1u);
        }
    } else if which <= 2u {
        let index = select(pick.x, pick.y, which == 2u);
        if index != SPHERICAL_NO_LIGHT {
            let light = lights[index];
            var dir = -light.direction_outer.xyz;
            var distance = frame.clip.y;
            if light.kind != 2u {
                let axis = normalize(light.position_range.xyz - p);
                let delta = light.position_range.xyz + 0.05 * spherical_light_jitter(lane, axis) - p;
                distance = length(delta);
                dir = delta / max(distance, EPS);
            }
            if closest(Ray(p, dir), distance - 2.0 * EPS, true).object == MISS {
                atomicAdd(&spherical_light_hits[which], 1u);
            }
        }
    }
    workgroupBarrier();
    if t == 0u {
        let hits = vec3<f32>(f32(atomicLoad(&spherical_light_hits[0])),
            f32(atomicLoad(&spherical_light_hits[1])), f32(atomicLoad(&spherical_light_hits[2]))) / 16.0;
        spherical_lookup[work.ids.y + SPHERICAL_LIGHT_ROW] = vec4<u32>(bitcast<u32>(hits.x),
            pick.x | (pick.y << 16u), bitcast<u32>(hits.y), bitcast<u32>(hits.z));
    }
}

// ---- Sky transfer, sun re-shade and light reach (spherical-sky) ----

// Composed sky miss flux of one (slot, patch) in unit-sky units under the frame's sky SH:
// sum_k T_k L_k / norm + R. The lookup scales it by sky diffuse x norm (row 69).
fn spherical_patch_sky(patch_index: u32) -> vec3<f32> {
    let row = SPHERICAL_TRANSFER_HEADER + 2u * patch_index;
    let a = spherical_transfer[row];
    let b = spherical_transfer[row + 1u];
    let t01 = unpack2x16float(a.x);
    let t23 = unpack2x16float(a.y);
    let t45 = unpack2x16float(a.z);
    let t67 = unpack2x16float(a.w);
    let t8r = unpack2x16float(b.x);
    let gb = unpack2x16float(b.y);
    let t = array<f32, 9>(t01.x, t01.y, t23.x, t23.y, t45.x, t45.y, t67.x, t67.y, t8r.x);
    var sh = vec3<f32>(0.0);
    for (var k = 0u; k < 9u; k++) { sh += t[k] * spherical_sky_coefficient(k); }
    let norm = spherical_transfer_norm();
    let composed = select(vec3<f32>(0.0), sh / norm, norm > 1e-7) + vec3<f32>(t8r.y, gb);
    return max(composed, vec3<f32>(0.0));
}

// Sky SH of the frame (the environment cube at mip 0, the sun cone left out) projected from
// 4096 fixed spherical-Fibonacci directions, and the sky norm, into the transfer header.
// One workgroup; no CPU readback.
var<workgroup> spherical_sky_partial: array<array<vec3<f32>, 9>, 64>;
@compute @workgroup_size(64, 1, 1)
fn spherical_sky_project(@builtin(local_invocation_index) px: u32) {
    let norm = spherical_sky_norm(px);
    var acc: array<vec3<f32>, 9>;
    if options[0].z != 0u {
        for (var i = px; i < SPHERICAL_SKY_PROJECT_SAMPLES; i += 64u) {
            let z = 1.0 - (2.0 * f32(i) + 1.0) / f32(SPHERICAL_SKY_PROJECT_SAMPLES);
            let r = sqrt(max(0.0, 1.0 - z * z));
            let phi = f32(i) * 2.39996323;
            let d = vec3<f32>(r * cos(phi), r * sin(phi), z);
            if spherical_in_sun_cone(d) { continue; }
            let rgb = textureSampleLevel(sky, raa_linear_clamp, d, 0.0).rgb;
            let y = spherical_sh(d);
            for (var k = 0u; k < 9u; k++) { acc[k] += rgb * y[k]; }
        }
    }
    spherical_sky_partial[px] = acc;
    workgroupBarrier();
    if px < 9u {
        var sum = vec3<f32>(0.0);
        for (var t = 0u; t < 64u; t++) { sum += spherical_sky_partial[t][px]; }
        let coefficient = sum * (4.0 * PI / f32(SPHERICAL_SKY_PROJECT_SAMPLES));
        spherical_transfer[px] = vec4<u32>(bitcast<vec3<u32>>(coefficient), 0u);
    }
    if px == 9u {
        spherical_transfer[SPHERICAL_TRANSFER_NORM_ROW] = vec4<u32>(bitcast<u32>(norm), 0u, 0u, 0u);
    }
}

// Relight: one thread per (slot, patch) rebuilds the patch's sky miss flux (lookup row 2 xyz)
// from its transfer under the frame's sky SH. No ray. options[0].w = lookup row of slot 0.
@compute @workgroup_size(64, 1, 1)
fn spherical_relight(@builtin(global_invocation_id) id: vec3<u32>) {
    let patches = (arrayLength(&spherical_transfer) - SPHERICAL_TRANSFER_HEADER) / 2u;
    let patch_index = id.x;
    if patch_index >= patches { return; }
    let slot = patch_index / 24u;
    let row = options[0].w + SPHERICAL_SLOT_ROWS * slot + SPHERICAL_PATCH_ROW + 4u * (patch_index % 24u) + 2u;
    let composed = spherical_patch_sky(patch_index);
    spherical_lookup[row] = vec4<u32>(bitcast<vec3<u32>>(composed), spherical_lookup[row].w);
}

// Re-shade: one 8x8 workgroup per (slot, face) job (work entries from options[1].z; ids.z =
// blend weight alpha as f32, ids.w = groups: 1 sun, 2 sky). From the stored hit position,
// both normals and the albedo, the sun group is one shadow ray per hit texel and the sky group
// the capture's own bounce rays (same seeds); a missed texel's sky (or residual) is resampled
// from the cube. No primary ray. The new value is blended into the stored one by alpha.
@compute @workgroup_size(8, 8, 1)
fn spherical_reshade(@builtin(local_invocation_id) id: vec3<u32>,
                     @builtin(workgroup_id) group: vec3<u32>) {
    let work = spherical_work[options[1].z + group.z];
    let px = id.x + 8u * id.y;
    let sky_norm = spherical_sky_norm(px);
    let inv_norm = select(0.0, 1.0 / sky_norm, sky_norm > 0.0);
    let alpha = bitcast<f32>(work.ids.z);
    let groups = work.ids.w;
    let index = 64u * (6u * work.ids.x + work.ids.y) + px;
    var texel = spherical_faces[index];
    if texel.hit_position.w <= 0.5 {
        if (groups & 2u) != 0u {
            let uv = (vec2<f32>(id.xy) + 0.5) * 0.25 - 1.0;
            let direction = normalize(raa_spherical_face_direction(work.ids.y, uv));
            let fresh = spherical_miss_sky(direction, inv_norm);
            texel.sky = spherical_pack(mix(spherical_unpack(texel.sky), fresh, alpha), spherical_extra(texel.sky));
        }
    } else {
        let albedo = spherical_albedo(texel);
        if any(albedo > vec3<f32>(0.0)) {
            let p = texel.hit_position.xyz;
            let ng = spherical_hit_normal(texel.hit_position.w);
            let n = normalize(vec3<f32>(unpack2x16float(texel.hit_normal.x), unpack2x16float(texel.hit_normal.y).x));
            if (groups & 1u) != 0u {
                let fresh = spherical_unit_sun_albedo(p, ng, n, albedo);
                texel.sun = spherical_pack(mix(spherical_unpack(texel.sun), fresh, alpha), albedo.g);
            }
            if (groups & 2u) != 0u {
                var seed = spherical_seed(work.ids.x, work.ids.y, px);
                let light_count = min(256u, frame.light_info.x);
                let light_samples = options[3].z;
                if light_samples != 0u && light_samples < light_count { _ = random(&seed); }
                let limit = max(frame.clip.y, bitcast<f32>(options[2].y));
                let fresh = spherical_sky_bounce(p, ng, n, albedo, limit, inv_norm, &seed);
                texel.sky = spherical_pack(mix(spherical_unpack(texel.sky), fresh, alpha), albedo.r);
            }
        }
    }
    spherical_faces[index] = texel;
}

// Light reach: one 64-thread workgroup per pool face tests every stored hit texel against up to
// four changed lights (dense indices in transfer row 10, count in row 11 x): the light's
// unshadowed diffuse contribution must pass the threshold (options[2].w luma, the DDGI
// `bound`: range, cone, falloff) and one shadow ray must reach it. Bit j of the face word =
// light j reaches a surface this face sees (read back: only those faces are recaptured).
var<workgroup> spherical_reach_bits: atomic<u32>;
@compute @workgroup_size(64, 1, 1)
fn spherical_reach_test(@builtin(local_invocation_index) px: u32,
                        @builtin(workgroup_id) group: vec3<u32>) {
    let face = group.x;
    if px == 0u { atomicStore(&spherical_reach_bits, 0u); }
    workgroupBarrier();
    let texel = spherical_faces[64u * face + px];
    let albedo = spherical_albedo(texel);
    let count = min(spherical_transfer[SPHERICAL_REACH_ROW + 1u].x, 4u);
    let threshold = bitcast<f32>(options[2].w);
    if texel.hit_position.w > 0.5 && any(albedo > vec3<f32>(0.0)) {
        let p = texel.hit_position.xyz;
        let ng = spherical_hit_normal(texel.hit_position.w);
        let n = normalize(vec3<f32>(unpack2x16float(texel.hit_normal.x), unpack2x16float(texel.hit_normal.y).x));
        let indices = spherical_transfer[SPHERICAL_REACH_ROW];
        for (var j = 0u; j < count; j++) {
            let light = lights[min(indices[j], 255u)];
            if !any(light.radiance.rgb > vec3<f32>(0.0)) { continue; }
            let delta = light.position_range.xyz - p;
            var distance = length(delta);
            var l = delta / max(distance, EPS);
            var weight = 1.0;
            var test = true;
            if light.kind == 2u {
                l = -light.direction_outer.xyz;
                distance = frame.clip.y;
            } else if light.kind == 3u {
                // Area panel: reachable within range plus its half diagonal, in front of it.
                let reach = light.position_range.w + 0.5 * length(vec2<f32>(light.width, light.height));
                if distance >= reach || dot(light.direction_outer.xyz, -delta) < -reach { continue; }
                test = false;
            } else {
                if distance >= light.position_range.w { continue; }
                let window = saturate(1.0 - pow(distance / light.position_range.w, 4.0));
                weight = window * window / max(distance * distance, 0.04);
                if light.kind == 1u {
                    weight *= smoothstep(light.direction_outer.w, light.inner, dot(light.direction_outer.xyz, -l));
                }
            }
            if test {
                let nl = saturate(dot(n, l));
                if nl <= 0.0 || dot(ng, l) <= 0.0 || weight <= 0.0 { continue; }
                let c = albedo * (nl * weight / PI) * light.radiance.rgb;
                if dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)) <= threshold { continue; }
                if closest(Ray(p + ng * (2.0 * EPS), l), distance - 2.0 * EPS, true).object != MISS { continue; }
            }
            atomicOr(&spherical_reach_bits, 1u << j);
        }
    }
    workgroupBarrier();
    if px == 0u { spherical_reach[face] = atomicLoad(&spherical_reach_bits); }
}
