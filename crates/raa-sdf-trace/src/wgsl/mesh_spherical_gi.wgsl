// One 8×8 workgroup traces one scheduled face through the shared mesh BVH from a
// free-space probe. The packed 48-byte texel also keeps world-space hit position,
// validity and normal. The normal's fourth half is the mirror flag (0 or 1) on a
// front face; a back face stores -0.25 (or 1.25 for a mirror), so `> 0.5` still
// means mirror and `< 0 || > 1.1` means back face.
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

fn spherical_pack(rgb: vec3<f32>) -> vec2<u32> {
    let safe = clamp(rgb, vec3<f32>(0.0), vec3<f32>(65504.0));
    return vec2<u32>(pack2x16float(safe.xy), pack2x16float(vec2<f32>(safe.z, 0.0)));
}

fn spherical_unpack(words: vec2<u32>) -> vec3<f32> {
    return vec3<f32>(unpack2x16float(words.x), unpack2x16float(words.y).x);
}

fn spherical_sky(direction: vec3<f32>) -> vec3<f32> {
    if options[0].z == 0u { return frame.ambient.rgb; }
    return bitcast<f32>(options[2].x) * textureSampleLevel(sky, raa_linear_clamp, direction, 0.0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn spherical_capture(@builtin(global_invocation_id) id: vec3<u32>) {
    let work = spherical_work[id.z];
    let px = id.x + 8u * id.y;
    if px < 3u {
        atomicStore(&spherical_hit_min[px], 0xffffffffu);
    }
    workgroupBarrier();
    let uv = (vec2<f32>(id.xy) + 0.5) * 0.25 - 1.0;
    let direction = normalize(raa_spherical_face_direction(work.ids.y, uv));
    let origin = work.position.xyz + work.normal.xyz * (2.0 * EPS);
    let limit = max(frame.clip.y, bitcast<f32>(options[2].y));
    let hit = closest(Ray(origin, direction), limit, false);
    var hit_position = vec4<f32>(0.0);
    var hit_normal = vec2<u32>(0u);
    var sky_rgb = vec3<f32>(0.0);
    var sun_rgb = vec3<f32>(0.0);
    var other_rgb = vec3<f32>(0.0);
    if hit.object == MISS {
        sky_rgb = spherical_sky(direction);
    } else {
        let tri = mesh_triangle(hit.object);
        let material = hit_material(tri, hit.bary);
        let p = origin + direction * hit.t;
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
        hit_position = vec4<f32>(p, 1.0);
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
            var seed = (work.ids.x * 747796405u + work.ids.y * 277803737u + px) ^ 0x85ebca6bu;
            sun_rgb = mesh_sun_at_hit(p, ng, n, -direction, material, true);
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
            let sky_samples = options[3].y;
            var sky_sum = vec3<f32>(0.0);
            for (var k = 0u; k < sky_samples; k++) {
                let u = random(&seed);
                let phi = 2.0 * PI * random(&seed);
                let bounce = basis(n) * vec3<f32>(sqrt(u) * cos(phi), sqrt(u) * sin(phi), sqrt(1.0 - u));
                if dot(ng, bounce) > 0.0 && closest(Ray(p + ng * (2.0 * EPS), bounce), limit, true).object == MISS {
                    sky_sum += spherical_sky(bounce);
                }
            }
            if sky_samples > 0u {
                sky_rgb += diffuse(material) * sky_sum / f32(sky_samples);
            }
        }
    }
    let face = 6u * work.ids.x + work.ids.y;
    spherical_faces[64u * face + px] =
        SphericalTexel(hit_position, spherical_pack(sky_rgb), spherical_pack(sun_rgb),
            spherical_pack(other_rgb), hit_normal);
    let flags = u32(any(sky_rgb > vec3<f32>(0.0))) | (u32(any(sun_rgb > vec3<f32>(0.0))) << 1u);
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
// RAA_SPHERICAL_* in spherical_gi.wgsl): 24 patches x 3 rows, 96 depth-moment bins
// two per row, then one probe information row and one light row (sphere-only direct). Everything is f32 so consumers
// validate without the float16-in-float32 capability.
const SPHERICAL_PATCH_ROW: u32 = 21u;
const SPHERICAL_MOMENT_ROW: u32 = 93u;
const SPHERICAL_INFO_ROW: u32 = 141u;

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
    for (var face = first; face < last; face++) {
        let weight = jacobian;
        let texel = spherical_faces[384u * work.ids.x + 64u * face + px];
        sky_rgb += spherical_unpack(texel.sky) * weight;
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

// Patch accumulators: a * hit (xyz), a; a * normal (xyz), distance; hit flux; miss flux.
var<workgroup> spherical_bin_area: array<vec4<f32>, 64>;
var<workgroup> spherical_bin_normal: array<vec4<f32>, 64>;
var<workgroup> spherical_bin_hit: array<vec3<f32>, 64>;
var<workgroup> spherical_bin_miss: array<vec3<f32>, 64>;

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
    let miss_distance = 1.5 * bitcast<f32>(options[2].y);
    let px = id.x + 8u * id.y;
    let uv = (vec2<f32>(id.xy) + 0.5) * 0.25 - 1.0;
    let direction = normalize(raa_spherical_face_direction(face, uv));
    let solid_angle = 0.0625 * pow(1.0 + dot(uv, uv), -1.5);
    let texel = spherical_faces[384u * work.ids.x + 64u * face + px];
    let radiance = spherical_masked(texel, channels) * solid_angle;
    var area = vec4<f32>(0.0);
    var normal = vec4<f32>(0.0, 0.0, 0.0, miss_distance);
    var hit = vec3<f32>(0.0);
    var miss = vec3<f32>(0.0);
    if texel.hit_position.w > 0.5 {
        let delta = texel.hit_position.xyz - probe;
        let r = length(delta);
        let n = vec3<f32>(unpack2x16float(texel.hit_normal.x), unpack2x16float(texel.hit_normal.y).x);
        let facing = max(dot(n, -direction), 0.2);
        let a = solid_angle * r * r / facing;
        area = vec4<f32>(a * texel.hit_position.xyz, a);
        normal = vec4<f32>(a * n, r);
        hit = radiance;
    } else {
        miss = radiance;
    }
    spherical_bin_area[px] = area;
    spherical_bin_normal[px] = normal;
    spherical_bin_hit[px] = hit;
    spherical_bin_miss[px] = miss;
    workgroupBarrier();
    let slot_base = work.ids.y;
    if px < 4u {
        let corner = vec2<u32>(px & 1u, px >> 1u) * 4u;
        var a = vec4<f32>(0.0);
        var n = vec3<f32>(0.0);
        var flux_hit = vec3<f32>(0.0);
        var flux_miss = vec3<f32>(0.0);
        for (var y = 0u; y < 4u; y++) {
            for (var x = 0u; x < 4u; x++) {
                let t = corner.x + x + 8u * (corner.y + y);
                a += spherical_bin_area[t];
                n += spherical_bin_normal[t].xyz;
                flux_hit += spherical_bin_hit[t];
                flux_miss += spherical_bin_miss[t];
            }
        }
        var row0 = vec4<f32>(probe, -1.0);
        var intensity = vec3<f32>(0.0);
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
            row0 = vec4<f32>(h, r2);
        }
        let patch_row = slot_base + SPHERICAL_PATCH_ROW + 3u * (4u * face + px);
        spherical_lookup[patch_row] = bitcast<vec4<u32>>(row0);
        spherical_lookup[patch_row + 1u] = bitcast<vec4<u32>>(vec4<f32>(intensity, coherence));
        spherical_lookup[patch_row + 2u] = vec4<u32>(bitcast<vec3<u32>>(flux_miss),
            pack2x16snorm(spherical_oct_encode(mean_normal)));
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
const SPHERICAL_LIGHT_ROW: u32 = 142u;
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
