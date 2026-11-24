// One 8×8 workgroup traces one scheduled face through the shared mesh BVH.
// The packed 48-byte texel also keeps world-space hit position, validity and normal.
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
        let tri = triangles[hit.object];
        let material = hit_material(tri, hit.bary);
        let p = origin + direction * hit.t;
        let geom = normalize(cross(tri.b.position - tri.a.position, tri.c.position - tri.a.position));
        let ng = select(geom, -geom, dot(geom, direction) > 0.0);
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
        hit_normal = vec2<u32>(pack2x16float(n.xy), pack2x16float(vec2<f32>(n.z, 0.0)));
        other_rgb = material.emission_metallic.rgb;
        if (material.flags.x & 2u) != 0u {
            other_rgb += material.base_roughness.rgb;
        } else if (material.flags.x & 1u) == 0u {
            var seed = (work.ids.x * 747796405u + work.ids.y * 277803737u + px) ^ 0x85ebca6bu;
            sun_rgb = mesh_sun_at_hit(p, ng, n, -direction, material, true);
            let shadow_origin = p + ng * (2.0 * EPS);
            for (var i = 0u; i < min(256u, frame.light_info.x); i++) {
                // A change to dense light order must not resample unaffected lights.
                var light_seed = seed;
                let contribution = mesh_local_at_hit(i, shadow_origin, p, ng, n, -direction,
                    material, &light_seed, true, false);
                if any(contribution > vec3<f32>(0.0)) {
                    atomicOr(&spherical_light_masks[i >> 5u], 1u << (i & 31u));
                    other_rgb += contribution;
                }
            }

            // One cosine sample estimates sky reflected by the hit surface.
            let u = random(&seed);
            let phi = 2.0 * PI * random(&seed);
            let bounce = basis(n) * vec3<f32>(sqrt(u) * cos(phi), sqrt(u) * sin(phi), sqrt(1.0 - u));
            if dot(ng, bounce) > 0.0 && closest(Ray(p + ng * (2.0 * EPS), bounce), limit, true).object == MISS {
                sky_rgb += diffuse(material) * spherical_sky(bounce);
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

@compute @workgroup_size(8, 8, 1)
fn spherical_project(@builtin(local_invocation_id) id: vec3<u32>,
                     @builtin(workgroup_id) group: vec3<u32>) {
    let work = spherical_work[options[0].y + group.z];
    let px = id.x + 8u * id.y;
    let uv = (vec2<f32>(id.xy) + 0.5) * 0.25 - 1.0;
    let jacobian = pow(1.0 + dot(uv, uv), -1.5);
    var sky_rgb = vec3<f32>(0.0);
    var sun_rgb = vec3<f32>(0.0);
    var other_rgb = vec3<f32>(0.0);
    var total = 0.0;
    for (var face = 0u; face < 6u; face++) {
        let direction = normalize(raa_spherical_face_direction(face, uv));
        let weight = jacobian * max(0.0, dot(direction, work.normal.xyz));
        let texel = spherical_faces[384u * work.ids.x + 64u * face + px];
        sky_rgb += spherical_unpack(texel.sky) * weight;
        sun_rgb += spherical_unpack(texel.sun) * weight;
        other_rgb += spherical_unpack(texel.other) * weight;
        total += weight;
    }
    spherical_sky_sum[px] = vec4<f32>(sky_rgb, total);
    spherical_sun_sum[px] = sun_rgb;
    spherical_other_sum[px] = other_rgb;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if px < stride {
            spherical_sky_sum[px] += spherical_sky_sum[px + stride];
            spherical_sun_sum[px] += spherical_sun_sum[px + stride];
            spherical_other_sum[px] += spherical_other_sum[px + stride];
        }
        workgroupBarrier();
    }
    if px == 0u {
        let inv = 1.0 / max(spherical_sky_sum[0].w, 1e-8);
        let base = work.ids.y;
        spherical_lookup[base] = bitcast<vec4<u32>>(vec4<f32>(spherical_sky_sum[0].rgb * inv, 0.0));
        spherical_lookup[base + 1u] = bitcast<vec4<u32>>(vec4<f32>(spherical_sun_sum[0] * inv, 0.0));
        spherical_lookup[base + 2u] = bitcast<vec4<u32>>(vec4<f32>(spherical_other_sum[0] * inv, 0.0));
    }
}
