// Mesh intersections share the conservative BVH node layout and slab test with SDF tracing.
struct TraceNode { lo: vec3<f32>, left: u32, hi: vec3<f32>, count: u32 }
struct Vertex { position: vec3<f32>, material: u32, normal: vec3<f32>, padding: u32 }
struct Triangle { a: Vertex, b: Vertex, c: Vertex }
struct Material { base_roughness: vec4<f32>, emission_metallic: vec4<f32>, flags: vec4<u32> }
struct Ray { origin: vec3<f32>, direction: vec3<f32> }
struct Hit { t: f32, object: u32, bary: vec2<f32> }
struct Frame {
    view_proj: mat4x4<f32>, view: mat4x4<f32>, proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>, inv_view: mat4x4<f32>,
    camera_pos: vec4<f32>, sun_dir: vec4<f32>, sun_color: vec4<f32>,
    ambient: vec4<f32>, viewport: vec4<f32>, clip: vec4<f32>, flags: vec4<u32>,
    exposure: vec4<f32>, reserved: array<vec4<f32>, 4>,
    previous_view_proj: mat4x4<f32>, unjittered_view_proj: mat4x4<f32>,
    jitter: vec4<f32>, light_info: vec4<u32>, temporal: vec4<u32>,
}
// These four lanes match RAA's packed world-space local-light records.
struct Light {
    position_range: vec4<f32>, radiance: vec4<f32>, direction_outer: vec4<f32>,
    inner: f32, kind: u32, width: f32, height: f32,
}
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> options: array<vec4<u32>, 3>;
@group(0) @binding(2) var<storage, read> nodes: array<TraceNode>;
@group(0) @binding(3) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(4) var<storage, read> materials: array<Material>;
@group(0) @binding(5) var<storage, read_write> counters: array<atomic<u32>, 12>;
@group(0) @binding(6) var depth: texture_depth_2d;
@group(0) @binding(7) var normals: texture_2d<f32>;
@group(0) @binding(8) var motion: texture_2d<f32>;
@group(0) @binding(9) var confidence: texture_2d<f32>;
@group(0) @binding(10) var previous_depth: texture_2d<f32>;
@group(0) @binding(11) var screen_radiance: texture_2d<f32>;
@group(0) @binding(12) var radiance: texture_storage_2d<rgba16float, write>;
@group(0) @binding(13) var<uniform> lights: array<Light, 256>;
@group(0) @binding(14) var sky: texture_cube<f32>;
@group(0) @binding(15) var raa_linear_clamp: sampler;

// Same 96-byte record and atlas used by the lit GI passes.
struct RaaGiVolumeParams {
    levels: array<vec4<f32>, 5>,
    resolution: f32,
    intensity: f32,
    level_count: u32,
    flags: u32,
}
@group(0) @binding(16) var raa_gi_volume: texture_3d<f32>;
@group(0) @binding(17) var<uniform> raa_gi_volume_params: RaaGiVolumeParams;
const GI_VOLUME_VALID: u32 = 1u;

const PI: f32 = 3.14159265;
const EPS: f32 = 0.001;
const MISS: u32 = 0xffffffffu;
const ELIGIBLE: u32 = 0u;
const ACCEPTED: u32 = 1u;
const TRACED: u32 = 2u;
const SCREEN_MISS: u32 = 3u;
const OFFSCREEN: u32 = 4u;
const DISOCCLUSION: u32 = 5u;
const LOW_CONFIDENCE: u32 = 6u;
const NESTED_MIRROR: u32 = 7u;
const GEOMETRY_HIT: u32 = 8u;
const MIRROR_RAYS: u32 = 9u;
const PATH_RAYS: u32 = 10u;
const REFERENCE_PIXELS: u32 = 11u;

fn closest(ray: Ray, limit: f32, any_hit: bool) -> Hit {
    var result = Hit(limit, MISS, vec2<f32>(0.0));
    if options[0].x == 0u { return result; }
    var stack: array<u32, 32>;
    stack[0] = 0u;
    var count = 1u;
    loop {
        if count == 0u { break; }
        count--;
        let node = nodes[stack[count]];
        let interval = trace_interval(node.lo, node.hi, ray.origin, ray.direction, result.t);
        if interval.x > interval.y { continue; }
        if node.count == 0u {
            stack[count] = node.left;
            stack[count + 1u] = node.left + 1u;
            count += 2u;
            continue;
        }
        for (var i = node.left; i < node.left + node.count; i++) {
            let tri = triangles[i];
            let edge1 = tri.b.position - tri.a.position;
            let edge2 = tri.c.position - tri.a.position;
            let p = cross(ray.direction, edge2);
            let det = dot(edge1, p);
            if abs(det) < 1e-8 { continue; }
            let offset = ray.origin - tri.a.position;
            let u = dot(offset, p) / det;
            if u < 0.0 || u > 1.0 { continue; }
            let q = cross(offset, edge1);
            let v = dot(ray.direction, q) / det;
            if v < 0.0 || u + v > 1.0 { continue; }
            let t = dot(edge2, q) / det;
            if t >= EPS && t < result.t {
                result = Hit(t, i, vec2<f32>(u, v));
                if any_hit { return result; }
            }
        }
    }
    return result;
}

fn random(seed: ptr<function, u32>) -> f32 {
    *seed = *seed * 747796405u + 2891336453u;
    let word = ((*seed >> ((*seed >> 28u) + 4u)) ^ *seed) * 277803737u;
    return f32((word >> 22u) ^ word) / 4294967296.0;
}

fn basis(n: vec3<f32>) -> mat3x3<f32> {
    let axis = select(vec3<f32>(0, 1, 0), vec3<f32>(1, 0, 0), abs(n.y) > 0.99);
    let tangent = normalize(cross(axis, n));
    return mat3x3<f32>(tangent, cross(n, tangent), n);
}

fn f0(mat: Material) -> vec3<f32> {
    return mix(vec3<f32>(0.04), min(mat.base_roughness.rgb, vec3<f32>(1.0)), mat.emission_metallic.w);
}

fn fresnel(mat: Material, cos_angle: f32) -> vec3<f32> {
    let normal = f0(mat);
    return normal + (1.0 - normal) * pow(1.0 - saturate(cos_angle), 5.0);
}

fn diffuse(mat: Material) -> vec3<f32> {
    return mat.base_roughness.rgb * (1.0 - mat.emission_metallic.w) * (1.0 - f0(mat));
}

fn ggx_distribution(alpha: f32, nh: f32) -> f32 {
    let a2 = alpha * alpha;
    let bottom = 1.0 + (a2 - 1.0) * nh * nh;
    return a2 / max(PI * bottom * bottom, 1e-9);
}

fn ggx_visibility(alpha: f32, nv: f32, nl: f32) -> f32 {
    let a2 = alpha * alpha;
    let light = nv * sqrt(a2 + (1.0 - a2) * nl * nl);
    let view = nl * sqrt(a2 + (1.0 - a2) * nv * nv);
    return 0.5 / max(light + view, 1e-5);
}

fn bsdf(mat: Material, n: vec3<f32>, v: vec3<f32>, l: vec3<f32>) -> vec3<f32> {
    let half_vector = normalize(v + l);
    let nv = max(dot(n, v), 0.001);
    let nl = saturate(dot(n, l));
    let nh = saturate(dot(n, half_vector));
    let a = max(mat.base_roughness.w * mat.base_roughness.w, 0.01);
    return diffuse(mat) / PI + fresnel(mat, dot(v, half_vector))
        * ggx_distribution(a, nh) * ggx_visibility(a, nv, nl);
}

fn sun_and_lights(p: vec3<f32>, ng: vec3<f32>, n: vec3<f32>, v: vec3<f32>, mat: Material,
                  seed: ptr<function, u32>) -> vec3<f32> {
    let origin = p + ng * (2.0 * EPS);
    var direct = vec3<f32>(0.0);
    let sun = -frame.sun_dir.xyz;
    let nl = saturate(dot(n, sun));
    if nl > 0.0 && dot(ng, sun) > 0.0 && any(frame.sun_color.rgb > vec3<f32>(0.0))
        && closest(Ray(origin, sun), frame.clip.y, true).object == MISS {
        direct += bsdf(mat, n, v, sun) * (nl * frame.sun_color.rgb);
    }
    for (var i = 0u; i < min(256u, frame.light_info.x); i++) {
        let light = lights[i];
        if !any(light.radiance.rgb > vec3<f32>(0.0)) { continue; }
        let travel = light.direction_outer.xyz;
        var position = light.position_range.xyz;
        let area = light.kind == 3u;
        if area {
            let uv = vec2<f32>(random(seed) - 0.5, random(seed) - 0.5);
            position += basis(travel) * vec3<f32>(uv.x * light.width, uv.y * light.height, 0.0);
        }
        let delta = position - p;
        var distance = length(delta);
        var l = delta / max(distance, EPS);
        var weight = 1.0;
        if light.kind == 2u {
            l = -travel;
            distance = frame.clip.y;
        } else {
            if distance >= light.position_range.w { continue; }
            let window = saturate(1.0 - pow(distance / light.position_range.w, 4.0));
            weight = window * window / max(distance * distance, 0.04);
            if area { weight *= light.width * light.height * saturate(dot(travel, -l)); }
            if light.kind == 1u { weight *= smoothstep(light.direction_outer.w, light.inner, dot(travel, -l)); }
        }
        let nl = saturate(dot(n, l));
        if nl > 0.0 && dot(ng, l) > 0.0 && weight > 0.0
            && closest(Ray(origin, l), distance - 2.0 * EPS, true).object == MISS {
            direct += bsdf(mat, n, v, l) * (nl * weight * light.radiance.rgb);
        }
    }
    return direct;
}

// Read the current radiance on the visible side of the exact mesh hit, as the
// GI clipmap propagates one-sided outgoing radiance around each surface.
fn cached_surface_radiance(p: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let params = raa_gi_volume_params;
    let level = raa_gi_level_at(p, 0u);
    if level >= params.level_count { return vec3<f32>(0.0); }
    let world = p + normal * (0.6 * params.levels[level].w);
    let sample_level = raa_gi_level_at(world, level);
    if sample_level >= params.level_count { return vec3<f32>(0.0); }
    let sample = raa_gi_sample(world, sample_level);
    if sample.a > 3.0 * params.levels[sample_level].w { return vec3<f32>(0.0); }
    return sample.rgb;
}

// Estimate incident diffuse at the actual mesh hit from the current world clipmap.
// The cosine-weighted mean is irradiance / pi, so use the same diffuse weight
// as the direct BRDF. Only distinct hits supply cached outgoing radiance.
fn terminal_diffuse_gi(p: vec3<f32>, ng: vec3<f32>, n: vec3<f32>, mat: Material) -> vec3<f32> {
    let params = raa_gi_volume_params;
    let weight = diffuse(mat) * params.intensity;
    if (params.flags & GI_VOLUME_VALID) == 0u || !any(weight > vec3<f32>(0.0))
        || raa_gi_level_at(p, 0u) >= params.level_count {
        return vec3<f32>(0.0);
    }
    let origin = p + ng * (2.0 * EPS);
    let surface_basis = basis(n);
    let sky_scale = bitcast<f32>(options[2].x);
    let sky_lod = f32(textureNumLevels(sky) - 1u);
    const SAMPLES: u32 = 12u;
    var incident = vec3<f32>(0.0);
    for (var i = 0u; i < SAMPLES; i++) {
        let u = (f32(i) + 0.5) / f32(SAMPLES);
        let phi = f32(i) * 2.39996323;
        let direction = surface_basis * vec3<f32>(sqrt(u) * cos(phi), sqrt(u) * sin(phi), sqrt(1.0 - u));
        if dot(ng, direction) <= 0.0 { continue; }
        let hit = closest(Ray(origin, direction), frame.clip.y, false);
        if hit.object == MISS {
            if sky_scale > 0.0 {
                incident += textureSampleLevel(sky, raa_linear_clamp, direction, sky_lod).rgb * sky_scale;
            }
            continue;
        }
        let tri = triangles[hit.object];
        let normal = normalize(cross(tri.b.position - tri.a.position, tri.c.position - tri.a.position));
        let facing = select(-normal, normal, dot(normal, direction) < 0.0);
        incident += cached_surface_radiance(origin + direction * hit.t, facing);
    }
    return weight * (incident / f32(SAMPLES));
}

fn trace_path(initial: Ray, mirrors: u32, path: bool, seed: ptr<function, u32>) -> vec3<f32> {
    var ray = initial;
    var throughput = vec3<f32>(1.0);
    var sum = vec3<f32>(0.0);
    var mirror_steps = mirrors;
    var path_steps = options[0].z;
    for (var step = 0u; step < 13u; step++) {
        let hit = closest(ray, frame.clip.y, false);
        if hit.object == MISS {
            return sum + throughput * textureSampleLevel(sky, raa_linear_clamp, ray.direction, 0).rgb;
        }
        if !path && step == 0u { atomicAdd(&counters[GEOMETRY_HIT], 1u); }
        let tri = triangles[hit.object];
        let mat = materials[tri.a.material];
        let p = ray.origin + ray.direction * hit.t;
        let ng0 = normalize(cross(tri.b.position - tri.a.position, tri.c.position - tri.a.position));
        let ng = select(-ng0, ng0, dot(ng0, ray.direction) < 0.0);
        let ns = normalize(tri.a.normal * (1.0 - hit.bary.x - hit.bary.y)
            + tri.b.normal * hit.bary.x + tri.c.normal * hit.bary.y);
        let oriented = select(-ns, ns, dot(ns, ng) >= 0.0);
        let n = select(ng, oriented, dot(oriented, -ray.direction) > 0.001);
        if (mat.flags.x & 1u) != 0u && mirror_steps > 0u {
            atomicAdd(&counters[MIRROR_RAYS], 1u);
            ray = Ray(p + ng * (2.0 * EPS), reflect(ray.direction, ng));
            throughput *= 0.92;
            mirror_steps--;
            continue;
        }
        sum += throughput * mat.emission_metallic.rgb;
        if (mat.flags.x & 2u) != 0u { return sum + throughput * mat.base_roughness.rgb; }
        sum += throughput * sun_and_lights(p, ng, n, -ray.direction, mat, seed);
        if !path {
            sum += throughput * terminal_diffuse_gi(p, ng, n, mat);
            break;
        }
        if path_steps == 0u { break; }

        let chance = clamp(max(f0(mat).x, max(f0(mat).y, f0(mat).z)), 0.12, 0.88);
        let select_specular = random(seed) < chance;
        let angle = 2.0 * PI * random(seed);
        let xi = min(random(seed), 0.99999);
        let a = max(mat.base_roughness.w * mat.base_roughness.w, 0.01);
        var out_dir = basis(n) * vec3<f32>(sqrt(xi) * cos(angle), sqrt(xi) * sin(angle), sqrt(1.0 - xi));
        if select_specular {
            let cos_half = sqrt((1.0 - xi) / (1.0 + (a * a - 1.0) * xi));
            let sin_half = sqrt(max(0.0, 1.0 - cos_half * cos_half));
            let half_vector = basis(n) * vec3<f32>(sin_half * cos(angle), sin_half * sin(angle), cos_half);
            out_dir = reflect(ray.direction, half_vector);
        }
        let nl = dot(n, out_dir);
        if nl <= 0.0 || dot(ng, out_dir) <= 0.0 { break; }
        let v = -ray.direction;
        let half_vector = normalize(v + out_dir);
        let nh = saturate(dot(n, half_vector));
        let pdf_spec = ggx_distribution(a, nh) * nh / max(4.0 * dot(v, half_vector), 1e-5);
        let pdf = mix(nl / PI, pdf_spec, chance);
        throughput *= bsdf(mat, n, v, out_dir) * (nl / max(pdf, 1e-5));
        ray = Ray(p + ng * (2.0 * EPS), normalize(out_dir));
        atomicAdd(&counters[PATH_RAYS], 1u);
        path_steps--;
    }
    return sum;
}

fn world_at(uv: vec2<f32>, z: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, z, 1.0);
    let p = frame.inv_view * frame.inv_proj * ndc;
    return p.xyz / p.w;
}

fn pixel_at(uv: vec2<f32>, dims: vec2<u32>) -> vec2<i32> {
    return clamp(vec2<i32>(uv * vec2<f32>(dims)), vec2<i32>(0), vec2<i32>(dims) - 1);
}

fn uv_at(pixel: vec2<i32>, dims: vec2<u32>) -> vec2<f32> {
    return (vec2<f32>(pixel) + 0.5) / vec2<f32>(dims);
}

fn on_screen(uv: vec2<f32>) -> bool {
    return all(uv > vec2<f32>(0.0)) && all(uv < vec2<f32>(1.0));
}

fn linear_depth(z: f32) -> f32 {
    let near = frame.clip.x;
    let far = frame.clip.y;
    if frame.proj[3].w != 0.0 { return mix(near, far, z); }
    return near * far / max(far - z * (far - near), near);
}

// Require the current world point to be visible in the previous frame. Moving
// hits with mismatched reprojection use geometry instead of an unverified history.
fn visible_in_history(pixel: vec2<i32>, dims: vec2<u32>) -> bool {
    if options[1].z == 0u || frame.temporal.z == 0u { return false; }
    let m = textureLoad(motion, pixel, 0);
    if m.z >= 0.5 { return false; }
    let uv = uv_at(pixel, dims);
    let p = world_at(uv, textureLoad(depth, pixel, 0));
    let c = frame.previous_view_proj * vec4<f32>(p, 1.0);
    if c.w <= 0.0 { return false; }
    let ndc = c.xyz / c.w;
    let previous_uv = vec2<f32>(0.5 + 0.5 * ndc.x, 0.5 - 0.5 * ndc.y);
    let jitter_uv = (frame.jitter.xy - frame.jitter.zw) * vec2<f32>(0.5, -0.5);
    let moved = abs(previous_uv - (uv - m.xy - jitter_uv)) * vec2<f32>(dims);
    if !on_screen(previous_uv) || ndc.z < 0.0 || ndc.z >= 1.0 || any(moved > vec2<f32>(0.7)) { return false; }
    let prior = textureLoad(previous_depth, pixel_at(previous_uv, dims), 0).r;
    let expected = linear_depth(ndc.z);
    return prior < 1.0 && abs(linear_depth(prior) - expected) <= max(0.02, expected * 0.0075);
}

fn screen_status(screen: vec4<f32>, receiver: vec2<i32>, dims: vec2<u32>) -> u32 {
    if screen.w < -1.5 { return OFFSCREEN; }
    if screen.w != 1.0 { return SCREEN_MISS; }
    if !on_screen(screen.xy) { return OFFSCREEN; }
    let hit = pixel_at(screen.xy, dims);
    if textureLoad(depth, hit, 0) >= 1.0 { return SCREEN_MISS; }
    if textureLoad(motion, hit, 0).w > 0.5 { return NESTED_MIRROR; }
    if !visible_in_history(receiver, dims) || !visible_in_history(hit, dims) { return DISOCCLUSION; }
    if screen.z <= 0.0 || screen.z < bitcast<f32>(options[1].x) { return LOW_CONFIDENCE; }
    return ACCEPTED;
}

@compute @workgroup_size(8, 8)
fn trace_mesh(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(radiance);
    if any(id.xy >= size) { return; }
    let sample_pixel = vec2<i32>(id.xy);
    let uv = uv_at(sample_pixel, size);
    var seed = id.x + id.y * size.x + 1u;
    if options[0].w != 0u {
        atomicAdd(&counters[REFERENCE_PIXELS], 1u);
        var value = vec3<f32>(0.0);
        let mirrors = select(0u, options[0].y, options[1].y != 0u);
        for (var i = 0u; i < options[1].w; i++) {
            let subpixel = (vec2<f32>(random(&seed), random(&seed)) - 0.5) / vec2<f32>(size);
            let origin = world_at(uv + subpixel, 0.0);
            let far = world_at(uv + subpixel, 1.0);
            value += trace_path(Ray(origin, normalize(far - origin)), mirrors, true, &seed);
        }
        textureStore(radiance, sample_pixel, vec4<f32>(value / f32(options[1].w), 1.0));
        return;
    }

    let dims = textureDimensions(depth);
    let pixel = pixel_at(uv, dims);
    let z = textureLoad(depth, pixel, 0);
    if z >= 1.0 || textureLoad(motion, pixel, 0).w < 0.5 {
        textureStore(radiance, sample_pixel, vec4<f32>(0.0));
        return;
    }
    atomicAdd(&counters[ELIGIBLE], 1u);
    let screen = textureLoad(confidence, pixel, 0);
    let status = screen_status(screen, pixel, dims);
    atomicAdd(&counters[status], 1u);
    if status == ACCEPTED {
        let value = textureLoad(screen_radiance, pixel_at(screen.xy, dims), 0);
        textureStore(radiance, sample_pixel, vec4<f32>(value.rgb, 1.0));
        return;
    }

    // All exits above precede the first BVH lookup, including the primary mirror hit.
    atomicAdd(&counters[TRACED], 1u);
    let center = uv_at(pixel, dims);
    let p = world_at(center, z);
    let origin = world_at(center, 0.0);
    let normal = normalize((frame.inv_view * vec4<f32>(textureLoad(normals, pixel, 0).xyz, 0.0)).xyz);
    let ray = Ray(p + normal * (2.0 * EPS), reflect(normalize(p - origin), normal));
    atomicAdd(&counters[MIRROR_RAYS], 1u);
    let value = trace_path(ray, options[0].y - 1u, false, &seed);
    textureStore(radiance, sample_pixel, vec4<f32>(value, 1.0));
}
