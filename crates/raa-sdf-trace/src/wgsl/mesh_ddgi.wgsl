// DDGI probe update: one 64-thread workgroup traces, relocates, classifies and blends one
// scheduled probe. Rays go through the shared world-space mesh BVH and are shaded with the sun,
// the local lights (shadow rays), emission and last frame's volume (multiple bounces).

struct DdgiParams {
    rotation: array<vec4<f32>, 3>,
    // rays per probe, scheduled probes, frame, flags (1 relocation, 2 classification, 4 bounce)
    counts: vec4<u32>,
    // hysteresis, ray limit, irradiance change threshold, minimum front-face distance (m)
    tuning: vec4<f32>,
    // sky scale, backface fraction threshold, unused, unused
    extra: vec4<f32>,
}

@group(0) @binding(27) var<storage, read_write> ddgi_probes: array<vec4<u32>>;
@group(0) @binding(28) var<uniform> ddgi_params: DdgiParams;
// Low 16 bits: slot; high 16 bits: updates the slot's probe already received (0 = fresh).
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

var<workgroup> ddgi_radiance: array<vec3<f32>, 256>;
var<workgroup> ddgi_direction: array<vec3<f32>, 256>;
var<workgroup> ddgi_distance: array<f32, 256>;
var<workgroup> ddgi_irr: array<vec3<f32>, 64>;
var<workgroup> ddgi_depth_tile: array<vec2<f32>, 256>;
// slot, run (0 skip, 1 trace), fresh, scheduled age
var<workgroup> ddgi_job: vec4<u32>;
var<workgroup> ddgi_origin: vec4<f32>;

fn ddgi_sky(direction: vec3<f32>) -> vec3<f32> {
    if options[0].z == 0u { return frame.ambient.rgb; }
    return ddgi_params.extra.x * textureSampleLevel(sky, raa_linear_clamp, direction, 0.0).rgb;
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
    let s = vec3<u32>(slot % v.counts.x, (slot / v.counts.x) % v.counts.y,
        slot / (v.counts.x * v.counts.y));
    let n = vec3<i32>(v.counts);
    // The unique coordinate of the current volume that maps to this slot.
    let shift = ((vec3<i32>(s) - v.base) % n + n) % n;
    return v.base + shift;
}

fn ddgi_shade(origin: vec3<f32>, direction: vec3<f32>, limit: f32, seed: ptr<function, u32>) -> vec4<f32> {
    let hit = closest(Ray(origin, direction), limit, false);
    if hit.object == MISS {
        return vec4<f32>(ddgi_sky(direction), DDGI_MISS);
    }
    let tri = mesh_triangle(hit.object);
    let geom = normalize(cross(tri.b.position - tri.a.position, tri.c.position - tri.a.position));
    if dot(geom, direction) > 0.0 && (mesh_material(tri.material).flags.x & 2u) == 0u {
        // Back face: the probe may sit inside geometry. Shorten as the reference does.
        return vec4<f32>(0.0, 0.0, 0.0, -0.2 * hit.t);
    }
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
    var rgb = material.emission_metallic.rgb;
    if (material.flags.x & 2u) != 0u {
        return vec4<f32>(rgb + material.base_roughness.rgb, hit.t);
    }
    let v = -direction;
    rgb += mesh_sun_at_hit(p, ng, n, v, material, true);
    rgb += ddgi_local_lights(p, ng, n, v, material, seed);
    if (ddgi_params.counts.w & DDGI_BOUNCE) != 0u {
        let bounce = raa_ddgi_sample(p, n, v);
        rgb += diffuse(material) * bounce.rgb * bounce.w;
    }
    return vec4<f32>(rgb, hit.t);
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
// keep the shared one-tap estimator with its own shadow ray.
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
            panels += mesh_local_at_hit(i, shadow_origin, p, ng, n, v, mat, &light_seed, true, false);
            continue;
        }
        total += ddgi_luma(ddgi_light_unshadowed(light, p, ng, n, v, mat).rgb);
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

@compute @workgroup_size(64, 1, 1)
fn ddgi_update(@builtin(local_invocation_index) lid: u32,
               @builtin(workgroup_id) group: vec3<u32>) {
    let v = raa_ddgi_volume();
    let rays = min(ddgi_params.counts.x, DDGI_MAX_RAYS);
    if lid == 0u {
        let job = ddgi_jobs[group.x];
        let slot = job & 0xffffu;
        let g = ddgi_slot_coord(v, slot);
        let row = v.meta_row + 2u * slot;
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
        ddgi_origin = vec4<f32>(vec3<f32>(g) * v.spacing + offset, 0.0);
    }
    let job = workgroupUniformLoad(&ddgi_job);
    if job.y == 0u { return; }
    let origin = ddgi_origin.xyz;
    let slot = job.x;
    let fresh = job.z != 0u;

    // 1. Trace.
    let limit = ddgi_params.tuning.y;
    for (var r = lid; r < rays; r += DDGI_THREADS) {
        let direction = ddgi_ray_direction(r, rays);
        var seed = (slot * 747796405u + r * 277803737u + ddgi_params.counts.z * 2891336453u) ^ 0x85ebca6bu;
        let shaded = ddgi_shade(origin, direction, limit, &seed);
        ddgi_radiance[r] = shaded.rgb;
        ddgi_direction[r] = direction;
        ddgi_distance[r] = shaded.w;
    }
    workgroupBarrier();

    // 2. Relocation and classification of this probe (one thread).
    if lid == 0u {
        let row = v.meta_row + 2u * slot;
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
            if d < 0.0 {
                backfaces++;
                if -5.0 * d < back_distance { back_distance = -5.0 * d; back_dir = ddgi_direction[r]; }
            } else if d < DDGI_MISS {
                if d < front_distance { front_distance = d; front_dir = ddgi_direction[r]; }
                let reach = abs(ddgi_direction[r] * d);
                if all(reach <= vec3<f32>(v.spacing)) { near_surface = true; }
            }
        }
        let backface_fraction = f32(backfaces) / f32(max(rays, 1u));
        let inside = backface_fraction > ddgi_params.extra.y;
        let flags = ddgi_params.counts.w;
        if (flags & DDGI_RELOCATE) != 0u {
            let min_front = ddgi_params.tuning.w;
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

    // 3. Blend irradiance (one interior texel per thread) and depth moments (four per thread).
    // A young probe averages its first updates progressively (weight 1/(age+1)) before the
    // configured hysteresis takes over, so a scrolled-in probe converges in a few updates.
    let age = f32(job.w);
    let hysteresis = select(min(ddgi_params.tuning.x, age / (age + 1.0)), 0.0, fresh);
    {
        let xy = vec2<u32>(lid % RAA_DDGI_IRR, lid / RAA_DDGI_IRR);
        let texel = raa_ddgi_oct_decode((vec2<f32>(xy) + 0.5) / f32(RAA_DDGI_IRR) * 2.0 - 1.0);
        var sum = vec3<f32>(0.0);
        var total = 0.0;
        for (var r = 0u; r < rays; r++) {
            if ddgi_distance[r] < 0.0 { continue; }
            let w = max(0.0, dot(texel, ddgi_direction[r]));
            sum += w * ddgi_radiance[r];
            total += w;
        }
        let old = raa_ddgi_irr_texel(v, slot, xy + 1u);
        var result = old;
        if total > 1e-4 {
            result = sum / total;
            var h = hysteresis;
            let change = max(max(abs(result.x - old.x), abs(result.y - old.y)), abs(result.z - old.z));
            if change > ddgi_params.tuning.z { h = max(0.0, h - 0.15); }
            result = mix(result, old, h);
        }
        ddgi_irr[lid] = result;
    }
    let max_depth = v.max_depth;
    for (var k = 0u; k < 4u; k++) {
        let index = lid + k * DDGI_THREADS;
        let xy = vec2<u32>(index % RAA_DDGI_DEPTH, index / RAA_DDGI_DEPTH);
        let texel = raa_ddgi_oct_decode((vec2<f32>(xy) + 0.5) / f32(RAA_DDGI_DEPTH) * 2.0 - 1.0);
        var sum = vec2<f32>(0.0);
        var total = 0.0;
        for (var r = 0u; r < rays; r++) {
            let c = max(0.0, dot(texel, ddgi_direction[r]));
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
    workgroupBarrier();

    // 4. Write both bordered tiles, whole rows per thread (two irradiance or four depth texels).
    let irr_first = slot * RAA_DDGI_IRR_TEXELS;
    for (var t = 2u * lid; t < RAA_DDGI_IRR_TEXELS; t += 2u * DDGI_THREADS) {
        let a = ddgi_border_source(vec2<u32>(t % RAA_DDGI_IRR_TILE, t / RAA_DDGI_IRR_TILE), RAA_DDGI_IRR);
        let u = t + 1u;
        let b = ddgi_border_source(vec2<u32>(u % RAA_DDGI_IRR_TILE, u / RAA_DDGI_IRR_TILE), RAA_DDGI_IRR);
        ddgi_probes[v.irr_row + (irr_first + t) / 2u] =
            ddgi_pack_irr(ddgi_irr[a.x + a.y * RAA_DDGI_IRR], ddgi_irr[b.x + b.y * RAA_DDGI_IRR]);
    }
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

// Screen resolve: the volume interpolated once per pixel of the prepass depth and normal, so the
// forward shader reads one row per pixel instead of eight probes. A row holds rgb and the blend
// weight (f16) and the camera distance of the surface it was resolved for (f32): forward falls
// back to the full lookup when its own surface differs (MSAA edges, surfaces absent from the
// prepass).
@compute @workgroup_size(8, 8, 1)
fn ddgi_resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    let screen = raa_ddgi_word(9u);
    if id.x >= screen.y || id.y >= screen.z { return; }
    let row = screen.x + id.y * screen.y + id.x;
    let pixel = vec2<i32>(id.xy);
    let z = textureLoad(depth, pixel, 0);
    let normal_view = textureLoad(normals, pixel, 0).xyz;
    if z >= 1.0 || dot(normal_view, normal_view) <= 0.0 {
        ddgi_probes[row] = vec4<u32>(0u);
        return;
    }
    let dims = vec2<f32>(f32(screen.y), f32(screen.z));
    let p = world_at((vec2<f32>(id.xy) + 0.5) / dims, z);
    let n = normalize((frame.inv_view * vec4<f32>(normal_view, 0.0)).xyz);
    let to_eye = frame.camera_pos.xyz - p;
    let distance = length(to_eye);
    let sample = raa_ddgi_sample(p, n, to_eye / max(distance, 1e-6));
    let rgb = clamp(sample.rgb, vec3<f32>(0.0), vec3<f32>(65504.0));
    ddgi_probes[row] = vec4<u32>(pack2x16float(rgb.xy), pack2x16float(vec2<f32>(rgb.z, sample.w)),
        bitcast<u32>(distance), 1u);
}
