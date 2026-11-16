// P4 M8 screen-space indirect diffuse: the trace kernel of `gi.ssgi`. One invocation per pixel
// marches the near field of the S12 depth with a visibility bitmask of 32 cosine-weighted
// sectors per slice, after Therrien, Levesque and Gilet, "Screen Space Indirect Lighting with
// Visibility Bitmask" (2023). The slice and step loop extends raa's GTAO,
// crates/raa-gtao/src/wgsl/gtao.wgsl:95-162; the sector maths are the CPU twins of
// crates/raa-gi/src/parts/ssgi/cpu.rs (contract 'ssgi-cpu'). This is RAA's own code; it adapts
// no solstice source. Light comes from the previous lit HDR (binding 3), reprojected through
// previous_view_proj by `raa_gi_screen_hit_radiance` of screen_trace.wgsl.
//
// Per slice, `mask` holds every occluded sector (a farther sample never lights a sector a nearer
// one hides) and `lit` only the sectors a facing sample with available radiance claimed
// (`cpu::lit_mask`). Only `lit` leaves the probe term: V = mean(1 - popcount(lit) / 32), so a
// sector hidden by a back face or a sample without radiance keeps the probe irradiance.
//
// Group-2 binding registry (binding number to resource): 0 raa_gi_frame, 1 raa_gi_depth,
// 2 raa_gi_normal, 3 raa_gi_history, 5 raa_gi_ssgi_raw, 8 raa_gi_ssgi_params. Bindings 4 and 7
// stay declared by the composed screen files and are unused here.

// The per-frame trace parameters (32 B), contract 'ssgi-trace-wgsl'.
struct SsgiParamsGpu {
    slices: u32,
    steps: u32,
    history_valid: u32,
    frame: u32,
    radius_m: f32,
    thickness_m: f32,
    temporal_valid: u32,
    pad: u32,
}

// The raw trace: E_ss rgb, V a.
@group(2) @binding(5) var raa_gi_ssgi_raw: texture_storage_2d<rgba16float, write>;
@group(2) @binding(8) var<uniform> raa_gi_ssgi_params: SsgiParamsGpu;

const RAA_GI_SSGI_PI: f32 = 3.14159265359;
const RAA_GI_SSGI_HALF_PI: f32 = 1.57079632679;
const RAA_GI_SSGI_SECTORS: u32 = 32u;

// Interleaved gradient noise (Jimenez 2014), shifted per frame (gtao.wgsl:35-38).
fn raa_gi_ssgi_ign(pixel: vec2<f32>, frame: f32) -> f32 {
    let p = pixel + vec2<f32>(5.588238 * frame);
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

// `cpu::sector_u`: the cosine-weighted sector coordinate of angle theta in [0, 1].
fn raa_gi_ssgi_sector_u(theta: f32, gamma: f32) -> f32 {
    let angle = clamp(theta - gamma, -RAA_GI_SSGI_HALF_PI, RAA_GI_SSGI_HALF_PI);
    return (sin(angle) + 1.0) * 0.5;
}

// `cpu::update_sectors`: the bits of the sectors covering [min_u, max_u].
fn raa_gi_ssgi_update_sectors(min_u: f32, max_u: f32) -> u32 {
    let scale = f32(RAA_GI_SSGI_SECTORS);
    let start = min(u32(max(floor(min_u * scale), 0.0)), RAA_GI_SSGI_SECTORS - 1u);
    let count = min(u32(max(ceil((max_u - min_u) * scale), 0.0)), RAA_GI_SSGI_SECTORS - start);
    if count == 0u {
        return 0u;
    }
    return (0xFFFFFFFFu >> (RAA_GI_SSGI_SECTORS - count)) << start;
}

// The signed angle of p from the view vector, on `side` of the slice.
fn raa_gi_ssgi_view_angle(p: vec3<f32>, view: vec3<f32>, side: f32) -> f32 {
    return side * acos(clamp(dot(normalize(p), view), -1.0, 1.0));
}

// `cpu::sample_bits`: the sectors a sample at `delta` occludes, with its front face and the
// back face `thickness` metres behind it along the view vector.
fn raa_gi_ssgi_sample_bits(
    delta: vec3<f32>,
    view: vec3<f32>,
    thickness: f32,
    side: f32,
    gamma: f32,
) -> u32 {
    let back = delta - view * thickness;
    let u_front = raa_gi_ssgi_sector_u(raa_gi_ssgi_view_angle(delta, view, side), gamma);
    let u_back = raa_gi_ssgi_sector_u(raa_gi_ssgi_view_angle(back, view, side), gamma);
    return raa_gi_ssgi_update_sectors(min(u_front, u_back), max(u_front, u_back));
}

@compute @workgroup_size(8, 8)
fn raa_gi_ssgi_trace(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<u32>(raa_gi_frame.viewport.xy);
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let pixel = vec2<i32>(id.xy);
    let d = textureLoad(raa_gi_depth, pixel, 0);
    if raa_depth_is_far(d) || raa_gi_ssgi_params.history_valid == 0u {
        textureStore(raa_gi_ssgi_raw, pixel, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let c = vec2<f32>(id.xy) + vec2<f32>(0.5);
    let p = raa_gi_view_position(c, d);
    let v = normalize(-p);
    var n = normalize(textureLoad(raa_gi_normal, pixel, 0).xyz);
    if dot(n, v) < 0.0 {
        n = -n;
    }
    let radius_m = raa_gi_ssgi_params.radius_m;
    let thickness_m = raa_gi_ssgi_params.thickness_m;
    let viewport_y = raa_gi_frame.viewport.y;
    let proj11 = 1.0 / raa_gi_frame.inv_proj[1][1];
    let radius_px = min(
        radius_m * proj11 * 0.5 * viewport_y / max(-p.z, 1e-4),
        0.25 * viewport_y,
    );
    let frame = f32(raa_gi_ssgi_params.frame % 64u);
    let noise = raa_gi_ssgi_ign(c, frame);
    let jitter = raa_gi_ssgi_ign(c + vec2<f32>(17.0, 23.0), frame);
    let slices = max(raa_gi_ssgi_params.slices, 1u);
    let steps = max(raa_gi_ssgi_params.steps, 1u);

    var total = vec3<f32>(0.0);
    var open = 0.0;
    for (var s = 0u; s < slices; s = s + 1u) {
        let phi = (f32(s) + noise) * RAA_GI_SSGI_PI / f32(slices);
        let dir = vec2<f32>(cos(phi), sin(phi));
        let base3 = vec3<f32>(dir.x, -dir.y, 0.0);
        let omega = normalize(base3 - dot(base3, v) * v);
        let nv = dot(n, v);
        let no = dot(n, omega);
        var mask = 0u;
        var lit = 0u;
        var sum = vec3<f32>(0.0);
        if sqrt(nv * nv + no * no) >= 1e-4 {
            let gamma = atan2(no, nv);
            for (var k = 0u; k < 2u; k = k + 1u) {
                let side = select(-1.0, 1.0, k == 0u);
                for (var i = 0u; i < steps; i = i + 1u) {
                    let t = (f32(i) + jitter) / f32(steps);
                    let dist = max(t * t * radius_px, f32(i + 1u));
                    let sp = vec2<i32>(floor(c + side * dist * dir));
                    if any(sp < vec2<i32>(0)) || any(sp >= vec2<i32>(size)) {
                        break;
                    }
                    let sd = textureLoad(raa_gi_depth, sp, 0);
                    if raa_depth_is_far(sd) {
                        continue;
                    }
                    let q = raa_gi_view_position(vec2<f32>(sp) + vec2<f32>(0.5), sd);
                    let delta = q - p;
                    let len = length(delta);
                    if len < 1e-5 || len > radius_m {
                        continue;
                    }
                    let bits = raa_gi_ssgi_sample_bits(delta, v, thickness_m, side, gamma);
                    let fresh = bits & ~mask;
                    if fresh != 0u {
                        let world = (raa_gi_frame.inv_view * vec4<f32>(q, 1.0)).xyz;
                        let hit = raa_gi_screen_hit_radiance(world);
                        let sample_n = normalize(textureLoad(raa_gi_normal, sp, 0).xyz);
                        let facing = dot(sample_n, -delta / len) > 0.0;
                        if facing && hit.a > 0.0 {
                            let share = f32(countOneBits(fresh)) / f32(RAA_GI_SSGI_SECTORS);
                            sum += hit.rgb * share;
                            lit |= fresh;
                        }
                    }
                    mask |= bits;
                }
            }
        }
        total += sum;
        open += 1.0 - f32(countOneBits(lit)) / f32(RAA_GI_SSGI_SECTORS);
    }
    let e_ss = RAA_GI_SSGI_PI * total / f32(slices);
    textureStore(raa_gi_ssgi_raw, pixel, vec4<f32>(e_ss, open / f32(slices)));
}
