// The spatial half of the screen-part denoiser: variance-guided a-trous passes over the
// accumulated irradiance (Schied et al., "Spatiotemporal Variance-Guided Filtering", HPG 2017,
// section 4.3 and eq. 1-3). Pass i reads (irradiance rgb, variance a), weighs a 5 x 5 B3-spline
// kernel spread 2^i pixels apart by a plane term (the tap's distance from this pixel's tangent
// plane), a normal term and a luminance term scaled by the 3 x 3 blurred variance, and writes the
// filtered irradiance with its variance filtered by the squared weights. The last pass writes the
// screen output instead: the filtered irradiance and the history's occlusion. Because the
// variance is that of the accumulated mean (accumulate.wgsl), the filter loosens on a fresh
// history and tightens as a still view converges. Built on the bilateral of
// area_shadow/spatial.wgsl (Gaussian times depth and normal terms, normalised weights). RAA's own
// code; CPU twins in raa_gi::parts::screen::accumulate_cpu (`atrous_*`).
//
// Composed after probe_common.wgsl (bindings 0 1 2). Group-2 bindings declared here: 15 the
// input, 16 the output, 17 the history (occlusion), 18 the pass record.

struct RaaGiAtrousGpu {
    step: i32,
    last: u32,
    pad0: u32,
    pad1: u32,
}

@group(2) @binding(15) var raa_gi_atrous_in: texture_2d<f32>;
@group(2) @binding(16) var raa_gi_atrous_out: texture_storage_2d<rgba16float, write>;
@group(2) @binding(17) var raa_gi_atrous_history: texture_2d<f32>;
@group(2) @binding(18) var<uniform> raa_gi_atrous: RaaGiAtrousGpu;

// The 1D B3-spline kernel (1, 4, 6, 4, 1) / 16 (`accumulate_cpu::ATROUS_KERNEL`).
const RAA_GI_ATROUS_KERNEL: array<f32, 5> = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);
// Luminance edge-stopping: sigma_l of SVGF eq. 3 (`accumulate_cpu::ATROUS_SIGMA_LUMINANCE`).
const RAA_GI_ATROUS_SIGMA_LUMINANCE: f32 = 4.0;
// Normal edge-stopping exponent (`accumulate_cpu::ATROUS_NORMAL_POWER`).
const RAA_GI_ATROUS_NORMAL_POWER: f32 = 32.0;
// Plane edge-stopping: a tap this fraction of the view depth off the tangent plane weighs 1 / e.
const RAA_GI_ATROUS_PLANE: f32 = 0.01;

// The weight of one tap from its kernel weight and its three edge terms.
fn raa_gi_atrous_weight(
    kernel: f32,
    plane_distance: f32,
    depth: f32,
    normal_cos: f32,
    luminance_delta: f32,
    sigma: f32,
) -> f32 {
    let w_plane = exp(-abs(plane_distance) / (RAA_GI_ATROUS_PLANE * depth + 1.0e-3));
    let w_normal = pow(max(normal_cos, 0.0), RAA_GI_ATROUS_NORMAL_POWER);
    let w_luminance = exp(-abs(luminance_delta) / (RAA_GI_ATROUS_SIGMA_LUMINANCE * sigma + 1.0e-4));
    return kernel * w_plane * w_normal * w_luminance;
}

// The 3 x 3 Gaussian (1 2 1) / 16 of the variance channel around `pixel`.
fn raa_gi_atrous_blurred_variance(pixel: vec2<i32>, last: vec2<i32>) -> f32 {
    var sum = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let q = clamp(pixel + vec2<i32>(dx, dy), vec2<i32>(0), last);
            let k = f32((2 - abs(dx)) * (2 - abs(dy))) / 16.0;
            sum += textureLoad(raa_gi_atrous_in, q, 0).a * k;
        }
    }
    return sum;
}

@compute @workgroup_size(8, 8)
fn raa_gi_atrous_pass(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<u32>(raa_gi_frame.viewport.xy);
    if any(id.xy >= size) {
        return;
    }
    let pixel = vec2<i32>(id.xy);
    let depth_raw = textureLoad(raa_gi_depth, pixel, 0);
    if raa_depth_is_far(depth_raw) {
        let empty = select(vec4<f32>(0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0), raa_gi_atrous.last != 0u);
        textureStore(raa_gi_atrous_out, pixel, empty);
        return;
    }
    let last = vec2<i32>(size) - vec2<i32>(1);
    let center = vec2<f32>(id.xy) + vec2<f32>(0.5);
    let world = raa_gi_world_position(center, depth_raw);
    let depth = -raa_gi_view_position(center, depth_raw).z;
    let normal = raa_gi_world_normal(pixel);
    let own = textureLoad(raa_gi_atrous_in, pixel, 0);
    let own_luminance = dot(own.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let sigma = sqrt(max(raa_gi_atrous_blurred_variance(pixel, last), 0.0));

    var sum = vec3<f32>(0.0);
    var variance = 0.0;
    var total = 0.0;
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let q = pixel + vec2<i32>(dx, dy) * raa_gi_atrous.step;
            if any(q < vec2<i32>(0)) || any(q > last) {
                continue;
            }
            let kernel = RAA_GI_ATROUS_KERNEL[dx + 2] * RAA_GI_ATROUS_KERNEL[dy + 2];
            var w = kernel;
            let tap = textureLoad(raa_gi_atrous_in, q, 0);
            if dx != 0 || dy != 0 {
                let dq = textureLoad(raa_gi_depth, q, 0);
                if raa_depth_is_far(dq) {
                    continue;
                }
                let xq = raa_gi_world_position(vec2<f32>(q) + vec2<f32>(0.5), dq);
                let nq = raa_gi_world_normal(q);
                let lq = dot(tap.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
                w = raa_gi_atrous_weight(kernel, dot(normal, xq - world), depth, dot(normal, nq),
                    lq - own_luminance, sigma);
            }
            sum += tap.rgb * w;
            variance += tap.a * w * w;
            total += w;
        }
    }
    let filtered = sum / max(total, 1.0e-6);
    let filtered_variance = variance / max(total * total, 1.0e-12);
    if raa_gi_atrous.last != 0u {
        let occlusion = textureLoad(raa_gi_atrous_history, pixel, 0).a;
        textureStore(raa_gi_atrous_out, pixel, vec4<f32>(filtered, occlusion));
    } else {
        textureStore(raa_gi_atrous_out, pixel, vec4<f32>(filtered, min(filtered_variance, 65000.0)));
    }
}
