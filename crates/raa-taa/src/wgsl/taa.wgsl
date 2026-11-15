// Temporal anti-aliasing at render resolution 1:1. Each frame's camera jitter makes every
// pixel sample a different point; reprojecting and accumulating those samples gives both
// anti-aliasing and a stable image over time. The module is `REPROJECT_WGSL` followed by
// this source (contract 'taa-bindings').
//
// Adapted from solstice@7e465b5,
// solstice:crates/solstice-render/shaders/post/taa.wgsl:1-168.
// MIT License
//
// Copyright (c) 2026 SamG-Coder
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

@group(0) @binding(0) var<uniform> raa_taa_frame: RaaTemporalFrame;
@group(0) @binding(1) var<uniform> raa_taa_params: RaaTaaParams;
@group(0) @binding(2) var raa_taa_scene: texture_2d<f32>;
@group(0) @binding(3) var raa_taa_depth: texture_depth_2d;
@group(0) @binding(4) var raa_taa_motion: texture_2d<f32>;
@group(0) @binding(5) var raa_taa_history: texture_2d<f32>;
@group(0) @binding(6) var raa_taa_sampler: sampler;
@group(0) @binding(7) var raa_taa_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var raa_taa_history_out: texture_storage_2d<rgba16float, write>;

// Group 0 binding 1: `history_usable` is 1 while the pass's own history holds a frame
// committed since its last invalidation.
struct RaaTaaParams {
    history_usable: u32,
    reserved0: u32,
    reserved1: u32,
    reserved2: u32,
}

fn raa_taa_luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Tonemap weights keep very bright samples from dominating the filter.
fn compress(c: vec3<f32>) -> vec3<f32> {
    return c / (1.0 + raa_taa_luminance(c));
}

fn decompress(c: vec3<f32>) -> vec3<f32> {
    return c / max(1.0 - raa_taa_luminance(c), 1.0e-4);
}

fn to_ycocg(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c, vec3<f32>(0.25, 0.5, 0.25)),
        dot(c, vec3<f32>(0.5, 0.0, -0.5)),
        dot(c, vec3<f32>(-0.25, 0.5, -0.25)),
    );
}

fn from_ycocg(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// Five-tap Catmull-Rom: nearly bicubic history sampling, which is the main thing keeping
// reprojection from blurring the image over time.
fn sample_history(uv: vec2<f32>) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(raa_taa_history));
    let position = uv * size;
    let center = floor(position - vec2<f32>(0.5)) + vec2<f32>(0.5);
    let f = position - center;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let offset12 = w2 / w12;
    let t0 = (center - vec2<f32>(1.0)) / size;
    let t3 = (center + vec2<f32>(2.0)) / size;
    let t12 = (center + offset12) / size;

    var taps = array<vec3<f32>, 5>(
        vec3<f32>(t12.x, t0.y, w12.x * w0.y),
        vec3<f32>(t0.x, t12.y, w0.x * w12.y),
        vec3<f32>(t12.x, t12.y, w12.x * w12.y),
        vec3<f32>(t3.x, t12.y, w3.x * w12.y),
        vec3<f32>(t12.x, t3.y, w12.x * w3.y),
    );
    var sum = vec3<f32>(0.0);
    var total = 0.0;
    for (var i = 0; i < 5; i++) {
        let tap = taps[i];
        sum += textureSampleLevel(raa_taa_history, raa_taa_sampler, tap.xy, 0.0).rgb * tap.z;
        total += tap.z;
    }
    return sum / max(total, 1.0e-4);
}

@compute @workgroup_size(8, 8, 1)
fn cs_taa(@builtin(global_invocation_id) id: vec3<u32>) {
    let output_size = textureDimensions(raa_taa_output);
    if any(id.xy >= output_size) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + vec2<f32>(0.5)) / vec2<f32>(output_size);
    let scene_size = vec2<i32>(raa_taa_frame.viewport.xy);

    // The scene pixel nearest this output pixel, removing the jitter applied this frame.
    let scene_position = uv * raa_taa_frame.viewport.xy;
    let nearest = clamp(vec2<i32>(floor(scene_position)), vec2<i32>(0), scene_size - vec2<i32>(1));

    // Gather the 3x3 neighbourhood. Track the closest depth (standard-Z: the far plane is
    // 1.0), so that edges reproject with the foreground's motion, and the variance, which
    // bounds what history is believable.
    var mean = vec3<f32>(0.0);
    var second_moment = vec3<f32>(0.0);
    var filtered = vec3<f32>(0.0);
    var filter_total = 0.0;
    var closest_depth = 1.0;
    var closest_pixel = nearest;
    let jitter_pixels =
        raa_taa_frame.camera_jitter.xy * 0.5 * raa_taa_frame.viewport.xy * vec2<f32>(1.0, -1.0);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let p = clamp(nearest + vec2<i32>(dx, dy), vec2<i32>(0), scene_size - vec2<i32>(1));
            let color = compress(textureLoad(raa_taa_scene, p, 0).rgb);
            let ycocg = to_ycocg(color);
            mean += ycocg;
            second_moment += ycocg * ycocg;

            // A Gaussian centred on this output pixel, using where the sample really landed.
            let sample_position = vec2<f32>(p) + vec2<f32>(0.5) + jitter_pixels;
            let delta = sample_position - scene_position;
            let weight = exp(-2.3 * dot(delta, delta));
            filtered += color * weight;
            filter_total += weight;

            let depth = textureLoad(raa_taa_depth, p, 0);
            if depth < closest_depth {
                closest_depth = depth;
                closest_pixel = p;
            }
        }
    }
    mean /= 9.0;
    second_moment /= 9.0;
    let deviation = sqrt(max(second_moment - mean * mean, vec3<f32>(0.0)));
    let current = filtered / max(filter_total, 1.0e-4);

    // Reproject. The sky (closest_depth at the far plane) has no stored motion, so derive
    // it from the camera alone; geometry uses the S12 motion of the nearest pixel, whose
    // previous uv is uv - motion.xy.
    var reprojected: vec3<f32>;
    if closest_depth >= 1.0 {
        reprojected = raa_temporal_sky_uv(
            uv,
            raa_taa_frame.inv_view,
            raa_taa_frame.inv_proj,
            raa_taa_frame.previous_view_proj,
            raa_taa_frame.camera_jitter,
        );
    } else {
        reprojected = raa_temporal_surface_uv(uv, textureLoad(raa_taa_motion, closest_pixel, 0));
    }
    let previous_uv = reprojected.xy;

    var result = current;
    if reprojected.z > 0.5
        && raa_taa_params.history_usable != 0u
        && raa_temporal_history_usable(raa_taa_frame.temporal) {
        let previous = compress(max(sample_history(previous_uv), vec3<f32>(0.0)));

        // Clip history against the current neighbourhood's colour distribution: anything far
        // outside it is a disocclusion or lighting change, not a useful earlier sample.
        let tolerance = 1.25;
        let low = mean - deviation * tolerance;
        let high = mean + deviation * tolerance;
        let y = to_ycocg(previous);
        let center_box = (high + low) * 0.5;
        let extents = max((high - low) * 0.5, vec3<f32>(1.0e-4));
        let ratio = (y - center_box) / extents;
        let largest = max(abs(ratio.x), max(abs(ratio.y), abs(ratio.z)));
        let clipped = select(y, center_box + (y - center_box) / largest, largest > 1.0);

        // More weight on the current frame when the scene is moving, since resampled
        // history gets softer with distance. The render ratio is 1 (no upscaling).
        let moved = length((previous_uv - uv) * raa_taa_frame.viewport.xy);
        let blend = clamp(0.06 + moved * 0.012, 0.04, 0.3);
        result = mix(from_ycocg(clipped), current, blend);
    }
    let color = decompress(max(result, vec3<f32>(0.0)));
    let alpha = textureLoad(raa_taa_scene, nearest, 0).a;
    textureStore(raa_taa_output, id.xy, vec4<f32>(color, alpha));
    textureStore(raa_taa_history_out, id.xy, vec4<f32>(color, alpha));
}
