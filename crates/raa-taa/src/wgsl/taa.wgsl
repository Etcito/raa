// Temporal anti-aliasing and analytic temporal super-resolution. Each frame's camera jitter makes every
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

override RAA_TAA_FILTER_SHARPNESS: f32 = 8.0;
override RAA_TAA_CLIP_TOLERANCE: f32 = 1.25;
override RAA_TAA_BLEND_BASE: f32 = 0.04;
override RAA_TAA_BLEND_MOTION_GAIN: f32 = 0.03;
override RAA_TAA_BLEND_MIN: f32 = 0.04;
override RAA_TAA_BLEND_MAX: f32 = 0.5;
override RAA_TAA_HISTORY_SHARPNESS: f32 = 0.6;
override RAA_TAA_BLEND_STATIC: f32 = 0.015;
override RAA_TAA_CLIP_STATIC: f32 = 3.0;
override RAA_TAA_UPSAMPLE_CONFIDENCE: f32 = 3.0;
override RAA_TAA_UPSAMPLE_CLIP_SCENE: f32 = 1.0;
override RAA_TAA_UPSAMPLE_KERNEL: f32 = 1.0;

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

// Keys cubic weight at distance x in [0, 2) for a = -RAA_TAA_HISTORY_SHARPNESS: 0.5 is
// Catmull-Rom; larger values deepen the negative lobes, which offsets the softening of the
// subpixel resampling the history goes through every frame (0.6 measured best on Sponza).
fn keys_weight(x: vec2<f32>) -> vec2<f32> {
    let a = -RAA_TAA_HISTORY_SHARPNESS;
    let near = ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    let far = ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
    return select(far, near, x <= vec2<f32>(1.0));
}

// Five-tap Keys cubic (Catmull-Rom at sharpness 0.5): nearly bicubic history sampling, which
// is the main thing keeping reprojection from blurring the image over time.
fn sample_history(uv: vec2<f32>) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(raa_taa_history));
    let position = uv * size;
    let center = floor(position - vec2<f32>(0.5)) + vec2<f32>(0.5);
    let f = position - center;
    let w0 = keys_weight(vec2<f32>(1.0) + f);
    let w1 = keys_weight(f);
    let w2 = keys_weight(vec2<f32>(1.0) - f);
    let w3 = keys_weight(vec2<f32>(2.0) - f);
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
    let scene_size = vec2<i32>(textureDimensions(raa_taa_scene));
    let scene_extent = vec2<f32>(scene_size);
    let display_extent = vec2<f32>(output_size);
    let scaled = any(vec2<u32>(scene_size) != output_size);
    let scene_position = uv * scene_extent;
    let jitter_pixels = raa_taa_frame.camera_jitter.xy * 0.5 * scene_extent * vec2<f32>(1.0, -1.0);
    let raster_position = scene_position + jitter_pixels;
    // Keep the native TAA footprint. At scaled extents centre the footprint on the
    // raster position, because each raster sample represents its centre minus jitter.
    let nearest = clamp(
        vec2<i32>(floor(select(scene_position, raster_position, scaled))),
        vec2<i32>(0),
        scene_size - vec2<i32>(1),
    );

    // Gather the 3x3 neighbourhood. Track the closest depth (standard-Z: the far plane is
    // 1.0), so that edges reproject with the foreground's motion, and the variance, which
    // bounds what history is believable.
    var mean = vec3<f32>(0.0);
    var second_moment = vec3<f32>(0.0);
    var filtered = vec3<f32>(0.0);
    var filter_total = 0.0;
    var closest_depth = select(1.0, textureLoad(raa_taa_depth, nearest, 0), scaled);
    var closest_pixel = nearest;
    // Squared distance, in display pixels, from this output pixel to the nearest jittered
    // scene sample (temporal upscaling confidence).
    var nearest_sample = 1.0e9;
    // RAA_TAA_UPSAMPLE_KERNEL >= 0.5 (with a confidence > 0): the current value is the
    // display-pixel Gaussian exp(-confidence * d^2) of the jittered samples and its weight sum
    // (at most 1) is the confidence, instead of the nearest sample's distance.
    let kernel = scaled && RAA_TAA_UPSAMPLE_CONFIDENCE > 0.0 && RAA_TAA_UPSAMPLE_KERNEL >= 0.5;
    var kernel_sum = vec3<f32>(0.0);
    var kernel_total = 0.0;
    let to_display = display_extent / scene_extent;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let p = clamp(nearest + vec2<i32>(dx, dy), vec2<i32>(0), scene_size - vec2<i32>(1));
            let color = compress(textureLoad(raa_taa_scene, p, 0).rgb);
            // At scaled extents measure the clipping distribution around the display
            // pixel, using the jitter-corrected scene footprint of each display neighbour.
            let display_neighbor = clamp(
                vec2<i32>(id.xy) + vec2<i32>(dx, dy),
                vec2<i32>(0),
                vec2<i32>(output_size) - vec2<i32>(1),
            );
            var clip_color = color;
            // RAA_TAA_UPSAMPLE_CLIP_SCENE >= 0.5 keeps the 3x3 scene samples around the raster
            // position as the clipping distribution (a box of nine distinct samples instead of
            // the display neighbours' nearest samples, which repeat at upscaled extents).
            if scaled && RAA_TAA_UPSAMPLE_CLIP_SCENE < 0.5 {
                let clip_position = (vec2<f32>(display_neighbor) + vec2<f32>(0.5)) * scene_extent
                    / display_extent + jitter_pixels;
                let clip_pixel = clamp(vec2<i32>(floor(clip_position)), vec2<i32>(0), scene_size - vec2<i32>(1));
                clip_color = compress(textureLoad(raa_taa_scene, clip_pixel, 0).rgb);
            }
            let ycocg = to_ycocg(clip_color);
            mean += ycocg;
            second_moment += ycocg * ycocg;

            // A Gaussian centred on this output pixel, using where the sample really landed.
            // raa-lit's jittered projection adds camera_jitter.xy to the NDC, so the scene
            // pixel p shows the unjittered point at its centre minus the jitter (solstice adds
            // it here, which moves the filter away from the true sample and softens edges).
            let sample_position = vec2<f32>(p) + vec2<f32>(0.5) - jitter_pixels;
            let delta = sample_position - scene_position;
            let weight = exp(-RAA_TAA_FILTER_SHARPNESS * dot(delta, delta));
            let delta_display = delta * to_display;
            nearest_sample = min(nearest_sample, dot(delta_display, delta_display));
            let kernel_weight = exp(-RAA_TAA_UPSAMPLE_CONFIDENCE * dot(delta_display, delta_display));
            kernel_sum += color * kernel_weight;
            kernel_total += kernel_weight;
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
    let current = select(
        filtered / max(filter_total, 1.0e-4),
        kernel_sum / max(kernel_total, 1.0e-6),
        kernel && kernel_total > 1.0e-3,
    );

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
        // At rest (no reprojection offset) the history is the same surface point, so a
        // longer history and a looser box average the jitter positions instead of flickering.
        let moved = length((previous_uv - uv) * display_extent);
        let rest = 1.0 - smoothstep(0.0, 0.5, moved);
        let tolerance = RAA_TAA_CLIP_TOLERANCE * mix(1.0, RAA_TAA_CLIP_STATIC, rest);
        let low = mean - deviation * tolerance;
        let high = mean + deviation * tolerance;
        let y = to_ycocg(previous);
        let center_box = (high + low) * 0.5;
        let extents = max((high - low) * 0.5, vec3<f32>(1.0e-4));
        let ratio = (y - center_box) / extents;
        let largest = max(abs(ratio.x), max(abs(ratio.y), abs(ratio.z)));
        let clipped = select(y, center_box + (y - center_box) / max(largest, 1.0), largest > 1.0);

        // More weight on the current frame when the scene is moving, since resampled
        // history gets softer with distance on the accumulation grid.
        var blend = clamp(
            mix(RAA_TAA_BLEND_BASE, RAA_TAA_BLEND_STATIC, rest) + moved * RAA_TAA_BLEND_MOTION_GAIN,
            min(RAA_TAA_BLEND_MIN, RAA_TAA_BLEND_STATIC),
            RAA_TAA_BLEND_MAX,
        );
        // Temporal upscaling: a display pixel only learns from this frame as much as a jittered
        // sample landed near it; elsewhere the current value is an interpolation that would blur
        // the accumulated history (0 disables the weighting).
        if scaled && RAA_TAA_UPSAMPLE_CONFIDENCE > 0.0 {
            blend *= select(exp(-RAA_TAA_UPSAMPLE_CONFIDENCE * nearest_sample), min(kernel_total, 1.0), kernel);
        }
        result = mix(from_ycocg(clipped), current, blend);
    }
    let color = decompress(max(result, vec3<f32>(0.0)));
    let alpha = textureLoad(raa_taa_scene, nearest, 0).a;
    textureStore(raa_taa_output, id.xy, vec4<f32>(color, alpha));
    textureStore(raa_taa_history_out, id.xy, vec4<f32>(color, alpha));
}
