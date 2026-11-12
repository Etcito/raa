// World-space cloud dome of the M2 volumetric cloud backend (setting `cloudDome`): a square
// Rgba16Float texture of `cloudDomeSize`^2 texels indexed by WORLD view direction, so a moving
// camera never re-marches the clouds. Each texel marches ONE ray in the world direction its
// centre decodes to, with `cloud_samples` steps and a hash-noise offset along it, and blends the
// result into the dome's own history by count (`raa_cloud.forward.w` = n, frames accumulated
// including this one): 1/n while the sky stands still, never below 1/16 while it is animated.
// The host stops dispatching this pass once the still sky has converged.
//
// Direction mapping (twin of `clouds/dome.rs` `dome_direction`/`dome_uv` and of
// `raa_physical_sky_dome_uv` in `shaders.rs`): hemi-octahedral over the upper hemisphere, with
// the direction shifted by +DOME_BELOW in y before the encoding, so the dome reaches down to
// d.y = -DOME_BELOW and has no pole singularity.
//
// `clouds/shaders.rs` composes this text after `cloud_volume_wgsl()` and the bindings
// 10 `raa_cloud_output` (Rgba16Float, the dome) and 11 `raa_cloud_history` (the dome's previous
// result; alpha < 0 = not marched). `raa_cloud.screen` = (1, flags, jitter index, cloud_samples)
// with the flags of `screen.wgsl` (bit 1 animated sky, bit 2 dither off, bits 3.. noise block
// size); the camera lanes are unused. RGB is linear HDR cloud radiance, A the transmittance.

const RAA_CLOUD_DOME_BELOW: f32 = 0.05;
const RAA_CLOUD_DOME_LIGHT_SAMPLES: i32 = 5;
// The smallest share of a new frame while the sky is animated.
const RAA_CLOUD_DOME_ANIMATED_FLOOR: f32 = 0.0625;
// A texel without history marches this many times the steps.
const RAA_CLOUD_DOME_BURST: i32 = 2;

// Integer hash (PCG) of the texel block and a sequence index, uniform in [0, 1).
fn raa_cloud_dome_hash(pixel: vec2<u32>, sequence: u32) -> f32 {
    var v = pixel.x * 1664525u + pixel.y * 1013904223u + sequence * 747796405u + 2891336453u;
    v = v * 747796405u + 2891336453u;
    let word = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return f32(((word >> 22u) ^ word) >> 8u) * (1.0 / 16777216.0);
}

// The world direction the dome uv in [0, 1]^2 stands for.
fn raa_cloud_dome_direction(uv: vec2<f32>) -> vec3<f32> {
    let e = uv * 2.0 - 1.0;
    let p = vec2<f32>(e.x + e.y, e.x - e.y) * 0.5;
    let y = max(1.0 - abs(p.x) - abs(p.y), 0.0);
    let u = normalize(vec3<f32>(p.x, y, p.y));
    let b = RAA_CLOUD_DOME_BELOW;
    let t = b * u.y + sqrt(max(b * b * u.y * u.y - b * b + 1.0, 0.0));
    return normalize(vec3<f32>(u.x * t, u.y * t - b, u.z * t));
}

@compute @workgroup_size(8, 8)
fn raa_cloud_dome_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(raa_cloud_output);
    if any(id.xy >= size) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let direction = raa_cloud_dome_direction(uv);
    if direction.y <= -0.02 {
        // Below the horizon band nothing of the sky is seen, as in the panorama.
        textureStore(raa_cloud_output, id.xy, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let ray = normalize(vec3<f32>(direction.x, max(direction.y, 0.002), direction.z));
    let flags = u32(raa_cloud.screen.y + 0.5);
    let animated = (flags & 2u) != 0u;
    let dither_off = (flags & 4u) != 0u;
    let noise_size = clamp(flags >> 3u, 1u, 16u);
    let steps = clamp(i32(raa_cloud.screen.w + 0.5), 8, 256);
    if dither_off {
        // No noise at all: the same march every frame.
        let fixed = raa_cloud_along(ray, steps, RAA_CLOUD_DOME_LIGHT_SAMPLES, 0.5);
        textureStore(raa_cloud_output, id.xy, vec4<f32>(fixed.radiance, fixed.transmittance));
        return;
    }
    // Still frames: a fixed per-texel noise shifted by the golden ratio each frame (a
    // low-discrepancy sequence, so the running mean converges without flicker); the first
    // frame of a run: a fresh hash.
    let sequence = u32(raa_cloud.screen.z);
    let block = id.xy / vec2<u32>(noise_size);
    var jitter = raa_cloud_dome_hash(block, sequence);
    let n = raa_cloud.forward.w;
    if n >= 1.5 {
        jitter = fract(raa_cloud_dome_hash(block, 0u) + 0.61803398875 * f32(sequence));
    }
    let prior = textureLoad(raa_cloud_history, vec2<i32>(id.xy), 0);
    var result: vec4<f32>;
    if n < 1.5 || prior.a < 0.0 {
        let burst = raa_cloud_along(
            ray,
            min(steps * RAA_CLOUD_DOME_BURST, 512),
            RAA_CLOUD_DOME_LIGHT_SAMPLES,
            jitter,
        );
        result = vec4<f32>(burst.radiance, burst.transmittance);
    } else {
        let here = raa_cloud_along(ray, steps, RAA_CLOUD_DOME_LIGHT_SAMPLES, jitter);
        let mean = 1.0 / n;
        let share = select(mean, max(mean, RAA_CLOUD_DOME_ANIMATED_FLOOR), animated);
        result = mix(prior, vec4<f32>(here.radiance, here.transmittance), share);
    }
    textureStore(raa_cloud_output, id.xy, result);
}
