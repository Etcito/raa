// S39 probe trace: every probe ray first tries the screen's depth pyramid, where geometry is
// exact and last frame's lighting is sharp, and on leaving it continues through the GI
// volume. Screen hits recover the contact detail the volume's voxels are too coarse to
// represent.
//
// Adapted from solstice@7e465b5,
// crates/solstice-render/shaders/gi/probe_trace.wgsl:17-62. MIT License
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
//
// Group-2 binding registry of contract 'gi-screen-wgsl' (binding number to resource):
// 0 raa_gi_frame, 1 raa_gi_depth, 2 raa_gi_normal, 3 raa_gi_history, 4 raa_gi_pyramid,
// 5 raa_gi_rays_out, 7 raa_gi_screen_params.

// Probe rays only need the screen near their origin: farther lighting comes from the volume,
// which is stable, while long screen rays flicker as surfaces enter and leave the view.
const RAA_GI_PROBE_SCREEN_DISTANCE: f32 = 8.0;
const RAA_GI_PROBE_SCREEN_STEPS: u32 = 32u;
const RAA_GI_PROBE_SCREEN_THICKNESS: f32 = 0.15;
const RAA_GI_PROBE_VOLUME_DISTANCE: f32 = 320.0;
const RAA_GI_PROBE_VOLUME_STEPS: u32 = 40u;
const RAA_GI_PROBE_VOLUME_HIT_VOXELS: f32 = 0.85;
// The hit distance stored for a ray that found neither the screen nor the volume.
const RAA_GI_PROBE_MISS_DISTANCE: f32 = 1000.0;

// The ray atlas: one texel per probe ray, rgb radiance and a the hit distance in metres.
@group(2) @binding(5) var raa_gi_rays_out: texture_storage_2d<rgba16float, write>;

// Handoff: card 'raa-gi screen: passes.rs module builders, group-2 layouts and pipelines'
// builds the trace pipeline on `raa_gi_probe_trace`, composed after screen_trace.wgsl and
// attached through `probe_trace_module_wgsl()` with entry `PROBE_TRACE_ENTRY` and the
// PROBE_RAYS constant; today the entry stores one probe ray's (radiance, hit distance) into
// raa_gi_rays_out under the group-2 registry above. The pyramid entries' handoff by the same
// card sits above `raa_gi_pyramid_copy` in depth_pyramid.wgsl:47.
@compute @workgroup_size(8, 8)
fn raa_gi_probe_trace(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(raa_gi_rays_out)) {
        return;
    }
    let tile = id.xy / PROBE_RAYS;
    let ray = id.xy % PROBE_RAYS;

    let frame = raa_gi_frame.counts.y;
    let pixel = raa_gi_probe_pixel(tile, raa_gi_frame.viewport.xy, frame);
    let depth = textureLoad(raa_gi_depth, vec2<i32>(pixel), 0);
    if raa_depth_is_far(depth) {
        textureStore(raa_gi_rays_out, id.xy, vec4<f32>(0.0, 0.0, 0.0, -1.0));
        return;
    }
    let world = raa_gi_world_position(vec2<f32>(pixel) + vec2<f32>(0.5), depth);
    let normal = raa_gi_world_normal(vec2<i32>(pixel));
    let direction = raa_gi_probe_ray_direction(tile, ray, normal, frame);

    var screen = raa_gi_screen_miss();
    var screen_radiance = vec4<f32>(0.0);
    if raa_gi_screen_params.screen_traces == 1u {
        screen = raa_gi_trace_screen(
            world + normal * 0.03,
            direction,
            RAA_GI_PROBE_SCREEN_DISTANCE,
            RAA_GI_PROBE_SCREEN_STEPS,
            RAA_GI_PROBE_SCREEN_THICKNESS,
        );
        if screen.hit {
            screen_radiance = raa_gi_screen_hit_radiance(
                world + normal * 0.03 + direction * screen.t,
            );
        }
        if screen.hit && screen.weight >= 1.0 && screen_radiance.a == 1.0 {
            textureStore(raa_gi_rays_out, id.xy, vec4<f32>(screen_radiance.rgb, screen.t));
            return;
        }
    }

    // The volume is coarse near its surfaces, so start clear of the local voxel.
    var radiance = textureSampleLevel(raa_env_specular, raa_linear_clamp, direction, 0.0).rgb;
    var t = RAA_GI_PROBE_MISS_DISTANCE;
    let level_count = raa_gi_volume_params.level_count;
    if level_count > 0u {
        let level = min(raa_gi_level_at(world, 0u), level_count - 1u);
        let voxel = raa_gi_volume_params.levels[level].w;
        let hit = raa_gi_trace(
            world + normal * voxel * 1.5,
            direction,
            voxel * 0.5,
            RAA_GI_PROBE_VOLUME_DISTANCE,
            RAA_GI_PROBE_VOLUME_STEPS,
            RAA_GI_PROBE_VOLUME_HIT_VOXELS,
        );
        if hit.hit {
            radiance = hit.radiance;
            t = hit.t;
        }
    }
    if screen.hit && screen_radiance.a == 1.0 {
        // Near the screen edge, blend towards the volume so probes do not pop.
        radiance = mix(radiance, screen_radiance.rgb, screen.weight);
        t = mix(t, screen.t, screen.weight);
    }
    textureStore(raa_gi_rays_out, id.xy, vec4<f32>(radiance, t));
}
