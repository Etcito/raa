// Voxel lighting of the GI volume (share 3): one relit clipmap level per frame. Direct light is
// shadowed through the volume itself, indirect light gathers three cosine rays per voxel and the
// result blends into the radiance stored by earlier updates.
//
// Adapted from solstice@7e465b5, crates/solstice-render/shaders/gi/voxel_light.wgsl:12-107 and
// crates/solstice-render/shaders/common/math.wgsl:7-77. MIT License
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

// S38 record order: wrap_origin @0, level @12, updates @16, resolution @20, zero @24..32.
struct GiLightGpu {
    wrap_origin: vec3<i32>,
    level: u32,
    updates: u32,
    resolution: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(2) @binding(0) var<uniform> raa_gi_light: GiLightGpu;
@group(2) @binding(1) var<storage, read> raa_gi_voxels_read: array<u32>;
@group(2) @binding(2) var<storage, read_write> raa_gi_radiance: array<vec2<u32>>;
@group(2) @binding(3) var<uniform> raa_gi_frame: GiFrameGpu;

const RAA_GI_PI: f32 = 3.14159265359;
const RAA_GI_TAU: f32 = 6.28318530718;
const RAA_GI_EMISSIVE_RANGE: f32 = 128.0;
const RAA_GI_INDIRECT_RAYS: u32 = 3u;
const RAA_GI_LIGHT_BLEND: f32 = 0.22;

// Signed octahedral coordinates in [-1, 1]^2 to a unit vector.
fn raa_gi_oct_decode(e: vec2<f32>) -> vec3<f32> {
    var v = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if v.z < 0.0 {
        let signs = vec2<f32>(select(-1.0, 1.0, v.x >= 0.0), select(-1.0, 1.0, v.y >= 0.0));
        let folded = (vec2<f32>(1.0) - abs(v.yx)) * signs;
        v = vec3<f32>(folded.x, folded.y, v.z);
    }
    return normalize(v);
}

fn raa_gi_decode_emissive(packed: u32) -> vec3<f32> {
    let e = unpack4x8unorm(packed);
    return e.rgb * e.a * RAA_GI_EMISSIVE_RANGE;
}

// PCG hash: a well-distributed 32-bit integer hash.
fn raa_gi_pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Hash to a float in [0, 1).
fn raa_gi_hash_to_unit(v: u32) -> f32 {
    return f32(raa_gi_pcg(v) >> 8u) * (1.0 / 16777216.0);
}

// R2 low-discrepancy sequence, offset by a per-voxel seed.
fn raa_gi_r2_sequence(index: u32, seed: vec2<f32>) -> vec2<f32> {
    return fract(seed + f32(index) * vec2<f32>(0.7548776662, 0.5698402910));
}

// An orthonormal basis around `n` (Duff et al. 2017), returned as tangent, bitangent.
fn raa_gi_orthonormal_basis(n: vec3<f32>) -> mat3x3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    return mat3x3<f32>(
        vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x),
        vec3<f32>(b, s + n.y * n.y * a, -n.y),
        n,
    );
}

// A cosine-weighted direction around +Z of the given basis.
fn raa_gi_cosine_sample(u: vec2<f32>, basis: mat3x3<f32>) -> vec3<f32> {
    let r = sqrt(u.x);
    let phi = RAA_GI_TAU * u.y;
    return basis * vec3<f32>(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u.x)));
}

// S11 per-kind local-light irradiance at `origin` with surface `normal`: point window and
// falloff, spot cone, area representative point and directional without attenuation, each
// times the cosine and the volume visibility toward the light.
fn raa_gi_local_light(light: LightGpu, origin: vec3<f32>, normal: vec3<f32>, voxel: f32) -> vec3<f32> {
    let travel = light.direction_cos_outer.xyz;
    var direction = -travel;
    var distance = 0.0;
    var falloff = 1.0;
    if light.kind == 0u || light.kind == 1u {
        let to_light = light.position_range.xyz - origin;
        distance = length(to_light);
        // No incident direction exists at coincidence; return finite black before division.
        if distance == 0.0 || distance >= light.position_range.w { return vec3<f32>(0.0); }
        direction = to_light / distance;
        let q = distance / light.position_range.w;
        let window = clamp(1.0 - q * q * q * q, 0.0, 1.0);
        falloff = window * window / max(distance * distance, 0.04);
        if light.kind == 1u {
            let cosine = dot(travel, -direction);
            if light.cos_inner <= light.direction_cos_outer.w {
                falloff *= select(0.0, 1.0, cosine >= light.direction_cos_outer.w);
            } else {
                falloff *= smoothstep(light.direction_cos_outer.w, light.cos_inner, cosine);
            }
        }
    } else if light.kind == 3u {
        // One-sided emission along +travel; the Karis representative point is on the rectangle.
        if dot(origin - light.position_range.xyz, travel) <= 0.0 { return vec3<f32>(0.0); }
        let up0 = select(
            vec3<f32>(1.0, 0.0, 0.0),
            vec3<f32>(0.0, 1.0, 0.0),
            abs(dot(travel, vec3<f32>(0.0, 1.0, 0.0))) < 0.99,
        );
        let tangent = normalize(cross(up0, travel));
        let bitangent = cross(travel, tangent);
        let offset = origin - light.position_range.xyz;
        let half_extent = vec2<f32>(max(light.width, 0.0), max(light.height, 0.0)) * 0.5;
        let representative = light.position_range.xyz
            + tangent * clamp(dot(offset, tangent), -half_extent.x, half_extent.x)
            + bitangent * clamp(dot(offset, bitangent), -half_extent.y, half_extent.y);
        let to_light = representative - origin;
        distance = length(to_light);
        if distance == 0.0 || distance >= light.position_range.w { return vec3<f32>(0.0); }
        direction = to_light / distance;
        let q = distance / light.position_range.w;
        let window = clamp(1.0 - q * q * q * q, 0.0, 1.0);
        falloff = window * window / max(distance * distance, 0.04);
    } else if light.kind != 2u {
        return vec3<f32>(0.0);
    }
    let cosine = dot(normal, direction);
    if cosine <= 0.0 || falloff <= 0.0 { return vec3<f32>(0.0); }
    let max_t = select(max(distance - 3.0 * voxel, 0.0), 400.0, light.kind == 2u);
    let open = raa_gi_visibility(origin, direction, voxel, max_t, 24u);
    return light.color_inv_range_sq.rgb * (falloff * cosine * open);
}

@compute @workgroup_size(4, 4, 4)
fn raa_gi_voxel_light(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = raa_gi_light.resolution;
    if any(id >= vec3<u32>(n)) {
        return;
    }
    // Refresh half of the voxels each update, in a checkerboard.
    if ((id.x + id.y + id.z + raa_gi_light.updates) & 1u) != 0u {
        return;
    }
    let level = raa_gi_light.level;
    let level_offset = level * n * n * n;
    let local_index = id.z * n * n + id.y * n + id.x;
    let word = (level_offset + local_index) * 3u;
    let albedo_word = raa_gi_voxels_read[word];
    if (albedo_word & 0xff000000u) == 0u {
        return;
    }

    let grid = raa_gi_volume_params.levels[level];
    let voxel = grid.w;
    let albedo = unpack4x8unorm(albedo_word).rgb;
    let normal = raa_gi_oct_decode(unpack2x16snorm(raa_gi_voxels_read[word + 1u]));
    let emissive = raa_gi_decode_emissive(raa_gi_voxels_read[word + 2u]);
    let center = grid.xyz + (vec3<f32>(id) + vec3<f32>(0.5)) * voxel;
    let origin = center + normal * voxel * 1.5;

    // Sun, through the clouds and the volume.
    var irradiance = vec3<f32>(0.0);
    let sun = raa_gi_frame.sun_direction.xyz;
    let sun_cos = dot(normal, sun);
    if sun_cos > 0.0 {
        irradiance += raa_directional_radiance(raa_gi_frame.sun_color.rgb)
            * raa_cloud_shadow(center)
            * sun_cos
            * raa_gi_visibility(origin, sun, voxel, 400.0, 48u);
    }

    // Local lights, with visibility so that walls contain them.
    let light_count = min(raa_gi_frame.counts.x, MAX_LOCAL_LIGHTS);
    for (var i = 0u; i < light_count; i++) {
        irradiance += raa_gi_local_light(raa_local_lights[i], origin, normal, voxel);
    }

    // Indirect: light from other surfaces and the sky, three cosine rays from the R2 sequence.
    let basis = raa_gi_orthonormal_basis(normal);
    let seed = vec2<f32>(
        raa_gi_hash_to_unit(local_index * 7u + level),
        raa_gi_hash_to_unit(local_index * 13u + 5u),
    );
    var gathered = vec3<f32>(0.0);
    for (var r = 0u; r < RAA_GI_INDIRECT_RAYS; r++) {
        let direction = raa_gi_cosine_sample(
            raa_gi_r2_sequence(raa_gi_light.updates * RAA_GI_INDIRECT_RAYS + r, seed),
            basis,
        );
        let hit = raa_gi_trace(origin, direction, voxel, 320.0, 32u, 0.85);
        let env = textureSampleLevel(raa_env_specular, raa_linear_clamp, direction, 0.0).rgb;
        gathered += select(env, hit.radiance, hit.hit);
    }
    // With cosine-weighted rays, irradiance is pi times the mean radiance.
    irradiance += gathered / f32(RAA_GI_INDIRECT_RAYS) * RAA_GI_PI;

    let target_radiance = albedo / RAA_GI_PI * irradiance + emissive;

    // Radiance is stored by world position, wrapped around the grid, so it survives the
    // clipmap following the camera.
    let world = vec3<i32>(id) + raa_gi_light.wrap_origin;
    let wrapped = vec3<u32>((world % i32(n) + vec3<i32>(i32(n))) % i32(n));
    let slot = level_offset + wrapped.z * n * n + wrapped.y * n + wrapped.x;
    let previous = raa_gi_decode_radiance(raa_gi_radiance[slot]);
    raa_gi_radiance[slot] = raa_gi_encode_radiance(mix(previous, target_radiance, RAA_GI_LIGHT_BLEND));
}
