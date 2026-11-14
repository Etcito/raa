//! S37 GI volume tracing library: `GI_VOLUME_WGSL` validates after `PRELUDE_WGSL` with a
//! fragment caller and, without the prelude, after a compute module declaring bindings 6, 17
//! and 18 itself; its text keeps the contract names, declares no binding and samples only
//! with `textureSampleLevel`.

use raa_shader::{Composer, GI_VOLUME_WGSL, PRELUDE_WGSL};

const FRAGMENT_CALLER: &str = "
@fragment
fn fs_main(@location(0) p: vec3<f32>, @location(1) d: vec3<f32>) -> @location(0) vec4<f32> {
    let v0 = raa_gi_finest_voxel();
    let hit = raa_gi_trace(p, d, v0, 320.0, 32u, 0.85);
    let vis = raa_gi_visibility(p, d, v0, 400.0, 24u);
    let fade = raa_gi_edge_fade(p);
    return vec4<f32>(hit.radiance * vis * fade, select(0.0, hit.t, hit.hit));
}
";

const COMPUTE_DECLARATIONS: &str = "
struct RaaGiVolumeParams { levels: array<vec4<f32>, 5>, resolution: f32, intensity: f32, level_count: u32, flags: u32 }
@group(1) @binding(6) var raa_linear_clamp: sampler;
@group(1) @binding(17) var raa_gi_volume: texture_3d<f32>;
@group(1) @binding(18) var<uniform> raa_gi_volume_params: RaaGiVolumeParams;
@group(2) @binding(0) var<storage, read_write> out_values: array<vec4<f32>>;
";

const COMPUTE_CALLER: &str = "
@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = vec3<f32>(f32(id.x), 1.0, 2.0);
    let d = vec3<f32>(0.0, 1.0, 0.0);
    let v0 = raa_gi_finest_voxel();
    let hit = raa_gi_trace(p, d, v0, 320.0, 32u, 0.85);
    let vis = raa_gi_visibility(p, d, v0, 400.0, 48u);
    let level = raa_gi_level_at(p, 0u);
    out_values[id.x] = vec4<f32>(hit.radiance * vis, raa_gi_edge_fade(p) + f32(level));
}
";

#[test]
fn prelude_with_fragment_caller_validates() {
    let wgsl = format!("{PRELUDE_WGSL}\n{GI_VOLUME_WGSL}\n{FRAGMENT_CALLER}");
    if let Err(e) = Composer::validate(&wgsl) {
        panic!("prelude + GI_VOLUME_WGSL fragment caller: {e}");
    }
}

#[test]
fn compute_caller_without_prelude_validates() {
    let wgsl = format!("{COMPUTE_DECLARATIONS}\n{GI_VOLUME_WGSL}\n{COMPUTE_CALLER}");
    if let Err(e) = Composer::validate(&wgsl) {
        panic!("GI_VOLUME_WGSL compute caller without prelude: {e}");
    }
}

#[test]
fn text_keeps_contract_names_and_attribution() {
    for name in [
        "fn raa_gi_level_at(",
        "fn raa_gi_sample(",
        "struct RaaGiHit",
        "fn raa_gi_trace(",
        "fn raa_gi_visibility(",
        "fn raa_gi_edge_fade(",
        "fn raa_gi_finest_voxel(",
    ] {
        assert!(GI_VOLUME_WGSL.contains(name), "missing {name}");
    }
    assert!(GI_VOLUME_WGSL.contains("1.0 / raa_gi_volume_params.resolution"));
    assert!(!GI_VOLUME_WGSL.contains("@binding"));
    assert!(!GI_VOLUME_WGSL.contains("textureSample("));
    assert!(!GI_VOLUME_WGSL.contains("inv_resolution:"));
    assert!(GI_VOLUME_WGSL.contains("Copyright (c) 2026 SamG-Coder"));
}
