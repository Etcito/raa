//! CPU-only validation; never dispatches a GPU or compares against GPU output.
use raa_lit::{LIGHT_TILES_WGSL, light_tiles_source};
use raa_shader::Composer;

#[test]
fn compute_entrypoint_validates() {
    let source = light_tiles_source();
    assert!(source.contains("fn light_tiles("));
    assert!(source.contains("@compute @workgroup_size(TILE_SIZE, TILE_SIZE)"));
    Composer::validate(&source).expect("validate every compute entry point");
}

#[test]
fn standard_z_and_edge_coverage_contract() {
    for required in [
        "var tile_depth: texture_depth_2d;",
        "atomicStore(&nearest_bits, bitcast<u32>(1.0))",
        "atomicStore(&farthest_bits, 0u)",
        "depth >= 0.0 && depth < 1.0",
        "atomicMin(&nearest_bits",
        "atomicMax(&farthest_bits",
        "id.xy < textureDimensions(tile_depth)",
        "nearest < 1.0 && lane < min(raa_frame.light_info.x, MAX_LOCAL_LIGHTS)",
        "min(lo + vec2<f32>(f32(TILE_SIZE)), raa_frame.viewport.xy)",
        "tile_world(centre, nearest)",
        "textureStore(tile_words, vec2<i32>(group.xy), 0, vec4<u32>(0u))",
    ] {
        assert!(LIGHT_TILES_WGSL.contains(required), "{required}");
    }
}

#[test]
fn conservative_candidates_keep_zero_scores_and_stable_indices() {
    for required in [
        "if light.kind == 2u { return true; }",
        "radius += length(vec2<f32>(light.width, light.height) * 0.5)",
        "-radius * length(planes[p].xyz)",
        "candidate_indices[lane] = lane",
        "candidate_eligible[lane] = 0u",
        "candidate_eligible[lane] = 1u",
        "candidate_scores[lane] = tile_score(light, near)",
        "vec3<f32>(0.2126, 0.7152, 0.0722)",
        "min(raw, 3.402823466e38)",
    ] {
        assert!(LIGHT_TILES_WGSL.contains(required), "{required}");
    }
    let bounds = LIGHT_TILES_WGSL.split("fn tile_reaches").nth(1).unwrap();
    let bounds = bounds.split("@compute").next().unwrap();
    assert!(!bounds.contains("tile_score"));
    assert!(!bounds.contains("cos_inner"));
}
