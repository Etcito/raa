//! Shader-structure and installation validation; GPU parts run on the software adapter
//! (SKIP without one) and never compare against GPU output.
use raa_lit::{LIGHT_TILES_WGSL, light_tiles_source};
use raa_shader::{Composer, MAX_LOCAL_LIGHTS, TILE_MAX_LIGHTS, TILE_SIZE, TILE_WORDS};

#[test]
fn tiling_feature_uses_shared_logical_id_24() {
    assert_eq!(raa_shader::feature_address(24), Some((2, 24)));
    let mut params = raa_lit::FrameParams::default();
    params.set_feature(24, true);
    assert_eq!(params.feature_bits & (1 << 24), 1 << 24);
    params.set_feature(24, false);
    assert_eq!(params.feature_bits & (1 << 24), 0);
}

#[test]
fn compute_entrypoint_validates() {
    let source = light_tiles_source();
    assert!(source.contains("fn light_tiles("));
    assert!(source.contains("@compute @workgroup_size(TILE_SIZE, TILE_SIZE)"));
    assert_eq!(source.matches("@compute").count(), 1);
    assert!(!source.contains("@vertex") && !source.contains("@fragment"));
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
        "atomicStore(&qualifying_count, 0u)",
    ] {
        assert!(LIGHT_TILES_WGSL.contains(required), "{required}");
    }
}

#[test]
fn fixed_capacity_arrays_and_workgroup_fit_core_limits() {
    assert_eq!(
        (MAX_LOCAL_LIGHTS, TILE_SIZE, TILE_MAX_LIGHTS, TILE_WORDS),
        (256, 16, 127, 128)
    );
    assert_eq!(TILE_SIZE * TILE_SIZE, MAX_LOCAL_LIGHTS as u32);
    assert_eq!(TILE_WORDS, TILE_MAX_LIGHTS + 1);
    let source = light_tiles_source();
    for (name, value) in [
        ("MAX_LOCAL_LIGHTS", MAX_LOCAL_LIGHTS as u32),
        ("TILE_SIZE", TILE_SIZE),
        ("TILE_MAX_LIGHTS", TILE_MAX_LIGHTS),
        ("TILE_WORDS", TILE_WORDS),
    ] {
        assert!(source.contains(&format!("const {name}: u32 = {value}u;")));
    }
    // Account for every workgroup declaration, refusing unbounded/additional arrays.
    let mut bytes = 0;
    for declaration in source
        .lines()
        .filter(|line| line.starts_with("var<workgroup>"))
    {
        let ty = declaration.split_once(": ").unwrap().1;
        bytes += match ty {
            "atomic<u32>;" => 4,
            "array<u32, MAX_LOCAL_LIGHTS>;" | "array<f32, MAX_LOCAL_LIGHTS>;" => {
                4 * MAX_LOCAL_LIGHTS as u32
            }
            _ => panic!("unaccounted workgroup type: {ty}"),
        };
    }
    assert_eq!(bytes, 3 * 4 * MAX_LOCAL_LIGHTS as u32 + 3 * 4);
    let limits = wgpu::Limits::default();
    assert!(bytes <= limits.max_compute_workgroup_storage_size);
    assert!(MAX_LOCAL_LIGHTS as u32 <= limits.max_compute_invocations_per_workgroup);
    assert!(TILE_SIZE <= limits.max_compute_workgroup_size_x);
    assert!(TILE_SIZE <= limits.max_compute_workgroup_size_y);
    assert!(TILE_WORDS <= limits.max_texture_array_layers);
}

#[test]
fn fixed_network_ranks_all_candidates_before_emitting() {
    // Structural shader checks only; numeric ranking expectations stay in the
    // tile_selection module's text fixtures, never in a CPU/GPU comparison.
    let network = LIGHT_TILES_WGSL.split("// Exact 256-wide").nth(1).unwrap();
    let (sort, emit) = network.split_once("let qualified =").unwrap();
    for required in [
        "width = 2u; width <= MAX_LOCAL_LIGHTS; width *= 2u",
        "stride = width / 2u; stride > 0u; stride /= 2u",
        "let partner = lane ^ stride",
        "if partner > lane",
        "(lane & width) == 0u",
        "select(tile_precedes(lane, partner), tile_precedes(partner, lane), ascending)",
        "workgroupBarrier();",
    ] {
        assert!(sort.contains(required), "{required}");
    }
    assert!(!sort.contains("TILE_MAX_LIGHTS"));
    assert!(!sort.contains("break") && !sort.contains("return"));
    for name in ["eligible", "scores", "indices"] {
        assert!(sort.contains(&format!(
            "candidate_{name}[lane] = candidate_{name}[partner]"
        )));
        assert!(sort.contains(&format!("candidate_{name}[partner] =")));
    }
    let compare = LIGHT_TILES_WGSL.split("fn tile_precedes").nth(1).unwrap();
    let compare = compare.split("/// Handoff:").next().unwrap();
    let eligibility = compare
        .find("return candidate_eligible[a] > candidate_eligible[b]")
        .unwrap();
    let score = compare
        .find("return candidate_scores[a] > candidate_scores[b]")
        .unwrap();
    let tie = compare
        .find("return candidate_indices[a] < candidate_indices[b]")
        .unwrap();
    assert!(eligibility < score && score < tie);
    for required in [
        "atomicLoad(&qualifying_count)",
        "let retained = min(qualified, TILE_MAX_LIGHTS)",
        "if lane == 0u",
        "atomicAdd(&droppedLightReferences, qualified - retained)",
        "if lane < TILE_MAX_LIGHTS",
        "select(0u, candidate_indices[lane], lane < retained)",
    ] {
        assert!(emit.contains(required), "{required}");
    }
}

#[test]
fn s12_storage_texture_and_compute_counter_contract() {
    for required in [
        "@group(2) @binding(1) var<uniform> tile_lights: array<LightGpu, MAX_LOCAL_LIGHTS>;",
        "@group(2) @binding(2) var tile_words: texture_storage_2d_array<r32uint, write>;",
        "@group(2) @binding(3) var<storage, read_write> droppedLightReferences: atomic<u32>;",
        "textureStore(tile_words, vec2<i32>(group.xy), 0, vec4<u32>(retained, 0u, 0u, 0u))",
        "textureStore(tile_words, vec2<i32>(group.xy), i32(lane + 1u), vec4<u32>(index, 0u, 0u, 0u))",
        "atomicAdd(&qualifying_count, 1u)",
    ] {
        assert!(LIGHT_TILES_WGSL.contains(required), "{required}");
    }
    assert_eq!(LIGHT_TILES_WGSL.matches("var<storage,").count(), 1);
    assert!(!raa_shader::PRELUDE_WGSL.contains("droppedLightReferences"));
    assert_eq!(LIGHT_TILES_WGSL.matches("textureStore(").count(), 2);
    // Composer validates the concrete storage texture types and every entry point.
    Composer::validate(&light_tiles_source()).unwrap();
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

#[cfg(not(target_arch = "wasm32"))]
mod install {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    use raa_core::testing::{block_on, skip, software_gpu};
    use raa_core::{FrameGraph, OffscreenTarget, RaaError, wgpu};
    use raa_lit::{LitOptions, LitRenderer, LocalLight, install_local_lights};
    use raa_pipeline::{FeaturePass, PassRole, Pipeline, RecordCtx};
    use raa_scene::{RenderList, Scene};

    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    const SIZE: (u32, u32) = (32, 16);

    /// An optional S12 pass the order-35 tiling pass must precede.
    struct AoGtao;

    impl FeaturePass for AoGtao {
        fn name(&self) -> &str {
            "ao.gtao"
        }

        fn order(&self) -> u32 {
            40
        }

        fn record(&mut self, _ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
            Ok(())
        }
    }

    fn installed(test: &str) -> Option<(raa_core::GpuContext, Rc<RefCell<Pipeline>>, LitRenderer)> {
        let Some(gpu) = software_gpu() else {
            skip(test, "software_gpu returned None");
            return None;
        };
        let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, FORMAT, false)));
        let mut lit = LitRenderer::new(
            &gpu,
            &pipeline,
            LitOptions {
                msaa: 1,
                ..LitOptions::default()
            },
        )
        .expect("lit renderer");
        pipeline
            .borrow_mut()
            .add(Box::new(AoGtao))
            .expect("ao.gtao");
        install_local_lights(&gpu, &pipeline, &mut lit).expect("install tiling");
        Some((gpu, pipeline, lit))
    }

    fn names(pipeline: &Rc<RefCell<Pipeline>>) -> Vec<String> {
        pipeline
            .borrow()
            .passes()
            .into_iter()
            .map(|p| p.name)
            .collect()
    }

    #[test]
    fn install_registers_order_35_and_refuses_duplicates_atomically() {
        let test = "install_registers_order_35_and_refuses_duplicates_atomically";
        let Some((gpu, pipeline, mut lit)) = installed(test) else {
            return;
        };
        let rows: Vec<_> = pipeline
            .borrow()
            .passes()
            .into_iter()
            .filter(|p| p.role == PassRole::Pass)
            .map(|p| (p.name, p.order))
            .collect();
        assert_eq!(
            rows,
            [
                ("prepass".to_string(), 30),
                ("light.tiles".to_string(), 35),
                ("ao.gtao".to_string(), 40),
                ("forward".to_string(), 60),
                ("editor".to_string(), 210),
            ]
        );
        let before = names(&pipeline);
        let err = install_local_lights(&gpu, &pipeline, &mut lit).expect_err("duplicate refused");
        assert_eq!(err.code(), "pipeline.duplicate_pass");
        assert_eq!(names(&pipeline), before, "installation adds nothing twice");

        let handle = lit
            .local_lights_mut()
            .add(LocalLight::default())
            .expect("light");
        let stats = lit.light_stats();
        assert_eq!(
            (stats.active_lights, stats.tile_capacity),
            (1, TILE_MAX_LIGHTS)
        );
        pipeline
            .borrow_mut()
            .set_enabled("light.tiles", false)
            .expect("disabled");
        assert_eq!(
            lit.local_lights().active_count(),
            1,
            "authored lights survive"
        );
        assert!(lit.local_lights().get(handle).is_ok());
        assert!(lit.light_stats().dropped_light_references.is_none());
    }

    /// One prepared-and-submitted frame of an empty scene; returns a validation error.
    fn frame(
        gpu: &raa_core::GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
        lit: &mut LitRenderer,
    ) -> Option<wgpu::Error> {
        let mut scene = Scene::new();
        scene.camera.set_viewport(SIZE.0, SIZE.1);
        let mut list = RenderList::with_capacity(1);
        let (mut tags, mut slots) = (Vec::new(), Vec::new());
        lit.prepare(gpu, &mut scene, &mut list, &mut tags, &mut slots)
            .expect("prepare");
        let target = OffscreenTarget::new(gpu, SIZE.0, SIZE.1, FORMAT);
        let mut graph = FrameGraph::new(FORMAT);
        Pipeline::install(pipeline, &mut graph);
        graph.compile().expect("compile");
        let scope = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let commands = graph
            .execute(gpu, target.view(), target.size())
            .expect("execute");
        gpu.queue().submit([commands]);
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
        block_on(scope.pop())
    }

    #[test]
    fn disabled_tiling_renders_with_valid_dummies_and_zero_counts() {
        let test = "disabled_tiling_renders_with_valid_dummies_and_zero_counts";
        let Some((gpu, pipeline, mut lit)) = installed(test) else {
            return;
        };
        lit.local_lights_mut()
            .add(LocalLight::default())
            .expect("light");
        pipeline
            .borrow_mut()
            .set_enabled("light.tiles", false)
            .expect("disabled");
        let error = frame(&gpu, &pipeline, &mut lit);
        assert!(error.is_none(), "GPU validation error: {error:?}");
        let stats = lit.light_stats();
        assert_eq!(stats.active_lights, 1);
        assert!(
            stats.stats_frame_index.is_none(),
            "no result while disabled"
        );
        assert!(!stats.pending);
    }

    #[test]
    fn light_stats_publish_the_latest_completed_invocation() {
        let test = "light_stats_publish_the_latest_completed_invocation";
        let Some((gpu, pipeline, mut lit)) = installed(test) else {
            return;
        };
        lit.local_lights_mut()
            .add(LocalLight::default())
            .expect("light");
        assert!(frame(&gpu, &pipeline, &mut lit).is_none());
        assert!(
            lit.light_stats().stats_frame_index.is_none(),
            "the first invocation is still pending"
        );
        // The pipeline creates its targets after the first view prepare, so the first
        // frame's published grid is the fallback; the settled grid follows it.
        for _ in 0..2 {
            assert!(frame(&gpu, &pipeline, &mut lit).is_none());
        }
        let stats = lit.light_stats();
        assert_eq!(stats.stats_frame_index, Some(1));
        assert_eq!(stats.dropped_light_references, Some(0));
        assert_eq!(
            (stats.tile_columns, stats.tile_rows, stats.tile_capacity),
            (2, 1, TILE_MAX_LIGHTS)
        );
        assert_eq!((stats.active_lights, stats.attached_lights), (1, 0));
    }
}
