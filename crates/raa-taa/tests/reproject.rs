//! Contract `raa-temporal-wgsl`: the shared reprojection WGSL validates in a compute module,
//! its `RaaTemporalFrame` matches the raa-shader prelude `FrameGpu` layout, and each of its
//! four functions is declared exactly once.

use raa_shader::Composer;
use raa_taa::{REPROJECT_WGSL, TEMPORAL_FRAME_SIZE};

/// A compute module that binds the twin as a uniform and a storage buffer and calls every
/// helper: the shape the downstream temporal passes compose.
const PROBE_WGSL: &str = "
@group(0) @binding(0) var<uniform> f: RaaTemporalFrame;
@group(0) @binding(1) var<storage, read_write> probe_out: array<vec4<f32>>;

@compute @workgroup_size(1)
fn cs_reproject_probe(@builtin(global_invocation_id) id: vec3<u32>) {
    let uv = vec2<f32>(0.5, 0.5);
    let surface = raa_temporal_surface_uv(uv, vec4<f32>(0.0));
    let sky = raa_temporal_sky_uv(
        uv,
        f.inv_view,
        f.inv_proj,
        f.previous_view_proj,
        f.camera_jitter,
    );
    let on_screen = raa_temporal_on_screen(surface.xy);
    let usable = raa_temporal_history_usable(f.temporal);
    probe_out[id.x] = vec4<f32>(
        surface.z,
        sky.z,
        select(0.0, 1.0, on_screen),
        select(0.0, 1.0, usable),
    );
}
";

/// The `(name, byte offset)` members and span of struct `name` in `module`.
fn struct_layout(module: &naga::Module, name: &str) -> (u32, Vec<(String, u32)>) {
    let (_, ty) = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("{name} is declared"));
    let naga::TypeInner::Struct { members, span } = &ty.inner else {
        panic!("{name} is a struct");
    };
    let fields = members
        .iter()
        .map(|m| (m.name.clone().unwrap_or_default(), m.offset))
        .collect();
    (*span, fields)
}

#[test]
fn reproject_wgsl_validates_in_a_compute_module() {
    let source = format!("{REPROJECT_WGSL}\n{PROBE_WGSL}");
    assert_eq!(Composer::validate(&source), Ok(()));
}

#[test]
fn raa_temporal_frame_matches_the_prelude_frame_gpu() {
    let prelude = naga::front::wgsl::parse_str(raa_shader::PRELUDE_WGSL)
        .or_else(|_| {
            // The composer emits PRELUDE, LIB and CLOUD_SHADOW first; parse that prefix when
            // the prelude needs the later declarations.
            naga::front::wgsl::parse_str(&format!(
                "{}\n{}\n{}",
                raa_shader::PRELUDE_WGSL,
                raa_shader::LIB_WGSL,
                raa_shader::CLOUD_SHADOW_WGSL
            ))
        })
        .expect("the raa-shader prelude parses");
    let reproject = naga::front::wgsl::parse_str(REPROJECT_WGSL).expect("REPROJECT_WGSL parses");

    let (prelude_span, prelude_members) = struct_layout(&prelude, "FrameGpu");
    let (span, members) = struct_layout(&reproject, "RaaTemporalFrame");
    assert_eq!(members, prelude_members, "RaaTemporalFrame members");
    assert_eq!(span, prelude_span, "RaaTemporalFrame span");
    assert_eq!(span, 768);
    assert_eq!(u64::from(span), TEMPORAL_FRAME_SIZE);
}

#[test]
fn each_reprojection_function_is_declared_once() {
    for name in [
        "raa_temporal_on_screen",
        "raa_temporal_surface_uv",
        "raa_temporal_sky_uv",
        "raa_temporal_history_usable",
    ] {
        let declaration = format!("fn {name}(");
        assert_eq!(
            REPROJECT_WGSL.matches(&declaration).count(),
            1,
            "{declaration}"
        );
    }
}
