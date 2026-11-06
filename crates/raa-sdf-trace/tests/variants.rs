//! Section 67 WGSL evaluator: complete shader variants generated through the public
//! public generation interfaces parse and type-check with naga; no GPU rendering is done here.
use raa_sdf_trace::wgsl::{PROGRAM_EVAL_WGSL, composite_module, eval_module, trace_module};
use raa_sdf_trace::{
    COMPOSITE_NAME, COMPOSITE_ORDER, Instance, Source, TRACE_NAME, TRACE_ORDER, TraceProgram,
};

fn program(sources: usize, code_len: u32) -> TraceProgram {
    let source = Source {
        code_start: 0,
        code_len,
        data_start: 0,
        data_len: 18,
        min: [-1.0; 3],
        max: [1.0; 3],
        lipschitz: 1.0,
        flags: 0,
    };
    TraceProgram {
        sources: vec![source; sources],
        code: vec![1; code_len as usize],
        data: vec![0.0; 18],
        materials: Vec::new(),
        instances: (0..sources)
            .map(|index| Instance {
                world_from_local: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                local_from_world: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                dist_scale: 1.0,
                source: index as u32,
                material: -1,
                key: index as u32,
                deform_kind: u32::from(index % 2 == 1),
                bend_angle: 0.5,
                bend_length: 2.0,
                axes: 56,
                bend: (index % 2 == 1).then_some([0.0, 30.0, 90.0, -1.0, 1.0, 0.0, 1.0, 0.0]),
            })
            .collect(),
    }
}

fn check(program: &TraceProgram) {
    for source in [
        eval_module(program),
        trace_module(program),
        composite_module(),
    ] {
        validate(&source);
    }
}

fn validate(source: &str) -> naga::Module {
    let module =
        naga::front::wgsl::parse_str(source).unwrap_or_else(|e| panic!("parse: {}", e.message()));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("validate: {e:?}"));
    module
}

#[test]
fn generated_variants_parse_and_validate() {
    for (sources, code_len) in [(0, 0), (1, 2), (2, 7), (4, 64)] {
        check(&program(sources, code_len));
    }
}

#[test]
fn shared_evaluator_is_a_composable_snippet() {
    for forbidden in ["@compute", "@vertex", "@fragment"] {
        assert!(!PROGRAM_EVAL_WGSL.contains(forbidden), "{forbidden}");
    }
    for binding in [
        "@binding(0) var<storage, read> trace_code",
        "@binding(1) var<storage, read> trace_data",
    ] {
        assert!(PROGRAM_EVAL_WGSL.contains(binding), "{binding}");
    }
    for name in [
        "trace_map",
        "trace_primitive",
        "trace_combine",
        "trace_eval",
        "trace_rest_point",
        "trace_instance",
        "trace_material",
    ] {
        assert!(PROGRAM_EVAL_WGSL.contains(name), "{name}");
    }
    let module = eval_module(&program(1, 2));
    assert!(
        module.contains("fn sdf_eval"),
        "the generated module has its entry point"
    );
}

#[test]
fn trace_and_composite_entry_points_and_layouts() {
    let compute = validate(&trace_module(&program(1, 2)));
    assert_eq!(compute.entry_points.len(), 1);
    assert_eq!(compute.entry_points[0].workgroup_size, [8, 8, 1]);
    let composite = validate(&composite_module());
    assert_eq!(composite.entry_points.len(), 2);
    assert_eq!(composite.entry_points[0].stage, naga::ShaderStage::Vertex);
    assert_eq!(composite.entry_points[1].stage, naga::ShaderStage::Fragment);
    for (name, size) in [
        ("TraceSource", 48),
        ("TraceInstance", 128),
        ("TraceBend", 32),
        ("TraceNode", 32),
        ("TraceMaterial", 32),
        ("TraceFrame", 512),
    ] {
        let ty = compute
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some(name))
            .unwrap()
            .1;
        assert!(
            matches!(ty.inner, naga::TypeInner::Struct { span, .. } if span == size),
            "{name}"
        );
    }
}

fn bindings(module: &naga::Module) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = module
        .global_variables
        .iter()
        .filter_map(|(_, global)| {
            global
                .binding
                .map(|binding| (binding.group, binding.binding))
        })
        .collect();
    out.sort_unstable();
    out
}

fn storage_format(module: &naga::Module, group: u32, binding: u32) -> Option<naga::StorageFormat> {
    let global = module
        .global_variables
        .iter()
        .map(|(_, global)| global)
        .find(|global| {
            global
                .binding
                .is_some_and(|b| b.group == group && b.binding == binding)
        })?;
    match module.types[global.ty].inner {
        naga::TypeInner::Image {
            class: naga::ImageClass::Storage { format, .. },
            ..
        } => Some(format),
        _ => None,
    }
}

#[test]
fn installed_shader_layouts_match_the_frame_passes() {
    assert_eq!((TRACE_NAME, TRACE_ORDER), ("trace.sdf", 55));
    assert_eq!((COMPOSITE_NAME, COMPOSITE_ORDER), ("trace.composite", 560));
    let compute = validate(&trace_module(&program(1, 2)));
    assert_eq!(
        bindings(&compute),
        (0..=10).map(|binding| (0, binding)).collect::<Vec<_>>()
    );
    assert_eq!(
        storage_format(&compute, 0, 9),
        Some(naga::StorageFormat::Rgba16Float)
    );
    assert_eq!(
        storage_format(&compute, 0, 10),
        Some(naga::StorageFormat::R32Float)
    );
    let composite = validate(&composite_module());
    assert_eq!(bindings(&composite), vec![(0, 0), (1, 0), (1, 1)]);
}
