//! Section 67 WGSL evaluator: complete shader variants generated through the public
//! `eval_module` interface parse and type-check with naga; the shared snippet stays composable.
use raa_sdf_trace::wgsl::{PROGRAM_EVAL_WGSL, eval_module};
use raa_sdf_trace::{Source, TraceProgram};

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
        instances: Vec::new(),
    }
}

fn check(program: &TraceProgram) {
    let source = eval_module(program);
    let module =
        naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| panic!("parse: {}", e.message()));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("validate: {e:?}"));
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
