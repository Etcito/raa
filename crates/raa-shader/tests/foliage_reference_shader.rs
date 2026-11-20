use std::borrow::Cow;

use raa_shader::{
    Composer, MaterialFnId, MaterialFnRegistry, PassKind, SlotId, SlotProvider, VariantKey,
};

const VERTEX_MATERIAL: &str = "fn material_main(in: MaterialInput) -> MaterialOutput { return raa_default_output(in); }
fn material_vertex(in: VertexMaterialInput) -> VertexMaterialOutput { return VertexMaterialOutput(in.local_position, in.normal); }
";
const BROKEN_MATERIAL: &str = "fn material_main(in: MaterialInput) -> MaterialOutput {
    let x: f64 = 1.0lf;
    return raa_default_output(in);
}
";
const WRONG_SIGNATURE: &str = "fn material_main(x: f32) -> f32 { return x; }\n";

fn composer() -> Composer {
    let sources = [
        "fn raa_slot_deform(p: vec3<f32>, n: vec3<f32>, x: InstanceExtraGpu, t: f32) -> VertexMaterialOutput { return VertexMaterialOutput(p, n); }",
        "fn raa_slot_shadow(p: vec3<f32>, n: vec3<f32>, f: vec4<f32>) -> f32 { return 1.0; }",
        "fn raa_slot_ao(f: vec4<f32>) -> f32 { return 1.0; }",
        "fn raa_slot_ambient(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> { return vec3<f32>(0.0); }",
        "fn raa_slot_direct(s: MaterialOutput, c: ShadeCtx, l: vec3<f32>, r: vec3<f32>) -> vec3<f32> { return vec3<f32>(0.0); }",
        "fn raa_slot_transmission(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> { return vec3<f32>(0.0); }",
        "fn raa_slot_fog(color: vec3<f32>, p: vec3<f32>, c: ShadeCtx) -> vec3<f32> { return color; }",
        "fn raa_slot_local_visibility(i: u32, p: vec3<f32>, n: vec3<f32>, d: vec3<f32>, distance: f32) -> f32 { return 1.0; }",
        "fn raa_slot_indirect(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> { return vec3<f32>(0.0); }",
    ];
    let mut composer = Composer::new();
    for (slot, source) in SlotId::ALL.into_iter().zip(sources) {
        composer
            .add_slot(SlotProvider {
                slot,
                priority: 0,
                provider: "reference-test",
                wgsl: Cow::Borrowed(source),
            })
            .unwrap();
    }
    composer
}

fn key(id: MaterialFnId) -> VariantKey {
    VariantKey {
        material_fn: id,
        pass: PassKind::Prepass,
        msaa: 1,
        flags: 0,
    }
}

fn bindings(module: &naga::Module) -> Vec<(u32, u32)> {
    module
        .global_variables
        .iter()
        .filter_map(|(_, v)| v.binding.as_ref().map(|b| (b.group, b.binding)))
        .collect()
}

#[test]
fn reference_composes_material_targets_and_reuses_the_vertex_path() {
    let composer = composer();
    let mut registry = MaterialFnRegistry::new();
    let vertex = registry
        .register_builtin("reference.vertex", VERTEX_MATERIAL)
        .unwrap();
    for material in [MaterialFnId::STANDARD, vertex] {
        let source = composer
            .compose_foliage_reference(&registry, &key(material))
            .unwrap();
        Composer::validate(&source).unwrap();
        let module = naga::front::wgsl::parse_str(&source).unwrap();
        assert_eq!(module.entry_points.len(), 2);
        assert!(
            module
                .entry_points
                .iter()
                .any(|e| e.name == "vs_main" && e.stage == naga::ShaderStage::Vertex)
        );
        let fragment = module
            .entry_points
            .iter()
            .find(|e| e.name == "fs_main" && e.stage == naga::ShaderStage::Fragment)
            .unwrap();
        let result = fragment.function.result.as_ref().unwrap();
        let naga::TypeInner::Struct { members, .. } = &module.types[result.ty].inner else {
            panic!("expected the reference targets");
        };
        let mut locations = Vec::new();
        let mut depth = false;
        for member in members {
            match &member.binding {
                Some(naga::Binding::Location { location, .. }) => {
                    locations.push(*location);
                    assert_eq!(
                        module.types[member.ty].inner,
                        naga::TypeInner::Vector {
                            size: naga::VectorSize::Quad,
                            scalar: naga::Scalar::F32,
                        },
                        "location {location} is an RGBA float target"
                    );
                }
                Some(naga::Binding::BuiltIn(naga::BuiltIn::FragDepth)) => depth = true,
                other => panic!("unexpected binding {other:?}"),
            }
        }
        assert_eq!(locations, [0, 1, 2]);
        assert!(depth, "the standard-Z depth is also written as frag_depth");
        assert_eq!(
            bindings(&module),
            bindings(
                &naga::front::wgsl::parse_str(
                    &composer.compose(&registry, &key(material)).unwrap()
                )
                .unwrap()
            ),
            "the reference pass adds no binding"
        );
    }
    let reference = composer
        .compose_foliage_reference(&registry, &key(MaterialFnId::STANDARD))
        .unwrap();
    assert!(
        reference.contains("surface.opacity"),
        "coverage is material opacity"
    );
    assert!(reference.contains("surface.base_color"), "albedo is unlit");
    assert!(!reference.contains("raa_pass_unlit"));
    assert!(!reference.contains("RaaBakeVaryings"));
    assert_ne!(
        reference,
        composer
            .compose_billboard_bake(&registry, &key(MaterialFnId::STANDARD))
            .unwrap()
    );
}

#[test]
fn reference_refuses_malformed_materials() {
    let registry = MaterialFnRegistry::new();
    let missing = Composer::new()
        .compose_foliage_reference(&registry, &key(MaterialFnId::STANDARD))
        .unwrap_err();
    assert_eq!(missing.code(), "shader.slot_missing");
    let unknown = composer()
        .compose_foliage_reference(&registry, &key(MaterialFnId(999)))
        .unwrap_err();
    assert_eq!(unknown.code(), "shader.unknown_fn");

    let mut registry = MaterialFnRegistry::new();
    let refused = registry
        .register_user("reference.broken", WRONG_SIGNATURE, &composer())
        .unwrap_err();
    assert_eq!(refused.code(), "material.fn_invalid");
    assert_eq!(registry.len(), 1, "the refused function is not registered");

    let mut registry = MaterialFnRegistry::new();
    let broken = registry
        .register_builtin("reference.broken", BROKEN_MATERIAL)
        .unwrap();
    let source = composer()
        .compose_foliage_reference(&registry, &key(broken))
        .unwrap();
    let error = Composer::validate(&source).unwrap_err();
    assert_eq!(
        error.code(),
        "shader.validate",
        "the reference pass rejects it"
    );
}
