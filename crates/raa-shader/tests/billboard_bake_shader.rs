use std::borrow::Cow;

use raa_shader::{
    Composer, MaterialFnId, MaterialFnRegistry, PassKind, SlotId, SlotProvider, VariantKey,
};

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
                provider: "test",
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

#[test]
fn bake_composes_materials_and_public_interface() {
    let composer = composer();
    let mut registry = MaterialFnRegistry::new();
    let id = registry.register_builtin("vertex", "fn material_main(in: MaterialInput) -> MaterialOutput { return raa_default_output(in); }\nfn material_vertex(in: VertexMaterialInput) -> VertexMaterialOutput { return VertexMaterialOutput(in.local_position, in.normal); }").unwrap();
    for material in [MaterialFnId::STANDARD, id] {
        let source = composer
            .compose_billboard_bake(&registry, &key(material))
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
            panic!("expected targets");
        };
        let locations: Vec<_> = members
            .iter()
            .map(|m| match m.binding {
                Some(naga::Binding::Location { location, .. }) => location,
                _ => panic!("expected location"),
            })
            .collect();
        assert_eq!(locations, [0, 1, 2]);
        let bindings: Vec<_> = module
            .global_variables
            .iter()
            .filter_map(|(_, v)| v.binding.as_ref().map(|b| (b.group, b.binding)))
            .collect();
        let standard = composer.compose(&registry, &key(material)).unwrap();
        let standard_module = naga::front::wgsl::parse_str(&standard).unwrap();
        let standard_bindings: Vec<_> = standard_module
            .global_variables
            .iter()
            .filter_map(|(_, v)| v.binding.as_ref().map(|b| (b.group, b.binding)))
            .collect();
        assert_eq!(bindings, standard_bindings);
    }
}

#[test]
fn bake_reports_the_same_composition_errors() {
    let registry = MaterialFnRegistry::new();
    let missing = Composer::new()
        .compose_billboard_bake(&registry, &key(MaterialFnId::STANDARD))
        .unwrap_err();
    assert_eq!(missing.code(), "shader.slot_missing");
    let unknown = composer()
        .compose_billboard_bake(&registry, &key(MaterialFnId(999)))
        .unwrap_err();
    assert_eq!(unknown.code(), "shader.unknown_fn");
}
