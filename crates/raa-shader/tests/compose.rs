//! Composition rules (S11): section order, the passthrough `material_vertex`, slot
//! priorities and conflicts, `slots_hash`, and the `shader.slot_missing` and
//! `shader.unknown_fn` refusals; the standard function applies the vertex colour.

use std::borrow::Cow;

use raa_shader::{
    Composer, MaterialFnId, MaterialFnRegistry, PassKind, SlotId, SlotProvider, VariantKey,
};

/// Minimal valid body of each slot function (shapes of the S11 signatures).
fn stub_wgsl(slot: SlotId) -> &'static str {
    match slot {
        SlotId::Deform => "fn raa_slot_deform(p: vec3<f32>, n: vec3<f32>, x: InstanceExtraGpu, t: f32) -> VertexMaterialOutput {
    return VertexMaterialOutput(p, n);
}
",
        SlotId::Shadow => "fn raa_slot_shadow(world_pos: vec3<f32>, n: vec3<f32>, frag: vec4<f32>) -> f32 {
    return 1.0;
}
",
        SlotId::Ao => "fn raa_slot_ao(frag: vec4<f32>) -> f32 {
    return 1.0;
}
",
        SlotId::Ambient => "fn raa_slot_ambient(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {
    return raa_frame.ambient.rgb * s.base_color;
}
",
        SlotId::Direct => "fn raa_slot_direct(s: MaterialOutput, c: ShadeCtx, l: vec3<f32>, radiance: vec3<f32>) -> vec3<f32> {
    return s.base_color * radiance * max(dot(s.normal, l), 0.0);
}
",
        SlotId::Transmission => "fn raa_slot_transmission(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {
    return vec3<f32>(0.0);
}
",
        SlotId::Fog => "fn raa_slot_fog(color: vec3<f32>, world_pos: vec3<f32>, c: ShadeCtx) -> vec3<f32> {
    return color;
}
",
    }
}

fn stub(slot: SlotId, priority: i32, provider: &'static str) -> SlotProvider {
    SlotProvider {
        slot,
        priority,
        provider,
        wgsl: Cow::Borrowed(stub_wgsl(slot)),
    }
}

fn stub_composer() -> Composer {
    let mut composer = Composer::new();
    for slot in SlotId::ALL {
        composer.add_slot(stub(slot, 0, "stub")).expect("add");
    }
    composer
}

fn key(pass: PassKind) -> VariantKey {
    VariantKey {
        material_fn: MaterialFnId::STANDARD,
        pass,
        msaa: 4,
        flags: 0,
    }
}

fn position(wgsl: &str, needle: &str) -> usize {
    let found: Vec<usize> = wgsl.match_indices(needle).map(|(i, _)| i).collect();
    assert_eq!(found.len(), 1, "`{needle}` occurs {} times", found.len());
    found[0]
}

#[test]
fn sections_follow_contract_order() {
    let reg = MaterialFnRegistry::new();
    let wgsl = stub_composer()
        .compose(&reg, &key(PassKind::Forward))
        .expect("compose");
    assert_eq!(Composer::validate(&wgsl), Ok(()));
    let mut needles = vec![
        "// raa section: prelude",
        "struct FrameGpu {",
        "// raa section: lib",
        "fn raa_hash3(",
        "// raa section: material \"raa.standard\"",
        "fn material_main(",
        "// raa section: material_vertex passthrough",
        "fn material_vertex(",
    ];
    let slot_fns: Vec<String> = SlotId::ALL
        .iter()
        .map(|s| format!("fn {}(", s.fn_name()))
        .collect();
    needles.extend(slot_fns.iter().map(String::as_str));
    needles.extend([
        "// raa section: pass Forward",
        "const RAA_MSAA: u32 = 4u;",
        "fn vs_main(",
        "fn fs_main(",
    ]);
    let positions: Vec<usize> = needles.iter().map(|n| position(&wgsl, n)).collect();
    for (pair, names) in positions.windows(2).zip(needles.windows(2)) {
        assert!(
            pair[0] < pair[1],
            "`{}` must precede `{}`",
            names[0],
            names[1]
        );
    }
    // Nothing but the variant comment precedes the prelude.
    let first = position(&wgsl, "// raa section: prelude");
    assert!(
        wgsl[..first]
            .lines()
            .all(|l| l.starts_with("//") || l.is_empty())
    );
}

#[test]
fn each_pass_kind_composes_its_entry_points() {
    let reg = MaterialFnRegistry::new();
    let composer = stub_composer();
    for pass in PassKind::ALL {
        let wgsl = composer.compose(&reg, &key(pass)).expect("compose");
        assert!(
            wgsl.contains(&format!("// raa section: pass {pass:?}\n")),
            "{pass:?}"
        );
        assert!(wgsl.contains("@vertex\nfn vs_main("), "{pass:?}");
        let has_fs = wgsl.contains("@fragment\nfn fs_main(");
        assert_eq!(has_fs, pass != PassKind::Shadow, "{pass:?}");
        if let Err(e) = Composer::validate(&wgsl) {
            panic!("{pass:?}: {e}");
        }
    }
}

#[test]
fn material_vertex_of_the_function_replaces_the_passthrough() {
    let mut reg = MaterialFnRegistry::new();
    let src = "fn material_main(in: MaterialInput) -> MaterialOutput {
    return raa_default_output(in);
}
fn material_vertex(in: VertexMaterialInput) -> VertexMaterialOutput {
    return VertexMaterialOutput(in.local_position * 1.5, in.normal);
}
";
    let id = reg
        .register_builtin("raa.test_grow", src)
        .expect("register");
    let mut k = key(PassKind::Shadow);
    k.material_fn = id;
    let wgsl = stub_composer().compose(&reg, &k).expect("compose");
    position(&wgsl, "fn material_vertex(");
    position(&wgsl, "in.local_position * 1.5");
    assert!(!wgsl.contains("material_vertex passthrough"));
    assert_eq!(Composer::validate(&wgsl), Ok(()));
}

#[test]
fn equal_priority_on_one_slot_conflicts() {
    let mut composer = stub_composer();
    let e = composer
        .add_slot(stub(SlotId::Ao, 0, "other"))
        .expect_err("conflict");
    assert_eq!(e.code(), "shader.slot_conflict");
    assert!(e.message.contains("raa_slot_ao"), "{e}");
    assert!(
        e.message.contains("\"stub\"") && e.message.contains("\"other\""),
        "{e}"
    );
    assert_eq!(
        composer.winner(SlotId::Ao).map(|p| p.provider),
        Some("stub")
    );
    // Another priority on the slot, or the same priority on another slot, is accepted.
    composer
        .add_slot(stub(SlotId::Ao, 3, "other"))
        .expect("priority 3");
    let mut fresh = Composer::new();
    fresh.add_slot(stub(SlotId::Ao, 0, "a")).expect("ao");
    fresh.add_slot(stub(SlotId::Fog, 0, "a")).expect("fog");
}

#[test]
fn higher_priority_wins() {
    let reg = MaterialFnRegistry::new();
    let mut composer = stub_composer();
    let defaults_hash = composer.slots_hash();
    let tinted = "fn raa_slot_ambient(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {
    return raa_tinted_scale() * s.base_color;
}
fn raa_tinted_scale() -> vec3<f32> {
    return vec3<f32>(0.25, 0.5, 1.0);
}
";
    composer
        .add_slot(SlotProvider {
            slot: SlotId::Ambient,
            priority: 10,
            provider: "tinted",
            wgsl: Cow::Borrowed(tinted),
        })
        .expect("override");
    // A lower priority added later does not win either.
    composer
        .add_slot(stub(SlotId::Ambient, -5, "low"))
        .expect("low");
    let w = composer.winner(SlotId::Ambient).expect("winner");
    assert_eq!((w.provider, w.priority), ("tinted", 10));
    assert_eq!(
        composer.winner(SlotId::Fog).map(|p| p.provider),
        Some("stub")
    );

    let wgsl = composer
        .compose(&reg, &key(PassKind::Forward))
        .expect("compose");
    position(&wgsl, "fn raa_slot_ambient(");
    position(&wgsl, "fn raa_tinted_scale()");
    assert!(!wgsl.contains("raa_frame.ambient.rgb * s.base_color"));
    assert!(wgsl.contains("provider \"tinted\", priority 10"));
    assert_eq!(Composer::validate(&wgsl), Ok(()));
    assert_ne!(composer.slots_hash(), defaults_hash);
}

#[test]
fn slots_hash_depends_only_on_winners() {
    let mut a = stub_composer();
    a.add_slot(stub(SlotId::Fog, 7, "fog7")).expect("fog7");
    a.add_slot(stub(SlotId::Fog, 2, "fog2")).expect("fog2");
    // Same winners, other insertion order and no losing providers.
    let mut b = Composer::new();
    b.add_slot(stub(SlotId::Fog, 7, "fog7")).expect("fog7");
    for slot in SlotId::ALL.into_iter().rev().filter(|&s| s != SlotId::Fog) {
        b.add_slot(stub(slot, 0, "stub")).expect("add");
    }
    assert_eq!(a.slots_hash(), b.slots_hash());
    assert_eq!(a.slots_hash(), a.clone().slots_hash());
    assert_ne!(a.slots_hash(), stub_composer().slots_hash());
    assert_ne!(Composer::new().slots_hash(), stub_composer().slots_hash());
    // The provider name and the priority of a winner count.
    let mut renamed = Composer::new();
    let mut reprioritised = Composer::new();
    for slot in SlotId::ALL {
        renamed.add_slot(stub(slot, 0, "stub2")).expect("add");
        reprioritised.add_slot(stub(slot, 1, "stub")).expect("add");
    }
    assert_ne!(renamed.slots_hash(), stub_composer().slots_hash());
    assert_ne!(reprioritised.slots_hash(), stub_composer().slots_hash());
}

#[test]
fn missing_slot_is_refused() {
    let reg = MaterialFnRegistry::new();
    let e = Composer::new()
        .compose(&reg, &key(PassKind::Forward))
        .expect_err("no providers");
    assert_eq!(e.code(), "shader.slot_missing");
    assert!(e.message.contains("raa_slot_deform"), "{e}");

    let mut composer = Composer::new();
    for slot in SlotId::ALL.into_iter().filter(|&s| s != SlotId::Fog) {
        composer.add_slot(stub(slot, 0, "stub")).expect("add");
    }
    for pass in PassKind::ALL {
        let e = composer.compose(&reg, &key(pass)).expect_err("fog missing");
        assert_eq!(e.code(), "shader.slot_missing", "{pass:?}");
        assert!(e.message.contains("raa_slot_fog"), "{e}");
        assert!(!e.message.is_empty());
    }
}

#[test]
fn unknown_fn_is_refused() {
    let reg = MaterialFnRegistry::new();
    let mut k = key(PassKind::Forward);
    k.material_fn = MaterialFnId(7);
    for composer in [stub_composer(), Composer::new()] {
        let e = composer.compose(&reg, &k).expect_err("unknown");
        assert_eq!(e.code(), "shader.unknown_fn");
        assert!(e.message.contains('7'), "{e}");
    }
}

#[test]
fn variant_constants_follow_the_key() {
    let reg = MaterialFnRegistry::new();
    let composer = stub_composer();
    let mut k = key(PassKind::FurShell);
    k.msaa = 1;
    k.flags = 4;
    let one = composer.compose(&reg, &k).expect("msaa 1");
    assert!(one.contains("const RAA_MSAA: u32 = 1u;"));
    assert!(one.contains("const RAA_VARIANT_FLAGS: u32 = 4u;"));
    k.msaa = 4;
    let four = composer.compose(&reg, &k).expect("msaa 4");
    assert!(four.contains("const RAA_MSAA: u32 = 4u;"));
    assert_ne!(one, four);
    assert_eq!(composer.compose(&reg, &k).expect("again"), four);
}

#[test]
fn broken_provider_fails_validation() {
    let reg = MaterialFnRegistry::new();
    let mut composer = stub_composer();
    composer
        .add_slot(SlotProvider {
            slot: SlotId::Ao,
            priority: 1,
            provider: "broken",
            wgsl: Cow::Borrowed(
                "fn raa_slot_ao(frag: vec4<f32>) -> f32 {\n    return frag.xy;\n}\n",
            ),
        })
        .expect("add");
    let wgsl = composer
        .compose(&reg, &key(PassKind::Forward))
        .expect("compose");
    let e = Composer::validate(&wgsl).expect_err("type error");
    assert_eq!(e.code(), "shader.validate", "{e}");
}

#[test]
fn standard_function_multiplies_by_vertex_colour() {
    let reg = MaterialFnRegistry::new();
    let composer = stub_composer();
    for pass in PassKind::ALL {
        let wgsl = composer.compose(&reg, &key(pass)).expect("compose");
        position(&wgsl, "o.base_color = r.base.rgb * in.color.rgb;");
        position(&wgsl, "o.opacity = r.base.a * in.color.a;");
        if let Err(e) = Composer::validate(&wgsl) {
            panic!("{pass:?}: {e}");
        }
    }
}
