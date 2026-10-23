//! Q12: user material functions (S11 rules). Each rule is refused with
//! `material.fn_invalid` and a reason naming it, a valid function is accepted under the next
//! id, `material_vertex` sets `has_vertex`, and a registered name is refused with
//! `material.fn_duplicate`.

use raa_shader::{Composer, MaterialFnId, MaterialFnRegistry, PassKind, ShaderError, VariantKey};

/// One default provider per slot (priority 0, the S11 defaults), as raa-lit installs them.
/// Kept here so this test does not depend on tests/variants.rs.
mod default_slots {
    use std::borrow::Cow;

    use raa_shader::{SlotId, SlotProvider};

    pub const PROVIDER: &str = "default";

    /// Identity deform.
    pub const DEFORM: &str = "fn raa_slot_deform(p: vec3<f32>, n: vec3<f32>, x: InstanceExtraGpu, t: f32) -> VertexMaterialOutput {
    return VertexMaterialOutput(p, n);
}
";

    /// Sun fully visible.
    pub const SHADOW: &str =
        "fn raa_slot_shadow(world_pos: vec3<f32>, n: vec3<f32>, frag: vec4<f32>) -> f32 {
    return 1.0;
}
";

    /// No screen AO.
    pub const AO: &str = "fn raa_slot_ao(frag: vec4<f32>) -> f32 {
    return 1.0;
}
";

    /// S1 PbrLite ambient term.
    pub const AMBIENT: &str = "fn raa_slot_ambient(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {
    return raa_frame.ambient.rgb * s.base_color * (1.0 - 0.5 * s.metallic);
}
";

    /// S1 Lambert + GGX specular (Schlick Fresnel from c.f0, Smith-Schlick G).
    pub const DIRECT: &str = "fn raa_slot_direct(s: MaterialOutput, c: ShadeCtx, l: vec3<f32>, radiance: vec3<f32>) -> vec3<f32> {
    let n = s.normal;
    let v = c.view_dir;
    let h = normalize(l + v);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = c.n_dot_v;
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);
    let alpha = s.roughness * s.roughness;
    let alpha2 = alpha * alpha;
    let d_den = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
    let d = alpha2 / max(3.14159265 * d_den * d_den, 1e-6);
    let fresnel = c.f0 + (vec3<f32>(1.0) - c.f0) * pow(1.0 - v_dot_h, 5.0);
    let k = (s.roughness + 1.0) * (s.roughness + 1.0) / 8.0;
    let g_v = n_dot_v / max(n_dot_v * (1.0 - k) + k, 1e-6);
    let g_l = n_dot_l / max(n_dot_l * (1.0 - k) + k, 1e-6);
    let specular = d * g_v * g_l * fresnel / max(4.0 * n_dot_v * n_dot_l, 1e-6);
    let diffuse = s.base_color / 3.14159265 * (1.0 - s.metallic);
    return (diffuse + specular) * radiance * n_dot_l;
}
";

    /// Nothing transmitted.
    pub const TRANSMISSION: &str =
        "fn raa_slot_transmission(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {
    return vec3<f32>(0.0);
}
";

    /// No fog.
    pub const FOG: &str =
        "fn raa_slot_fog(color: vec3<f32>, world_pos: vec3<f32>, c: ShadeCtx) -> vec3<f32> {
    return color;
}
";

    /// The default provider of `slot`.
    pub fn provider(slot: SlotId) -> SlotProvider {
        let wgsl = match slot {
            SlotId::Deform => DEFORM,
            SlotId::Shadow => SHADOW,
            SlotId::Ao => AO,
            SlotId::Ambient => AMBIENT,
            SlotId::Direct => DIRECT,
            SlotId::Transmission => TRANSMISSION,
            SlotId::Fog => FOG,
        };
        SlotProvider {
            slot,
            priority: 0,
            provider: PROVIDER,
            wgsl: Cow::Borrowed(wgsl),
        }
    }

    /// One default provider per slot, in [`SlotId::ALL`] order.
    pub fn all() -> Vec<SlotProvider> {
        SlotId::ALL.iter().map(|&slot| provider(slot)).collect()
    }
}

/// A valid user function: LIB noise driven by the record's first parameter vector, with a
/// private helper; the comment mentions every forbidden token and must not count.
const MARBLE: &str =
    "// Not code: @group(0) @binding(1) @vertex @fragment @compute var<storage> var<uniform>
/* var<workgroup> enable f16; override k: f32; fn raa_helper() {} */
fn marble_band(x: f32) -> f32 {
    return 0.5 + 0.5 * sin(x);
}

fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let scale = in.record.p[0].x;
    let n = raa_fbm3(in.position * scale, 4);
    o.base_color = mix(o.base_color, in.record.p[0].yzw, marble_band(n * 6.0));
    return o;
}
";

/// A valid user function that also displaces vertices.
const WAVE: &str = "fn material_main(in: MaterialInput) -> MaterialOutput {
    return raa_default_output(in);
}

fn material_vertex(in: VertexMaterialInput) -> VertexMaterialOutput {
    let s = sin(in.time + in.local_position.x * 3.0) * in.record.p[1].x;
    return VertexMaterialOutput(in.local_position + in.normal * s, in.normal);
}
";

fn default_composer() -> Composer {
    let mut composer = Composer::new();
    for p in default_slots::all() {
        if let Err(e) = composer.add_slot(p) {
            panic!("{e}");
        }
    }
    composer
}

fn register(reg: &mut MaterialFnRegistry, name: &str, wgsl: &str) -> MaterialFnId {
    match reg.register_user(name, wgsl, &default_composer()) {
        Ok(id) => id,
        Err(e) => panic!("{name}: {e}"),
    }
}

/// Registers `src` as a user function, expects `material.fn_invalid` with a reason naming
/// `rule` and containing every `detail`, and checks the registry is unchanged.
fn assert_refused(src: &str, rule: &str, details: &[&str]) -> ShaderError {
    let mut reg = MaterialFnRegistry::new();
    let e = match reg.register_user("user.bad", src, &default_composer()) {
        Ok(id) => panic!("{rule}: accepted as {id:?}"),
        Err(e) => e,
    };
    assert_eq!(e.code(), "material.fn_invalid", "{rule}: {e}");
    assert!(e.message.contains("`user.bad`"), "{rule}: {e}");
    assert!(
        e.message.contains(&format!("refused: {rule}")),
        "{rule}: {e}"
    );
    for detail in details {
        assert!(
            e.message.contains(detail),
            "{rule}: {detail:?} missing in {e}"
        );
    }
    assert_eq!(reg.len(), 1, "{rule}: registry changed");
    assert_eq!(reg.id_of("user.bad"), None, "{rule}");
    e
}

#[test]
fn valid_fn_is_accepted_with_next_id() {
    let mut reg = MaterialFnRegistry::new();
    let builtin = reg
        .register_builtin("raa.extra", "fn material_main(in: MaterialInput) -> MaterialOutput {\n    return raa_default_output(in);\n}\n")
        .expect("builtin");
    assert_eq!(builtin, MaterialFnId(1));
    let id = register(&mut reg, "user.marble", MARBLE);
    assert_eq!(id, MaterialFnId(2));
    assert_eq!(reg.len(), 3);
    assert_eq!(reg.id_of("user.marble"), Some(id));
    assert_eq!(reg.names(), ["raa.standard", "raa.extra", "user.marble"]);
    let entry = reg.get(id).expect("entry");
    assert_eq!(entry.name, "user.marble");
    assert_eq!(entry.wgsl, MARBLE);
    assert!(!entry.builtin);
    assert!(!entry.has_vertex);

    // The accepted function composes and validates for every pass, not only Forward.
    let composer = default_composer();
    for pass in PassKind::ALL {
        for msaa in [1, 4] {
            let key = VariantKey {
                material_fn: id,
                pass,
                msaa,
                flags: 0,
            };
            let wgsl = composer
                .compose(&reg, &key)
                .unwrap_or_else(|e| panic!("{key:?}: {e}"));
            if let Err(e) = Composer::validate(&wgsl) {
                panic!("{key:?}: {e}");
            }
        }
    }
}

#[test]
fn material_vertex_sets_has_vertex() {
    let mut reg = MaterialFnRegistry::new();
    let id = register(&mut reg, "user.wave", WAVE);
    assert_eq!(id, MaterialFnId(1));
    let entry = reg.get(id).expect("entry");
    assert!(entry.has_vertex);
    assert!(!entry.builtin);
}

#[test]
fn duplicate_name_is_refused() {
    let mut reg = MaterialFnRegistry::new();
    let id = register(&mut reg, "user.marble", MARBLE);
    for name in ["user.marble", "raa.standard"] {
        let e = reg
            .register_user(name, WAVE, &default_composer())
            .unwrap_err();
        assert_eq!(e.code(), "material.fn_duplicate", "{e}");
        assert!(e.message.contains(name), "{e}");
    }
    assert_eq!(reg.len(), 2);
    assert_eq!(reg.get(id).map(|e| e.wgsl.as_str()), Some(MARBLE));
    assert_eq!(register(&mut reg, "user.wave", WAVE), MaterialFnId(2));
}

/// The smallest valid source, extended by the refusal cases.
const MAIN: &str = "fn material_main(in: MaterialInput) -> MaterialOutput {
    return raa_default_output(in);
}
";

#[test]
fn missing_material_main_is_refused() {
    let src =
        "fn shade(in: MaterialInput) -> MaterialOutput {\n    return raa_default_output(in);\n}\n";
    assert_refused(src, "material_main signature", &["does not define"]);
    // A comment does not define it either.
    let commented = format!("/* {MAIN} */");
    assert_refused(&commented, "material_main signature", &["does not define"]);
}

#[test]
fn wrong_material_main_signature_is_refused() {
    for src in [
        "fn material_main(in: VertexMaterialInput) -> MaterialOutput {\n    var o: MaterialOutput;\n    return o;\n}\n",
        "fn material_main(in: MaterialInput) -> VertexMaterialOutput {\n    return VertexMaterialOutput(in.local_position, in.normal);\n}\n",
        "fn material_main(in: MaterialInput, k: f32) -> MaterialOutput {\n    return raa_default_output(in);\n}\n",
        "fn material_main() -> MaterialOutput {\n    var o: MaterialOutput;\n    return o;\n}\n",
        "fn material_main(in: MaterialInput) {\n}\n",
    ] {
        assert_refused(
            src,
            "material_main signature",
            &["fn material_main(in: MaterialInput) -> MaterialOutput"],
        );
    }
}

#[test]
fn forbidden_attributes_are_refused() {
    for (extra, attr) in [
        ("@group(3) var<private> g: f32;", "`@group`"),
        ("@binding(0) var<private> b: f32;", "`@binding`"),
        (
            "@vertex fn v() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }",
            "`@vertex`",
        ),
        (
            "@fragment fn f() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }",
            "`@fragment`",
        ),
        ("@compute @workgroup_size(1) fn c() {}", "`@compute`"),
    ] {
        let src = format!("{MAIN}{extra}\n");
        assert_refused(&src, "forbidden attribute", &[attr, " at 4:1"]);
    }
}

#[test]
fn forbidden_address_spaces_are_refused() {
    for (extra, space) in [
        ("var<storage, read> s: array<f32>;", "`var<storage`"),
        ("var <uniform> u: vec4<f32>;", "`var<uniform`"),
        ("var<workgroup> w: f32;", "`var<workgroup`"),
    ] {
        let src = format!("{MAIN}{extra}\n");
        assert_refused(&src, "forbidden address space", &[space, " at 4:1"]);
    }
}

#[test]
fn forbidden_keywords_are_refused() {
    let src = format!("enable f16;\n{MAIN}");
    assert_refused(&src, "forbidden keyword", &["`enable`", " at 1:1"]);
    let src = format!("{MAIN}override k: f32 = 1.0;\n");
    assert_refused(&src, "forbidden keyword", &["`override`", " at 4:1"]);
}

#[test]
fn raa_prefixed_definitions_are_refused() {
    for (extra, what) in [
        ("fn raa_helper() -> f32 { return 1.0; }", "fn `raa_helper`"),
        ("struct raa_Params { k: f32 }", "struct `raa_Params`"),
        ("const raa_k: f32 = 2.0;", "const `raa_k`"),
        ("var<private> raa_state: f32;", "var `raa_state`"),
        // Same name as a LIB helper: naga's redefinition is reported as this rule.
        (
            "fn raa_fbm3(p: vec3<f32>, octaves: i32) -> f32 { return 0.0; }",
            "`raa_fbm3`",
        ),
        // Same name as a prelude global.
        ("var<private> raa_frame: f32;", "`raa_frame`"),
    ] {
        let src = format!("{MAIN}{extra}\n");
        assert_refused(&src, "reserved prefix", &[what, " at 4:"]);
    }
}

#[test]
fn oversized_source_is_refused() {
    let mut src = MAIN.to_string();
    while src.len() <= 65536 {
        src.push_str("// padding padding padding padding padding padding padding padding\n");
    }
    assert_refused(
        &src,
        "size limit",
        &[&format!("{} bytes", src.len()), "65536"],
    );

    // Exactly at the limit is accepted.
    let mut src = MAIN.to_string();
    src.push_str(&"/".repeat(65536 - src.len()));
    assert_eq!(src.len(), 65536);
    let mut reg = MaterialFnRegistry::new();
    assert_eq!(register(&mut reg, "user.big", &src), MaterialFnId(1));
}

#[test]
fn parse_error_is_refused_with_line_col() {
    let src = "fn material_main(in: MaterialInput) -> MaterialOutput {\n    let o = raa_default_output(in)\n    return o;\n}\n";
    assert_refused(src, "parse", &["3:5: "]);
}

#[test]
fn composed_forward_variant_must_validate() {
    // Parses alone but a path ends without `return`: naga's validation message, with the
    // line:col of the composed module and of the user source.
    let src = "fn material_main(in: MaterialInput) -> MaterialOutput {\n    var o = raa_default_output(in);\n    if (in.time > 0.0) {\n        return o;\n    }\n}\n";
    let e = assert_refused(
        src,
        "composed Forward variant (msaa 4) does not validate",
        &["shader.validate", "material_main", "(user source 1:1)"],
    );
    let (_, rest) = e
        .message
        .split_once("shader.validate: ")
        .expect("naga message");
    let (line, rest) = rest.split_once(':').expect("line");
    let (col, _) = rest.split_once(": ").expect("col");
    assert!(
        line.parse::<u32>().is_ok() && col.parse::<u32>().is_ok(),
        "{e}"
    );

    // A derivative in material_vertex is forbidden in the vertex stage that runs it.
    let src = format!(
        "{MAIN}fn material_vertex(in: VertexMaterialInput) -> VertexMaterialOutput {{\n    let d = dpdx(in.time);\n    return VertexMaterialOutput(in.local_position * d, in.normal);\n}}\n"
    );
    assert_refused(
        &src,
        "composed Forward variant (msaa 4) does not validate",
        &["vs_main", "forbidden at this stage"],
    );

    // A material_vertex with the wrong signature breaks the composed vertex stage.
    let src =
        format!("{MAIN}fn material_vertex(p: vec3<f32>) -> vec3<f32> {{\n    return p;\n}}\n");
    assert_refused(
        &src,
        "composed Forward variant (msaa 4) does not validate",
        &["shader."],
    );
}

#[test]
fn missing_slot_provider_is_reported_unchanged() {
    let mut reg = MaterialFnRegistry::new();
    let e = reg
        .register_user("user.marble", MARBLE, &Composer::new())
        .unwrap_err();
    assert_eq!(e.code(), "shader.slot_missing", "{e}");
    assert_eq!(reg.len(), 1);
    assert_eq!(register(&mut reg, "user.marble", MARBLE), MaterialFnId(1));
}
