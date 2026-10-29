//! m20 player tint (S23): the registry exports setMaterialTintMode and clearInstanceTint,
//! both tint modes compose valid WGSL, `graph` has no tint colour multiply, and the
//! `tint.mode_unknown` / `tint.material_range` refusals, and the tint flag of
//! setInstanceTint / clearInstanceTint.

use std::borrow::Cow;
use std::fs;
use std::path::Path;

use raa_lit::InstanceExtra;
use raa_shader::{
    ComposeOptions, Composer, MaterialFnId, MaterialFnRegistry, PassKind, SlotId, SlotProvider,
    TintMode, VariantKey,
};

#[allow(dead_code)]
#[path = "../src/api/m20_player_tint.rs"]
mod m20;

fn out_file(name: &str) -> String {
    let path = Path::new(env!("OUT_DIR")).join(name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[test]
fn registry_exports_set_material_tint_mode_and_clear_instance_tint() {
    let reg = out_file("api_registry.rs");
    assert!(reg.contains("mod m20_player_tint;"));
    assert!(reg.contains("m20_player_tint::install(ctx)?;"));
    assert!(reg.contains("m20_player_tint::frame(st, gpu, input)?;"));
    let src = include_str!("../src/api/m20_player_tint.rs");
    for name in ["setMaterialTintMode", "clearInstanceTint"] {
        assert!(src.contains(&format!("js_name = {name}")), "no {name}");
    }
}

/// Minimal valid body of each slot function (S11 signatures).
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

fn stub_composer() -> Composer {
    let mut composer = Composer::new();
    for slot in SlotId::ALL {
        let p = SlotProvider {
            slot,
            priority: 0,
            provider: "stub",
            wgsl: Cow::Borrowed(stub_wgsl(slot)),
        };
        composer.add_slot(p).expect("add");
    }
    composer
}

/// A material graph holding the S23 `input.player_color` node.
const PLAYER_COLOR_FN: &str = "fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let player = select(vec3<f32>(0.5), mat_instance_tint.rgb, mat_instance_tint_set > 0.5);
    o.base_color = player;
    return o;
}
";

fn key(material_fn: MaterialFnId, pass: PassKind) -> VariantKey {
    VariantKey {
        material_fn,
        pass,
        msaa: 4,
        flags: 0,
    }
}

fn mode(name: &str) -> ComposeOptions {
    ComposeOptions {
        tint_mode: m20::tint_mode(name).expect("mode"),
    }
}

#[test]
fn shaders_composed_in_both_modes_validate() {
    let composer = stub_composer();
    let mut reg = MaterialFnRegistry::new();
    let player = reg
        .register_user("player_color", PLAYER_COLOR_FN, &composer)
        .expect("register");
    for material_fn in [MaterialFnId::STANDARD, player] {
        for pass in PassKind::ALL {
            for name in ["multiply", "graph"] {
                let wgsl = composer
                    .compose_with(&reg, &key(material_fn, pass), &mode(name))
                    .expect("compose");
                if let Err(e) = Composer::validate(&wgsl) {
                    panic!("{material_fn:?} {pass:?} {name}: {e}");
                }
            }
        }
    }
}

#[test]
fn graph_mode_has_no_tint_colour_multiply() {
    let composer = stub_composer();
    let mut reg = MaterialFnRegistry::new();
    let player = reg
        .register_user("player_color", PLAYER_COLOR_FN, &composer)
        .expect("register");
    for pass in PassKind::ALL {
        let k = key(player, pass);
        let today = composer.compose(&reg, &k).expect("compose");
        let multiply = composer
            .compose_with(&reg, &k, &mode("multiply"))
            .expect("multiply");
        let graph = composer
            .compose_with(&reg, &k, &mode("graph"))
            .expect("graph");
        assert_eq!(multiply, today, "{pass:?}: multiply is today's shader");
        if pass == PassKind::Shadow {
            continue;
        }
        assert!(
            multiply.contains("raa_apply_tint(material_main(in), x)"),
            "{pass:?}"
        );
        assert!(
            !graph.contains("raa_apply_tint(material_main(in)"),
            "{pass:?}"
        );
        // The opacity multiply stays; the graph reads the tint itself.
        assert!(
            graph.contains("raa_apply_tint_opacity(material_main(in), x)"),
            "{pass:?}"
        );
        assert!(graph.contains("raa_set_instance_tint(x);"), "{pass:?}");
        assert!(graph.contains("mat_instance_tint.rgb"), "{pass:?}");
    }
}

#[test]
fn unknown_mode_and_bad_handle_codes() {
    assert_eq!(m20::tint_mode("multiply").unwrap(), TintMode::Multiply);
    assert_eq!(m20::tint_mode("graph").unwrap(), TintMode::Graph);
    for bad in ["", "add", "Graph", "MULTIPLY", "graph "] {
        assert_eq!(
            m20::tint_mode(bad).unwrap_err().code(),
            "tint.mode_unknown",
            "{bad:?}"
        );
    }
    assert_eq!(m20::material_range(0).code(), "tint.material_range");
    assert_eq!(m20::material_range(u32::MAX).code(), "tint.material_range");
}

#[test]
fn clear_resets_the_tint_and_its_flag_only() {
    let extra = InstanceExtra {
        tint: [0.2, 0.3, 0.4, 0.5],
        squash: [0.0, 1.0, 0.0, 0.3],
        jiggle: [1.0, 2.0, 3.0, 4.0],
        params: [[5.0; 4]; 4],
        tint_set: 1,
    };
    let cleared = m20::cleared(extra);
    assert_eq!(cleared.tint, [1.0; 4]);
    assert_eq!(cleared.tint_set, 0);
    assert_eq!(
        (cleared.tint, cleared.tint_set),
        (InstanceExtra::default().tint, InstanceExtra::default().tint_set)
    );
    assert_eq!(
        InstanceExtra {
            tint: extra.tint,
            tint_set: extra.tint_set,
            ..cleared
        },
        extra
    );
    // The cleared record writes reserved.x = 0, the shader's mat_instance_tint_set.
    let mut bytes = Vec::new();
    cleared.write_gpu(&mut bytes);
    assert_eq!(bytes[112..116], 0f32.to_le_bytes());
    let mut set = Vec::new();
    extra.write_gpu(&mut set);
    assert_eq!(set[112..116], 1f32.to_le_bytes());
}

#[test]
fn set_instance_tint_sets_the_flag() {
    let src = include_str!("../src/api/m15_animation.rs");
    let at = src.find("js_name = setInstanceTint").expect("setInstanceTint");
    let body = &src[at..at + src[at..].find("js_name = setInstanceParams").unwrap()];
    assert!(body.contains("extra.tint_set = 1;"));
    let m20 = include_str!("../src/api/m20_player_tint.rs");
    assert!(m20.contains("st.lit.handle().set_material_tint_mode(id, mode)"));
}
