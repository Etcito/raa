//! The lookdev scene's user material functions compose and validate. The m13 mud set adds
//! `demo.grass` and `demo.crown`; a reserved WGSL keyword or a broken helper would otherwise
//! only fail at capture time, on the GPU card.

#[path = "../../raa-lit/src/slots.rs"]
mod lit_slots;

use raa_shader::{Composer, MaterialFnRegistry, PassKind, VariantKey};

fn scene_wgsl(name: &str) -> String {
    let src = include_str!("../../../examples/lookdev/scenes/m13-liquids-particles.js");
    let marker = format!("const {name} = `");
    let start = src.find(&marker).expect("constant") + marker.len();
    let end = src[start..].find("`;").expect("end") + start;
    src[start..end].to_string()
}

fn validate(name: &str, wgsl: &str) {
    let mut composer = Composer::new();
    for p in lit_slots::default_slot_providers() {
        composer.add_slot(p).unwrap();
    }
    let mut reg = MaterialFnRegistry::new();
    let id = reg.register_builtin(name, wgsl).unwrap();
    for pass in [PassKind::Forward, PassKind::Prepass] {
        let wgsl = composer
            .compose(
                &reg,
                &VariantKey {
                    material_fn: id,
                    pass,
                    msaa: 4,
                    flags: 0,
                },
            )
            .unwrap();
        Composer::validate(&wgsl).unwrap();
    }
}

#[test]
fn scene_material_functions_validate() {
    for name in ["GRASS_WGSL", "CROWN_WGSL"] {
        validate(name, &scene_wgsl(name));
    }
}
