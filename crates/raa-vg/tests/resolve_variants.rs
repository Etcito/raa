//! The S32 resolve module (`resolve.wgsl`): naga validation of the composed module, its
//! `main` compute entry point and workgroup size, the group-2 bindings 0-12, the per-stage
//! storage-buffer and storage-texture budget, the one depth helper it calls, the removed
//! wetness/waterline/tier text and the 320-byte `VgFrame` record.

#![cfg(not(target_arch = "wasm32"))]

use raa_shader::{Composer, DEPTH_WGSL};
use raa_vg::culling::PASS_VIEW_WGSL;
use raa_vg::resolve::resolve_module_wgsl;
use std::collections::BTreeSet;
use wgpu::naga;

/// Every `raa_depth_` identifier in `src`, unique and in first-appearance order.
fn depth_calls(src: &str) -> Vec<&str> {
    const PREFIX: &str = "raa_depth_";
    let mut names: Vec<&str> = Vec::new();
    for (start, _) in src.match_indices(PREFIX) {
        let rest = &src[start..];
        let end = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        let name = &rest[..end];
        if name.len() > PREFIX.len() && !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

fn parse(source: &str) -> naga::Module {
    naga::front::wgsl::parse_str(source).expect("the module parses")
}

/// Every `(group, binding)` the entry point reaches, following the functions it calls.
fn used_bindings(module: &naga::Module, entry: &naga::EntryPoint) -> BTreeSet<(u32, u32)> {
    fn walk(
        module: &naga::Module,
        function: &naga::Function,
        seen: &mut BTreeSet<naga::Handle<naga::Function>>,
        out: &mut BTreeSet<(u32, u32)>,
    ) {
        for (_, expression) in function.expressions.iter() {
            if let naga::Expression::GlobalVariable(handle) = *expression
                && let Some(binding) = module.global_variables[handle].binding
            {
                out.insert((binding.group, binding.binding));
            }
        }
        let mut blocks = vec![&function.body];
        while let Some(block) = blocks.pop() {
            for statement in block.iter() {
                match statement {
                    naga::Statement::Call { function, .. } => {
                        if seen.insert(*function) {
                            walk(module, &module.functions[*function], seen, out);
                        }
                    }
                    naga::Statement::Block(inner) => blocks.push(inner),
                    naga::Statement::If { accept, reject, .. } => blocks.extend([accept, reject]),
                    naga::Statement::Switch { cases, .. } => {
                        blocks.extend(cases.iter().map(|case| &case.body));
                    }
                    naga::Statement::Loop {
                        body, continuing, ..
                    } => blocks.extend([body, continuing]),
                    _ => {}
                }
            }
        }
    }
    let mut bindings = BTreeSet::new();
    walk(module, &entry.function, &mut BTreeSet::new(), &mut bindings);
    bindings
}

fn global(module: &naga::Module, group: u32, binding: u32) -> &naga::GlobalVariable {
    module
        .global_variables
        .iter()
        .find(|(_, global)| global.binding == Some(naga::ResourceBinding { group, binding }))
        .map(|(_, global)| global)
        .unwrap_or_else(|| panic!("group {group} binding {binding} is declared"))
}

#[test]
fn resolve_module_validates_with_the_main_compute_entry_point() {
    let wgsl = resolve_module_wgsl();
    assert_eq!(Composer::validate(&wgsl), Ok(()));
    let module = parse(&wgsl);
    assert_eq!(
        module.entry_points.len(),
        1,
        "the resolve has one entry point"
    );
    let entry = &module.entry_points[0];
    assert_eq!(entry.name, "main");
    assert_eq!(entry.stage, naga::ShaderStage::Compute);
    assert_eq!(entry.workgroup_size, [8, 8, 1]);
}

#[test]
fn resolve_binds_group_two_zero_to_twelve() {
    let module = parse(&resolve_module_wgsl());
    let mut bindings: Vec<(u32, u32)> = module
        .global_variables
        .iter()
        .map(|(_, global)| {
            let binding = global.binding.expect("every global is bound");
            (binding.group, binding.binding)
        })
        .collect();
    bindings.sort_unstable();
    assert_eq!(
        bindings,
        (0..=12).map(|binding| (2, binding)).collect::<Vec<_>>()
    );
    for (_, global) in module.global_variables.iter() {
        assert_eq!(
            global.binding.expect("every global is bound").group,
            2,
            "no binding sits outside group 2"
        );
    }
    for binding in 1..=8 {
        let global = global(&module, 2, binding);
        assert!(
            matches!(
                global.space,
                naga::AddressSpace::Storage { access }
                    if access == naga::StorageAccess::LOAD
            ),
            "binding {binding} is a read-only storage buffer"
        );
    }
    for binding in 9..=12 {
        let global = global(&module, 2, binding);
        assert_eq!(global.space, naga::AddressSpace::Handle);
        assert!(
            matches!(
                module.types[global.ty].inner,
                naga::TypeInner::Image {
                    class: naga::ImageClass::Storage {
                        access: naga::StorageAccess::STORE,
                        ..
                    },
                    ..
                }
            ),
            "binding {binding} is a write-only storage texture"
        );
    }
}

#[test]
fn resolve_stage_uses_eight_storage_buffers_and_four_storage_textures() {
    let module = parse(&resolve_module_wgsl());
    let entry = &module.entry_points[0];
    let used = used_bindings(&module, entry);
    assert_eq!(
        used,
        (0..=12)
            .map(|binding| (2, binding))
            .collect::<BTreeSet<_>>(),
        "the stage reaches every declared binding"
    );
    let mut storage_buffers = 0;
    let mut storage_textures = 0;
    for (group, binding) in used {
        let global = global(&module, group, binding);
        match global.space {
            naga::AddressSpace::Storage { access } => {
                assert_eq!(access, naga::StorageAccess::LOAD);
                storage_buffers += 1;
            }
            naga::AddressSpace::Handle => {
                if matches!(
                    module.types[global.ty].inner,
                    naga::TypeInner::Image {
                        class: naga::ImageClass::Storage { .. },
                        ..
                    }
                ) {
                    storage_textures += 1;
                }
            }
            naga::AddressSpace::Uniform => {}
            ref other => panic!("unexpected binding {binding} in {other:?}"),
        }
    }
    assert_eq!(storage_buffers, 8, "S32 allows eight storage buffers");
    assert_eq!(storage_textures, 4, "S32 allows four storage textures");
}

#[test]
fn resolve_calls_only_the_depth_from_key_helper() {
    let wgsl = resolve_module_wgsl();
    assert!(wgsl.starts_with(DEPTH_WGSL));
    assert!(
        wgsl[DEPTH_WGSL.len()..].starts_with(PASS_VIEW_WGSL),
        "the scene twins come right after the depth helpers"
    );
    let body = wgsl
        .strip_prefix(DEPTH_WGSL)
        .expect("the resolve module starts with the depth helper");
    let calls = depth_calls(body);
    let allowed = ["raa_depth_from_key", "raa_depth_is_far"];
    for call in &calls {
        assert!(
            allowed.contains(call),
            "{call} is not one of the depth helpers the resolve may call"
        );
    }
    assert_eq!(calls, ["raa_depth_from_key"]);
}

#[test]
fn resolve_drops_the_wetness_waterline_and_tier_textures() {
    let wgsl = resolve_module_wgsl();
    for removed in ["waterline", "wetness", "texture_2d_array"] {
        assert!(
            !wgsl.contains(removed),
            "the adapted resolve removes {removed}"
        );
    }
}

#[test]
fn vg_frame_is_320_bytes() {
    let module = parse(&resolve_module_wgsl());
    let (handle, ty) = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some("VgFrame"))
        .expect("VgFrame is declared");
    let naga::TypeInner::Struct { span, .. } = &ty.inner else {
        panic!("VgFrame is a struct");
    };
    assert_eq!(*span, 320);
    let mut layouter = naga::proc::Layouter::default();
    layouter.update(module.to_ctx()).expect("the types lay out");
    assert_eq!(layouter[handle].size, 320);
}
