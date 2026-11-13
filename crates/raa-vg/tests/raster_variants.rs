//! The S32 hardware raster module (`raster_common`, `vis_write`, `raster_hw` and
//! `raster_hw_fragment`): naga validation of the module, its entry points and stages, the
//! group-2 bindings 0-8, the per-stage storage-buffer budget, the flat integer varyings and
//! the one depth helper its visibility write calls.

#![cfg(not(target_arch = "wasm32"))]

use raa_shader::{Composer, DEPTH_WGSL};
use raa_vg::culling::PASS_VIEW_WGSL;
use raa_vg::raster::{RASTER_COMMON_WGSL, VIS_WRITE_WGSL, raster_hw_module_wgsl};
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

/// The storage bindings (1-8) an entry point reaches in group 2.
fn stage_storage(module: &naga::Module, entry: &naga::EntryPoint) -> BTreeSet<u32> {
    used_bindings(module, entry)
        .into_iter()
        .filter(|(group, binding)| *group == 2 && (1..=8).contains(binding))
        .map(|(_, binding)| binding)
        .collect()
}

#[test]
fn raster_hw_module_validates_with_its_entry_points_and_stages() {
    let wgsl = raster_hw_module_wgsl();
    assert_eq!(Composer::validate(&wgsl), Ok(()));
    let module = parse(&wgsl);
    let mut entry_points: Vec<(&str, naga::ShaderStage)> = module
        .entry_points
        .iter()
        .map(|entry| (entry.name.as_str(), entry.stage))
        .collect();
    entry_points.sort_unstable();
    assert_eq!(
        entry_points,
        [
            ("fs_visibility", naga::ShaderStage::Fragment),
            ("vs_depth", naga::ShaderStage::Vertex),
            ("vs_visibility", naga::ShaderStage::Vertex),
        ]
    );
}

#[test]
fn raster_hw_binds_group_two_zero_to_eight() {
    let module = parse(&raster_hw_module_wgsl());
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
        [
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (2, 7),
            (2, 8)
        ]
    );
    for (_, global) in module.global_variables.iter() {
        let binding = global.binding.expect("every global is bound");
        assert_eq!(binding.group, 2, "no binding sits outside group 2");
        match binding.binding {
            0 => assert!(matches!(global.space, naga::AddressSpace::Uniform)),
            1..=6 => assert!(matches!(
                global.space,
                naga::AddressSpace::Storage { access } if access == naga::StorageAccess::LOAD
            )),
            _ => assert!(matches!(
                global.space,
                naga::AddressSpace::Storage { access }
                    if access == naga::StorageAccess::LOAD | naga::StorageAccess::STORE
            )),
        }
    }
}

#[test]
fn raster_stages_stay_within_the_storage_budget() {
    let module = parse(&raster_hw_module_wgsl());
    for entry in &module.entry_points {
        let storage = stage_storage(&module, entry);
        assert!(
            storage.len() <= 8,
            "entry point {} uses more than eight storage buffers",
            entry.name
        );
        match entry.stage {
            naga::ShaderStage::Vertex => assert_eq!(
                storage,
                BTreeSet::from([1, 2, 3, 4, 5, 6]),
                "the vertex stage reads the scene and cull-state buffers"
            ),
            naga::ShaderStage::Fragment => assert_eq!(
                storage,
                BTreeSet::from([7, 8]),
                "the fragment stage writes only the visibility buffers"
            ),
            _ => panic!("the raster module holds vertex and fragment stages only"),
        }
    }
}

#[test]
fn vertex_integer_outputs_are_flat() {
    let module = parse(&raster_hw_module_wgsl());
    let entry = module
        .entry_points
        .iter()
        .find(|entry| entry.name == "vs_visibility")
        .expect("vs_visibility is an entry point");
    let result = entry
        .function
        .result
        .as_ref()
        .expect("vs_visibility returns its varyings");
    let naga::TypeInner::Struct { members, .. } = &module.types[result.ty].inner else {
        panic!("the varyings of vs_visibility are a struct");
    };
    let mut integer_outputs = 0;
    for member in members {
        let naga::TypeInner::Scalar(scalar) = &module.types[member.ty].inner else {
            continue;
        };
        if !matches!(scalar.kind, naga::ScalarKind::Sint | naga::ScalarKind::Uint) {
            continue;
        }
        integer_outputs += 1;
        assert!(
            matches!(
                member.binding,
                Some(naga::Binding::Location {
                    interpolation: Some(naga::Interpolation::Flat),
                    ..
                })
            ),
            "the integer varying {:?} is flat",
            member.name
        );
    }
    assert_eq!(
        integer_outputs, 1,
        "the flat visible id is the only integer varying"
    );
}

/// The module's own WGSL (after the `DEPTH_WGSL` declarations) calls exactly one depth helper:
/// `raa_depth_key`, which orders standard-Z depth the way the reversed-Z bits of the source do.
#[test]
fn raster_calls_only_the_depth_key_helper() {
    let wgsl = raster_hw_module_wgsl();
    assert!(wgsl.starts_with(DEPTH_WGSL));
    assert!(
        wgsl[DEPTH_WGSL.len()..].starts_with(PASS_VIEW_WGSL),
        "the scene twins come right after the depth helpers"
    );
    let order = [
        DEPTH_WGSL,
        PASS_VIEW_WGSL,
        RASTER_COMMON_WGSL,
        VIS_WRITE_WGSL,
    ]
    .concat();
    assert!(
        wgsl.starts_with(&order),
        "the builder concatenates DEPTH_WGSL, PASS_VIEW_WGSL, RASTER_COMMON_WGSL and VIS_WRITE_WGSL in order"
    );
    let body = wgsl
        .strip_prefix(DEPTH_WGSL)
        .expect("the raster module starts with the depth helper");
    assert_eq!(depth_calls(body), ["raa_depth_key"]);
    let vis_write = wgsl
        .find(VIS_WRITE_WGSL)
        .expect("vis_write is part of the module");
    let vertex = wgsl
        .find("fn vs_visibility(")
        .expect("the vertex entry point follows the shared writes");
    assert!(
        vis_write < vertex,
        "write_visibility is declared before its callers"
    );
}
