//! Contract `gi-shaders`: every builder's module validates through `Composer::validate` and
//! naga, with its entry point, workgroup size and group-2 and group-1 globals pinned, plus the
//! browser bitwise-parenthesis rule of `raa-shader` `composer.rs:313`.

#![cfg(not(target_arch = "wasm32"))]

use raa_gi::shaders::*;
use raa_shader::{Composer, GI_VOLUME_WGSL, PRELUDE_WGSL};
use wgpu::naga::{self, valid};
use wgpu::{BindGroupLayoutEntry, BindingType, BufferBindingType, TextureSampleType};

struct Variant {
    name: &'static str,
    wgsl: String,
    entry: &'static str,
    workgroup: [u32; 3],
    group2: Vec<BindGroupLayoutEntry>,
    group1: bool,
}

fn variants() -> Vec<Variant> {
    let v = |name, wgsl, entry, workgroup, group2: &[BindGroupLayoutEntry], group1| Variant {
        name,
        wgsl,
        entry,
        workgroup,
        group2: group2.to_vec(),
        group1,
    };
    vec![
        v(
            "seed",
            seed_module_wgsl(),
            "raa_gi_seed",
            [4, 4, 4],
            &seed_group2_entries(),
            false,
        ),
        v(
            "flood",
            flood_module_wgsl(),
            "raa_gi_flood",
            [4, 4, 4],
            &flood_group2_entries(),
            false,
        ),
        v(
            "forward",
            voxelize_forward_module_wgsl(),
            "raa_gi_voxelize_forward",
            [64, 1, 1],
            &forward_group2_entries(),
            false,
        ),
        v(
            "write",
            volume_write_module_wgsl(),
            "raa_gi_volume_write",
            [4, 4, 4],
            &write_group2_entries(),
            false,
        ),
        v(
            "light",
            voxel_light_module_wgsl(),
            "raa_gi_voxel_light",
            [4, 4, 4],
            &light_group2_entries(),
            true,
        ),
        v(
            "debug",
            debug_view_module_wgsl(),
            "raa_gi_debug_view",
            [8, 8, 1],
            &debug_group2_entries(),
            true,
        ),
    ]
}

fn analyse(wgsl: &str) -> (naga::Module, valid::ModuleInfo) {
    let module = naga::front::wgsl::parse_str(wgsl).expect("the module parses");
    let info = valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
        .validate(&module)
        .expect("the module validates");
    (module, info)
}

/// Globals the single entry point reaches (callees included): (group, binding, handle).
fn used(
    module: &naga::Module,
    info: &valid::ModuleInfo,
) -> Vec<(u32, u32, naga::Handle<naga::GlobalVariable>)> {
    let usage = info.get_entry_point(0);
    module
        .global_variables
        .iter()
        .filter(|(handle, _)| !usage[*handle].is_empty())
        .filter_map(|(handle, global)| global.binding.map(|b| (b.group, b.binding, handle)))
        .collect()
}

fn naga_kind(module: &naga::Module, global: &naga::GlobalVariable) -> String {
    match global.space {
        naga::AddressSpace::Uniform => "uniform".into(),
        naga::AddressSpace::Storage { access } if access == naga::StorageAccess::LOAD => {
            "storage-ro".into()
        }
        naga::AddressSpace::Storage { .. } => "storage-rw".into(),
        _ => match module.types[global.ty].inner {
            naga::TypeInner::Image {
                dim,
                class: naga::ImageClass::Storage { format, access },
                ..
            } => {
                format!(
                    "{dim:?} {format:?} {}",
                    if access == naga::StorageAccess::STORE {
                        "write"
                    } else {
                        "other"
                    }
                )
            }
            naga::TypeInner::Image {
                dim,
                class: naga::ImageClass::Depth { .. },
                ..
            } => format!("{dim:?} depth"),
            naga::TypeInner::Image {
                dim,
                class: naga::ImageClass::Sampled { .. },
                ..
            } => format!("{dim:?} float"),
            ref other => panic!("unexpected global type {other:?}"),
        },
    }
}

fn entry_kind(entry: &BindGroupLayoutEntry) -> String {
    match entry.ty {
        BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            ..
        } => "uniform".into(),
        BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: true },
            ..
        } => "storage-ro".into(),
        BindingType::Buffer { .. } => "storage-rw".into(),
        BindingType::StorageTexture {
            format,
            view_dimension,
            ..
        } => format!("{view_dimension:?} {format:?} write"),
        BindingType::Texture {
            sample_type: TextureSampleType::Depth,
            view_dimension,
            ..
        } => format!("{view_dimension:?} depth"),
        BindingType::Texture { view_dimension, .. } => format!("{view_dimension:?} float"),
        ref other => panic!("unexpected entry {other:?}"),
    }
}

#[test]
fn every_builder_validates_with_its_entry_point_and_workgroup() {
    for var in variants() {
        assert_eq!(Composer::validate(&var.wgsl), Ok(()), "{}", var.name);
        let (module, _) = analyse(&var.wgsl);
        assert_eq!(module.entry_points.len(), 1, "{}", var.name);
        let entry = &module.entry_points[0];
        assert_eq!(entry.name, var.entry);
        assert_eq!(entry.stage, naga::ShaderStage::Compute);
        assert_eq!(entry.workgroup_size, var.workgroup, "{}", var.name);
    }
}

#[test]
fn group_two_globals_match_the_entries_and_group_one_stays_in_its_set() {
    for var in variants() {
        let (module, info) = analyse(&var.wgsl);
        for (group, binding, handle) in used(&module, &info) {
            let global = &module.global_variables[handle];
            match group {
                2 => {
                    let entry = var.group2.iter().find(|e| e.binding == binding);
                    let entry = entry.unwrap_or_else(|| panic!("{} binds 2/{binding}", var.name));
                    assert_eq!(
                        naga_kind(&module, global),
                        entry_kind(entry),
                        "{} 2/{binding}",
                        var.name
                    );
                }
                1 => assert!(
                    var.group1 && [3, 6, 12, 15, 16, 17, 18].contains(&binding),
                    "{} uses group 1 binding {binding}",
                    var.name
                ),
                other => panic!("{} uses group {other}", var.name),
            }
        }
    }
}

#[test]
fn voxel_light_composes_the_prelude_and_volume_library_once() {
    let wgsl = voxel_light_module_wgsl();
    assert_eq!(wgsl.matches(PRELUDE_WGSL).count(), 1);
    assert_eq!(wgsl.matches(GI_VOLUME_WGSL).count(), 1);
}

#[test]
fn driver_a_group_two_holds_six_storage_buffers() {
    let storage = |e: &BindGroupLayoutEntry| {
        matches!(
            e.ty,
            BindingType::Buffer {
                ty: BufferBindingType::Storage { .. },
                ..
            }
        )
    };
    assert_eq!(
        forward_group2_entries()
            .iter()
            .filter(|e| storage(e))
            .count(),
        6
    );
    let (module, info) = analyse(&voxelize_forward_module_wgsl());
    let count = used(&module, &info)
        .iter()
        .filter(|(_, _, h)| {
            matches!(
                module.global_variables[*h].space,
                naga::AddressSpace::Storage { .. }
            )
        })
        .count();
    assert_eq!(count, 6);
}

#[test]
fn write_module_binds_a_write_only_rgba16float_3d_atlas() {
    let (module, _) = analyse(&volume_write_module_wgsl());
    let atlas = module
        .global_variables
        .iter()
        .find(|(_, g)| g.binding.is_some_and(|b| b.group == 2 && b.binding == 3))
        .expect("binding 3")
        .1;
    assert_eq!(naga_kind(&module, atlas), "D3 Rgba16Float write");
    assert_eq!(
        entry_kind(&write_group2_entries()[3]),
        "D3 Rgba16Float write"
    );
}

/// Lines where a `&`, `|` or `^` shares a parenthesis level with another binary operator, or
/// two different bitwise operators do (a chain of one bitwise operator is legal WGSL).
fn bitwise_mixes(source: &str) -> Vec<String> {
    const BITWISE: [&str; 3] = [" & ", " | ", " ^ "];
    const OTHER: [&str; 13] = [
        " * ", " / ", " % ", " + ", " - ", " << ", " >> ", " == ", " != ", " < ", " > ", " <= ",
        " >= ",
    ];
    let mut found = Vec::new();
    for line in source.lines() {
        let code = line.split("//").next().unwrap_or("");
        let mut segments = Vec::new();
        let mut levels = vec![String::new()];
        for c in code.chars() {
            match c {
                '(' | '[' => {
                    levels.last_mut().unwrap().push(' ');
                    levels.push(String::new());
                }
                ')' | ']' => {
                    segments.push(levels.pop().unwrap_or_default());
                    if levels.is_empty() {
                        levels.push(String::new());
                    }
                }
                ',' | ';' => segments.push(std::mem::take(levels.last_mut().unwrap())),
                _ => levels.last_mut().unwrap().push(c),
            }
        }
        segments.extend(levels);
        for s in segments {
            let ops = BITWISE.iter().filter(|b| s.contains(**b)).count();
            if ops > 1 || (ops == 1 && OTHER.iter().any(|o| s.contains(o))) {
                found.push(line.trim().to_string());
                break;
            }
        }
    }
    found
}

#[test]
fn the_scanner_flags_a_mix_and_accepts_parentheses() {
    assert_eq!(
        bitwise_mixes("let a = x & y + 1u;\nlet b = f(x | y ^ z);").len(),
        2
    );
    assert!(bitwise_mixes("let a = (x & y) + 1u;\nlet b = x | y | z; // a ^ b + c").is_empty());
}

#[test]
fn no_builder_text_mixes_bitwise_operators_unparenthesised() {
    for var in variants() {
        assert_eq!(
            bitwise_mixes(&var.wgsl),
            Vec::<String>::new(),
            "{}",
            var.name
        );
    }
}

#[test]
fn adapted_files_keep_the_notice() {
    let files = [
        include_str!("../src/wgsl/radiance_codec.wgsl"),
        include_str!("../src/wgsl/voxelize_kernel.wgsl"),
        include_str!("../src/wgsl/seed.wgsl"),
        include_str!("../src/wgsl/jump_flood.wgsl"),
        include_str!("../src/wgsl/volume_write.wgsl"),
        include_str!("../src/wgsl/voxel_light.wgsl"),
    ];
    for (i, text) in files.iter().enumerate() {
        assert!(text.contains("Copyright (c) 2026 SamG-Coder"), "file {i}");
    }
}
