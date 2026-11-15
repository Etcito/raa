//! P4-Q1/S44: the area-shadow trace, temporal and spatial modules naga-validate, their entries
//! follow the 8x8 compute workgroup, their group-0 or group-2 bindings match the layout entries
//! and resource kinds, group 1 stays unused or inside the GI compute set, their records match
//! contract 'area-stochastic-gpu', contract 'area-shadow-trace-wgsl' and contract
//! 'area-shadow-denoise-wgsl'.
//!
//! The builders, entry constants and entry arrays come from the Handoff doc comment at
//! crates/raa-gi/src/parts/area_shadow/passes.rs:117 `temporal_module_wgsl`.

#![cfg(not(target_arch = "wasm32"))]

use naga::{AddressSpace, ImageClass, ShaderStage, TypeInner};
use raa_core::wgpu::naga;
use raa_core::wgpu::{BindGroupLayoutEntry, BindingType, BufferBindingType, TextureSampleType};
use raa_gi::parts::area_shadow::passes::{
    AREA_DISPATCH_SIZE, AREA_PARAMS_SIZE, AREA_SPATIAL_H_ENTRY, AREA_SPATIAL_V_ENTRY,
    AREA_TEMPORAL_ENTRY, AREA_TRACE_ENTRY, WORKGROUP, spatial_group0_entries, spatial_module_wgsl,
    temporal_group0_entries, temporal_module_wgsl, trace_group2_entries, trace_module_wgsl,
};

fn numbers(entries: &[BindGroupLayoutEntry]) -> Vec<u32> {
    entries.iter().map(|entry| entry.binding).collect()
}

/// A validated WGSL module and its per-entry resource usage.
struct Reflection {
    module: naga::Module,
    info: naga::valid::ModuleInfo,
}

impl Reflection {
    fn new(source: &str) -> Reflection {
        let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|e| panic!("parse: {e}"));
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("validate: {e}"));
        Reflection { module, info }
    }

    fn entry(&self, name: &str) -> usize {
        self.module
            .entry_points
            .iter()
            .position(|entry| entry.name == name)
            .unwrap_or_else(|| panic!("entry {name}"))
    }

    /// The `group` bindings of entry `index` and the globals that declare them, ascending.
    fn used_globals(&self, index: usize, group: u32) -> Vec<(u32, &naga::GlobalVariable)> {
        let usage = self.info.get_entry_point(index);
        let mut out = Vec::new();
        for (handle, global) in self.module.global_variables.iter() {
            let Some(binding) = &global.binding else {
                continue;
            };
            if binding.group == group && !usage[handle].is_empty() {
                out.push((binding.binding, global));
            }
        }
        out.sort_by_key(|(binding, _)| *binding);
        out
    }

    /// The binding numbers of `group` that entry `index` statically uses, sorted.
    fn used(&self, index: usize, group: u32) -> Vec<u32> {
        self.used_globals(index, group)
            .into_iter()
            .map(|(binding, _)| binding)
            .collect()
    }

    /// The storage textures and storage buffers entry `index` uses.
    fn storage(&self, index: usize) -> (u32, u32) {
        let usage = self.info.get_entry_point(index);
        let (mut textures, mut buffers) = (0, 0);
        for (handle, global) in self.module.global_variables.iter() {
            if usage[handle].is_empty() {
                continue;
            }
            match global.space {
                AddressSpace::Storage { .. } => buffers += 1,
                AddressSpace::Handle => {
                    if matches!(
                        self.module.types[global.ty].inner,
                        TypeInner::Image {
                            class: ImageClass::Storage { .. },
                            ..
                        }
                    ) {
                        textures += 1;
                    }
                }
                _ => {}
            }
        }
        (textures, buffers)
    }
}

/// A short description of a global's resource kind, comparable with `entry_kind`.
fn naga_kind(module: &naga::Module, global: &naga::GlobalVariable) -> String {
    match global.space {
        AddressSpace::Uniform => "uniform".into(),
        AddressSpace::Storage { access } if access == naga::StorageAccess::LOAD => {
            "storage-ro".into()
        }
        AddressSpace::Storage { .. } => "storage-rw".into(),
        _ => match &module.types[global.ty].inner {
            TypeInner::Sampler { .. } => "sampler".into(),
            TypeInner::Image {
                dim,
                arrayed,
                class,
            } => {
                let dim = if *arrayed {
                    format!("{dim:?}Array")
                } else {
                    format!("{dim:?}")
                };
                match class {
                    ImageClass::Storage { format, .. } => format!("{dim} {format:?} write"),
                    ImageClass::Depth { .. } => format!("{dim} depth"),
                    ImageClass::Sampled { .. } => format!("{dim} float"),
                    ImageClass::External => format!("{dim} external"),
                }
            }
            _ => panic!("unexpected global type"),
        },
    }
}

/// The resource kind a group-2 layout entry declares, in `naga_kind`'s terms.
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
        BindingType::Sampler(_) => "sampler".into(),
        ref other => panic!("unexpected entry {other:?}"),
    }
}

fn struct_span(module: &naga::Module, name: &str) -> usize {
    let (_, ty) = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("{name} is declared"));
    let TypeInner::Struct { span, .. } = &ty.inner else {
        panic!("{name} is a struct");
    };
    *span as usize
}

fn check_struct(module: &naga::Module, name: &str, span: usize, expected: &[(&str, u32)]) {
    let (_, ty) = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("{name} is declared"));
    let TypeInner::Struct {
        members: fields,
        span: actual,
    } = &ty.inner
    else {
        panic!("{name} is a struct");
    };
    assert_eq!(*actual as usize, span, "{name} span");
    let found: Vec<(&str, u32)> = fields
        .iter()
        .map(|member| (member.name.as_deref().unwrap_or(""), member.offset))
        .collect();
    assert_eq!(found, expected, "{name} member offsets");
}

#[test]
fn trace_module_validates_and_holds_its_kernel() {
    let source = trace_module_wgsl();
    let reflection = Reflection::new(&source);
    assert!(
        source.contains(AREA_TRACE_ENTRY),
        "the module holds its entry"
    );
    for name in [
        "fn raa_area_cell_side(",
        "fn raa_area_stratum_cell(",
        "fn raa_area_r2_jitter(",
        "fn raa_area_stratum_uv(",
        "fn raa_area_sample_weight(",
        "fn raa_area_volume_visibility(",
        "fn raa_area_ray_visibility(",
    ] {
        assert!(source.contains(name), "the trace module holds {name}");
    }
    let computes: Vec<&str> = reflection
        .module
        .entry_points
        .iter()
        .map(|entry| entry.name.as_str())
        .collect();
    assert_eq!(computes, [AREA_TRACE_ENTRY]);
}

#[test]
fn entry_follows_the_compute_workgroup() {
    let source = trace_module_wgsl();
    let marker = format!("@compute @workgroup_size(8, 8)\nfn {AREA_TRACE_ENTRY}(");
    assert!(source.contains(&marker), "the entry follows its workgroup");
    let reflection = Reflection::new(&source);
    let index = reflection.entry(AREA_TRACE_ENTRY);
    let entry = &reflection.module.entry_points[index];
    assert_eq!(entry.stage, ShaderStage::Compute);
    assert_eq!(entry.workgroup_size, [8, 8, 1]);
    assert_eq!(WORKGROUP, 8);
}

#[test]
fn entry_bindings_match_the_trace_layout() {
    let reflection = Reflection::new(&trace_module_wgsl());
    let index = reflection.entry(AREA_TRACE_ENTRY);
    let entries = trace_group2_entries();
    assert_eq!(numbers(&entries), [0, 1, 2, 4, 5, 6, 8, 9]);
    assert_eq!(
        reflection.used(index, 2),
        numbers(&entries),
        "group-2 bindings"
    );
    for (binding, global) in reflection.used_globals(index, 2) {
        let entry = entries
            .iter()
            .find(|entry| entry.binding == binding)
            .unwrap_or_else(|| panic!("group 2 binding {binding} has an entry"));
        assert_eq!(
            naga_kind(&reflection.module, global),
            entry_kind(entry),
            "group 2 binding {binding} kind"
        );
        if let BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            min_binding_size,
            ..
        } = entry.ty
        {
            let TypeInner::Struct { span, .. } = &reflection.module.types[global.ty].inner else {
                panic!("group 2 binding {binding} is not a struct");
            };
            assert_eq!(
                min_binding_size.map(|size| size.get()),
                Some(*span as u64),
                "group 2 binding {binding} size"
            );
        }
    }
    let group1 = [3, 6, 12, 15, 16, 17, 18];
    let used1 = reflection.used(index, 1);
    assert!(
        used1.iter().all(|binding| group1.contains(binding)),
        "group-1 bindings {used1:?}"
    );
    assert!(reflection.used(index, 0).is_empty(), "group 0 is unused");
    let (textures, buffers) = reflection.storage(index);
    assert!(textures <= 4, "{textures} storage textures");
    assert!(buffers <= 8, "{buffers} storage buffers");
}

#[test]
fn records_match_contract_area_stochastic_gpu() {
    let reflection = Reflection::new(&trace_module_wgsl());
    let table = [("lights", 0), ("weights", 32), ("params", 64)];
    check_struct(&reflection.module, "RaaAreaStochasticGpu", 80, &table);
    let params = [("frame_index", 0), ("pad0", 4), ("pad1", 8), ("pad2", 12)];
    check_struct(&reflection.module, "AreaShadowParamsGpu", 16, &params);
    assert_eq!(AREA_PARAMS_SIZE, 16);
}

#[test]
fn denoise_modules_validate_and_hold_their_kernels() {
    let temporal = temporal_module_wgsl();
    let reflection = Reflection::new(&temporal);
    assert!(
        temporal.contains(AREA_TEMPORAL_ENTRY),
        "the module holds its entry"
    );
    for name in [
        "fn raa_area_temporal_blend(",
        "fn raa_area_depth_consistent(",
    ] {
        assert!(temporal.contains(name), "the temporal module holds {name}");
    }
    let computes: Vec<&str> = reflection
        .module
        .entry_points
        .iter()
        .map(|entry| entry.name.as_str())
        .collect();
    assert_eq!(computes, [AREA_TEMPORAL_ENTRY]);

    let spatial = spatial_module_wgsl();
    let reflection = Reflection::new(&spatial);
    for name in [AREA_SPATIAL_H_ENTRY, AREA_SPATIAL_V_ENTRY] {
        assert!(spatial.contains(name), "the module holds {name}");
    }
    for name in [
        "fn raa_area_bilateral_weight(",
        "fn raa_area_ratio_visibility(",
    ] {
        assert!(spatial.contains(name), "the spatial module holds {name}");
    }
    let computes: Vec<&str> = reflection
        .module
        .entry_points
        .iter()
        .map(|entry| entry.name.as_str())
        .collect();
    assert_eq!(computes, [AREA_SPATIAL_H_ENTRY, AREA_SPATIAL_V_ENTRY]);
}

#[test]
fn denoise_entries_follow_the_compute_workgroup() {
    assert_eq!(WORKGROUP, 8);
    let temporal = Reflection::new(&temporal_module_wgsl());
    let spatial = Reflection::new(&spatial_module_wgsl());
    for (reflection, name) in [
        (&temporal, AREA_TEMPORAL_ENTRY),
        (&spatial, AREA_SPATIAL_H_ENTRY),
        (&spatial, AREA_SPATIAL_V_ENTRY),
    ] {
        let entry = &reflection.module.entry_points[reflection.entry(name)];
        assert_eq!(entry.stage, ShaderStage::Compute, "{name} stage");
        assert_eq!(entry.workgroup_size, [8, 8, 1], "{name} workgroup");
    }
}

/// Asserts that the group-0 globals of `name` are `used` and match the kinds and uniform sizes
/// of `entries`.
fn check_bindings(
    reflection: &Reflection,
    name: &str,
    entries: &[BindGroupLayoutEntry],
    used: &[u32],
) {
    let index = reflection.entry(name);
    assert_eq!(reflection.used(index, 0), used, "{name} group-0 bindings");
    for (binding, global) in reflection.used_globals(index, 0) {
        let entry = entries
            .iter()
            .find(|entry| entry.binding == binding)
            .unwrap_or_else(|| panic!("group 0 binding {binding} has an entry"));
        assert_eq!(
            naga_kind(&reflection.module, global),
            entry_kind(entry),
            "{name} group 0 binding {binding} kind"
        );
        if let BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            min_binding_size,
            ..
        } = entry.ty
        {
            let TypeInner::Struct { span, .. } = &reflection.module.types[global.ty].inner else {
                panic!("group 0 binding {binding} is not a struct");
            };
            assert_eq!(
                min_binding_size.map(|size| size.get()),
                Some(*span as u64),
                "{name} group 0 binding {binding} size"
            );
        }
    }
    assert!(
        reflection.used(index, 1).is_empty(),
        "{name} uses no group 1"
    );
    assert!(
        reflection.used(index, 2).is_empty(),
        "{name} uses no group 2"
    );
    let (textures, _) = reflection.storage(index);
    assert!(textures <= 4, "{name}: {textures} storage textures");
}

#[test]
fn temporal_bindings_match_their_layout() {
    let reflection = Reflection::new(&temporal_module_wgsl());
    let entries = temporal_group0_entries();
    assert_eq!(numbers(&entries), [0, 1, 2, 3, 4, 5, 6, 7, 8]);
    check_bindings(
        &reflection,
        AREA_TEMPORAL_ENTRY,
        &entries,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8],
    );
}

#[test]
fn spatial_entries_match_their_layout() {
    let reflection = Reflection::new(&spatial_module_wgsl());
    let entries = spatial_group0_entries();
    let allowed: [u32; 6] = [1, 2, 3, 4, 5, 7];
    assert_eq!(numbers(&entries), allowed, "the shared layout");
    let horizontal = reflection.used(reflection.entry(AREA_SPATIAL_H_ENTRY), 0);
    let vertical = reflection.used(reflection.entry(AREA_SPATIAL_V_ENTRY), 0);
    for used in [&horizontal, &vertical] {
        assert!(
            used.iter().all(|binding| allowed.contains(binding)),
            "group-0 bindings {used:?}"
        );
        for required in [1u32, 2, 3, 5, 7] {
            assert!(used.contains(&required), "binding {required} is used");
        }
    }
    assert_eq!(horizontal, [1, 2, 3, 5, 7], "the horizontal entry");
    assert!(vertical.contains(&4), "the vertical entry reads src_b");
    check_bindings(&reflection, AREA_SPATIAL_H_ENTRY, &entries, &horizontal);
    check_bindings(&reflection, AREA_SPATIAL_V_ENTRY, &entries, &vertical);
}

#[test]
fn records_match_contract_area_shadow_denoise_wgsl() {
    let temporal = Reflection::new(&temporal_module_wgsl());
    assert_eq!(struct_span(&temporal.module, "RaaTemporalFrame"), 768);
    let dispatch = [("index", 0), ("usable", 4), ("pairs", 8), ("limit", 12)];
    check_struct(&temporal.module, "AreaDispatchGpu", 16, &dispatch);
    assert_eq!(AREA_DISPATCH_SIZE, 16);
    let spatial = Reflection::new(&spatial_module_wgsl());
    check_struct(&spatial.module, "AreaDispatchGpu", 16, &dispatch);
}
