//! P4-Q1/S44: the area-shadow trace, temporal and spatial modules naga-validate, their entries
//! follow the 8x8 compute workgroup, their group-0 or group-2 bindings match the layout entries
//! and resource kinds, group 1 stays unused or inside the GI compute set, their records match
//! contract 'area-stochastic-gpu', contract 'area-shadow-trace-wgsl', contract
//! 'area-shadow-sdf-wgsl' and contract 'area-shadow-denoise-wgsl'.
//!
//! The builders, entry constants and entry arrays come from the Handoff doc comment at
//! crates/raa-gi/src/parts/area_shadow/passes.rs:138 `temporal_module_wgsl`.

#![cfg(not(target_arch = "wasm32"))]

use naga::{AddressSpace, ImageClass, ShaderStage, TypeInner};
use raa_core::wgpu::naga;
use raa_core::wgpu::{
    BindGroupLayoutEntry, BindingType, BufferBindingType, StorageTextureAccess, TextureFormat,
    TextureSampleType, TextureViewDimension,
};
use raa_gi::parts::area_shadow::passes::{
    AREA_DISPATCH_SIZE, AREA_PARAMS_SIZE, AREA_SPATIAL_H_ENTRY, AREA_SPATIAL_V_ENTRY,
    AREA_TEMPORAL_ENTRY, AREA_TRACE_ENTRY, WORKGROUP, spatial_group0_entries, spatial_module_wgsl,
    temporal_group0_entries, temporal_module_wgsl, trace_group2_entries, trace_module_wgsl,
};
use raa_gi::parts::area_shadow::restir::{
    RESERVOIR_BYTES, RESTIR_INITIAL_ENTRY, RESTIR_PARAMS_SIZE, RESTIR_SPATIAL_ENTRY,
    RESTIR_TEMPORAL_ENTRY, restir_common_module_wgsl, restir_group2_entries,
    restir_initial_module_wgsl, restir_spatial_module_wgsl, restir_temporal_module_wgsl,
};
use raa_gi::parts::area_shadow::restir_cpu::ROUGHNESS_MIN;

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
        "fn raa_area_sdf_ready(",
        "fn raa_area_sdf_distance(",
        "fn raa_area_sdf_visibility(",
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
    assert_eq!(numbers(&entries), [0, 1, 2, 4, 5, 6, 8, 9, 10, 11]);
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
    let sdf = [
        ("origin", 0),
        ("shape", 16),
        ("encoding", 32),
        ("build", 48),
    ];
    check_struct(&reflection.module, "RaaAreaSdfVolume", 64, &sdf);
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

#[test]
fn restir_common_module_validates_and_holds_its_helpers() {
    let source = restir_common_module_wgsl();
    let reflection = Reflection::new(&source);
    assert!(
        reflection.module.entry_points.is_empty(),
        "the stage-less common share has no entry point"
    );
    for name in [
        "struct RestirParamsGpu",
        "struct RaaRestirReservoir",
        "fn raa_restir_empty(",
        "fn raa_restir_pack(",
        "fn raa_restir_unpack(",
        "fn raa_restir_oct_encode(",
        "fn raa_restir_oct_decode(",
        "fn raa_restir_index(",
        "fn raa_restir_wrs_accept(",
        "fn raa_restir_contribution_weight(",
        "fn raa_restir_clamp_confidence(",
        "fn raa_restir_remap(",
        "fn raa_restir_boiling_keep(",
        "fn raa_restir_depth_similar(",
        "fn raa_restir_spatial_offset(",
        "fn raa_restir_random(",
        "fn raa_restir_luminance(",
        "fn raa_restir_srgb_decode(",
    ] {
        assert!(source.contains(name), "the common share holds {name}");
    }
}

#[test]
fn restir_params_record_matches_contract_area_restir_wgsl() {
    let reflection = Reflection::new(&restir_common_module_wgsl());
    let params = [
        ("frame_index", 0),
        ("history_usable", 4),
        ("candidates", 8),
        ("spatial_samples", 12),
        ("remap", 16),
        ("spatial_radius", 48),
        ("m_cap", 52),
        ("boiling_factor", 56),
        ("normal_cos", 60),
    ];
    check_struct(&reflection.module, "RestirParamsGpu", 64, &params);
    assert_eq!(RESTIR_PARAMS_SIZE, 64);
    assert_eq!(RESERVOIR_BYTES, 16);
    let reservoir = [
        ("slot", 0),
        ("uv", 8),
        ("weight", 16),
        ("confidence", 20),
        ("normal_oct", 24),
    ];
    check_struct(&reflection.module, "RaaRestirReservoir", 32, &reservoir);
}

#[test]
fn restir_group2_entries_match_the_registry() {
    let entries = restir_group2_entries();
    assert_eq!(
        numbers(&entries),
        [0, 1, 2, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    );
    let reflection = Reflection::new(&restir_common_module_wgsl());
    let mut declared = Vec::new();
    for (_, global) in reflection.module.global_variables.iter() {
        let Some(binding) = &global.binding else {
            continue;
        };
        if binding.group == 2 {
            declared.push((binding.binding, global));
        }
    }
    declared.sort_by_key(|(binding, _)| *binding);
    assert_eq!(
        declared
            .iter()
            .map(|(binding, _)| *binding)
            .collect::<Vec<_>>(),
        [0, 1, 2, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19],
        "the probe-common helpers (0, 1, 2, 7), the slot table and 12-19"
    );
    for (binding, global) in declared {
        let Some(entry) = entries.iter().find(|entry| entry.binding == binding) else {
            continue;
        };
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
    assert!(matches!(
        entries[4].ty,
        BindingType::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format: TextureFormat::Rgba16Float,
            view_dimension: TextureViewDimension::D2Array,
        }
    ));
    assert!(matches!(
        entries[7].ty,
        BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: false },
            view_dimension: TextureViewDimension::D3,
            multisampled: false,
        }
    ));
    assert!(matches!(
        entries[15].ty,
        BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        }
    ));
    for index in [13usize, 14] {
        assert!(matches!(
            entries[index].ty,
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                ..
            }
        ));
    }
}

/// Share 2 of contract 'area-restir-wgsl' (the surface, spherical-rectangle sampling, target and
/// LTC luminance functions of restir_common.wgsl), composed by the Handoff doc comment at
/// crates/raa-gi/src/parts/area_shadow/restir.rs:80 `restir_common_module_wgsl`.
const RESTIR_SHARE2_NAMES: [&str; 10] = [
    "fn raa_restir_surface(",
    "fn raa_restir_slot_light(",
    "fn raa_restir_light_point(",
    "fn raa_restir_representative(",
    "fn raa_restir_sample_rect(",
    "fn raa_restir_area_pdf(",
    "fn raa_restir_target(",
    "fn raa_restir_ltc_luminance(",
    "struct RaaRestirSurface",
    "struct RaaRestirRectSample",
];

/// The GI compute group 1 (`GiShared::compute_group1_layout`).
const GI_COMPUTE_GROUP1: [u32; 7] = [3, 6, 12, 15, 16, 17, 18];

/// A test-only compute caller of every share-2 function, appended to the stage-less common
/// module so naga checks the functions in the compute stage the ReSTIR entries run in.
const RESTIR_SHARE2_CALLER: &str = "
@compute @workgroup_size(8, 8)
fn raa_restir_share2_caller(@builtin(global_invocation_id) id: vec3<u32>) {
    let surface = raa_restir_surface(vec2<i32>(id.xy));
    let index = raa_restir_slot_light(id.z % 8u);
    if index == 0xFFFFFFFFu {
        return;
    }
    let light = raa_local_lights[index];
    let rect = raa_restir_sample_rect(light, surface.position, vec2<f32>(0.25, 0.75));
    let point = raa_restir_light_point(light, rect.uv);
    let to_point = point - surface.position;
    let cos_light = dot(light.direction_cos_outer.xyz, -normalize(to_point));
    let pdf = raa_restir_area_pdf(rect.solid_angle, cos_light, length(to_point));
    let value = raa_restir_target(surface, light, rect.uv) * pdf
        + raa_restir_ltc_luminance(surface, light);
    raa_restir_current[raa_restir_index(id.xy)] = vec4<u32>(bitcast<u32>(value));
}
";

fn function_handle(module: &naga::Module, name: &str) -> naga::Handle<naga::Function> {
    module
        .functions
        .iter()
        .find(|(_, function)| function.name.as_deref() == Some(name))
        .map(|(handle, _)| handle)
        .unwrap_or_else(|| panic!("fn {name} is declared"))
}

/// The `group` bindings function `name` uses (its own and its callees'), ascending.
fn function_bindings(reflection: &Reflection, name: &str, group: u32) -> Vec<u32> {
    let usage = &reflection.info[function_handle(&reflection.module, name)];
    let mut out: Vec<u32> = reflection
        .module
        .global_variables
        .iter()
        .filter(|(handle, _)| !usage[*handle].is_empty())
        .filter_map(|(_, global)| global.binding.as_ref())
        .filter(|binding| binding.group == group)
        .map(|binding| binding.binding)
        .collect();
    out.sort_unstable();
    out
}

#[test]
fn restir_common_share2_validates_and_holds_its_functions() {
    let source = restir_common_module_wgsl();
    let reflection = Reflection::new(&source);
    assert!(
        reflection.module.entry_points.is_empty(),
        "the stage-less common module has no entry point"
    );
    for name in RESTIR_SHARE2_NAMES {
        assert!(source.contains(name), "the common share holds {name}");
    }
    let members = |name: &str| -> Vec<String> {
        let (_, ty) = reflection
            .module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("{name} is declared"));
        let TypeInner::Struct { members, .. } = &ty.inner else {
            panic!("{name} is a struct");
        };
        members
            .iter()
            .map(|member| member.name.clone().unwrap_or_default())
            .collect()
    };
    assert_eq!(
        members("RaaRestirSurface"),
        [
            "position",
            "normal",
            "view",
            "roughness",
            "f0",
            "metallic",
            "view_depth",
            "valid"
        ]
    );
    assert_eq!(members("RaaRestirRectSample"), ["uv", "solid_angle"]);
    assert!(
        source.contains("textureSampleLevel(raa_restir_ltc, raa_linear_clamp,"),
        "the LTC tables are read from binding 18 with the clamp sampler at an explicit level"
    );
}

#[test]
fn restir_common_functions_never_read_group_0() {
    let reflection = Reflection::new(&restir_common_module_wgsl());
    for (_, function) in reflection.module.functions.iter() {
        let Some(name) = function.name.as_deref() else {
            continue;
        };
        if !name.starts_with("raa_restir_") {
            continue;
        }
        assert!(
            function_bindings(&reflection, name, 0).is_empty(),
            "fn {name} reads group 0"
        );
        let group1 = function_bindings(&reflection, name, 1);
        assert!(
            group1
                .iter()
                .all(|binding| GI_COMPUTE_GROUP1.contains(binding)),
            "fn {name} group-1 bindings {group1:?}"
        );
    }
    assert_eq!(
        function_bindings(&reflection, "raa_restir_ltc_luminance", 1),
        [6],
        "the LTC luminance samples with raa_linear_clamp"
    );
    assert_eq!(
        function_bindings(&reflection, "raa_restir_ltc_luminance", 2),
        [18],
        "the LTC luminance reads the part's own atlas"
    );
    assert_eq!(
        function_bindings(&reflection, "raa_restir_slot_light", 1),
        [12]
    );
    assert_eq!(
        function_bindings(&reflection, "raa_restir_slot_light", 2),
        [8]
    );
    assert_eq!(
        function_bindings(&reflection, "raa_restir_surface", 2),
        [0, 1, 2, 19],
        "the surface reads GiFrameGpu, the S12 depth, normal and material"
    );
    for name in [
        "raa_restir_light_point",
        "raa_restir_representative",
        "raa_restir_sample_rect",
        "raa_restir_area_pdf",
        "raa_restir_target",
    ] {
        for group in 0..3 {
            assert!(
                function_bindings(&reflection, name, group).is_empty(),
                "fn {name} is pure (group {group})"
            );
        }
    }
}

#[test]
fn restir_share2_compute_caller_stays_inside_its_groups() {
    let source = format!("{}{RESTIR_SHARE2_CALLER}", restir_common_module_wgsl());
    let reflection = Reflection::new(&source);
    let index = reflection.entry("raa_restir_share2_caller");
    let entry = &reflection.module.entry_points[index];
    assert_eq!(entry.stage, ShaderStage::Compute);
    assert_eq!(entry.workgroup_size, [WORKGROUP, WORKGROUP, 1]);
    assert!(reflection.used(index, 0).is_empty(), "group 0 is unused");
    assert_eq!(reflection.used(index, 1), [6, 12], "group-1 bindings");
    assert_eq!(
        reflection.used(index, 2),
        [0, 1, 2, 8, 16, 18, 19],
        "group-2 bindings"
    );
    let registry = numbers(&restir_group2_entries());
    for binding in reflection.used(index, 2) {
        assert!(registry.contains(&binding), "group 2 binding {binding}");
    }
    let (textures, buffers) = reflection.storage(index);
    assert!(textures <= 4, "{textures} storage textures");
    assert!(buffers <= 8, "{buffers} storage buffers");
}

#[test]
fn restir_roughness_floor_matches_its_cpu_twin() {
    let reflection = Reflection::new(&restir_common_module_wgsl());
    let (_, constant) = reflection
        .module
        .constants
        .iter()
        .find(|(_, constant)| constant.name.as_deref() == Some("RAA_RESTIR_ROUGHNESS_MIN"))
        .expect("RAA_RESTIR_ROUGHNESS_MIN is declared");
    let naga::Expression::Literal(naga::Literal::F32(value)) =
        reflection.module.global_expressions[constant.init]
    else {
        panic!("RAA_RESTIR_ROUGHNESS_MIN is an f32 literal");
    };
    assert_eq!(value, ROUGHNESS_MIN);
}

/// The ReSTIR initial-candidates module (contract 'area-restir-wgsl') composed by
/// crates/raa-gi/src/parts/area_shadow/restir.rs `restir_initial_module_wgsl` over the trace
/// module and the common module of the Handoff doc comment at restir.rs
/// `restir_common_module_wgsl`.
#[test]
fn restir_initial_module_validates_and_holds_its_kernel() {
    let source = restir_initial_module_wgsl();
    let reflection = Reflection::new(&source);
    let computes: Vec<&str> = reflection
        .module
        .entry_points
        .iter()
        .map(|entry| entry.name.as_str())
        .collect();
    assert_eq!(computes, [AREA_TRACE_ENTRY, RESTIR_INITIAL_ENTRY]);
    let marker = format!("@compute @workgroup_size(8, 8)\nfn {RESTIR_INITIAL_ENTRY}(");
    assert!(source.contains(&marker), "the entry follows its workgroup");
    let index = reflection.entry(RESTIR_INITIAL_ENTRY);
    let entry = &reflection.module.entry_points[index];
    assert_eq!(entry.stage, ShaderStage::Compute);
    assert_eq!(entry.workgroup_size, [WORKGROUP, WORKGROUP, 1]);
    for name in [
        "raa_restir_surface(",
        "raa_restir_ltc_luminance(",
        "raa_restir_sample_rect(",
        "raa_restir_area_pdf(",
        "raa_restir_target(",
        "raa_restir_wrs_accept(",
        "raa_restir_contribution_weight(",
        "raa_area_ray_visibility(surface.position, surface.normal, point)",
        "raa_restir_current[index] = raa_restir_pack(",
    ] {
        assert!(source.contains(name), "the initial module calls {name}");
    }
}

#[test]
fn restir_initial_bindings_match_the_restir_layout() {
    let reflection = Reflection::new(&restir_initial_module_wgsl());
    let index = reflection.entry(RESTIR_INITIAL_ENTRY);
    let entries = restir_group2_entries();
    assert_eq!(
        reflection.used(index, 2),
        [0, 1, 2, 4, 6, 8, 10, 11, 15, 16, 18, 19],
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
    let used1 = reflection.used(index, 1);
    assert!(
        used1
            .iter()
            .all(|binding| GI_COMPUTE_GROUP1.contains(binding)),
        "group-1 bindings {used1:?}"
    );
    assert!(reflection.used(index, 0).is_empty(), "group 0 is unused");
    let (textures, buffers) = reflection.storage(index);
    assert!(textures <= 4, "{textures} storage textures");
    assert!(buffers <= 8, "{buffers} storage buffers");
}

/// The ReSTIR temporal-reuse module (contract 'area-restir-wgsl') composed by
/// crates/raa-gi/src/parts/area_shadow/restir.rs `restir_temporal_module_wgsl`; its surface and
/// target calls are the Handoff comment of share 2 at restir_common.wgsl:336
/// `raa_restir_surface`.
#[test]
fn restir_temporal_module_validates_and_holds_its_kernel() {
    let source = restir_temporal_module_wgsl();
    let reflection = Reflection::new(&source);
    let computes: Vec<&str> = reflection
        .module
        .entry_points
        .iter()
        .map(|entry| entry.name.as_str())
        .collect();
    assert_eq!(computes, [AREA_TRACE_ENTRY, RESTIR_TEMPORAL_ENTRY]);
    let marker = format!("@compute @workgroup_size(8, 8)\nfn {RESTIR_TEMPORAL_ENTRY}(");
    assert!(source.contains(&marker), "the entry follows its workgroup");
    let index = reflection.entry(RESTIR_TEMPORAL_ENTRY);
    let entry = &reflection.module.entry_points[index];
    assert_eq!(entry.stage, ShaderStage::Compute);
    assert_eq!(entry.workgroup_size, [WORKGROUP, WORKGROUP, 1]);
    assert_eq!(
        source.matches("workgroupBarrier();").count(),
        2,
        "two barriers"
    );
    let body = &source[source.find(&marker).unwrap_or(0)..];
    assert!(
        !body.contains("return"),
        "every invocation reaches both barriers"
    );
    for name in [
        "raa_restir_unpack(raa_restir_current[",
        "raa_restir_surface(",
        "raa_temporal_surface_uv(uv, textureLoad(raa_restir_motion,",
        "raa_restir_history[raa_restir_index(",
        "raa_restir_depth_similar(",
        "raa_restir_params.normal_cos",
        "raa_restir_remap(t.slot)",
        "raa_restir_clamp_confidence(",
        "raa_restir_target(",
        "raa_restir_wrs_accept(",
        "raa_restir_contribution_weight(",
        "var<workgroup>",
        "raa_restir_boiling_keep(",
        "raa_restir_current[raa_restir_index(id.xy)] = raa_restir_pack(",
    ] {
        assert!(source.contains(name), "the temporal module calls {name}");
    }
}

#[test]
fn restir_temporal_bindings_match_the_restir_layout() {
    let reflection = Reflection::new(&restir_temporal_module_wgsl());
    let index = reflection.entry(RESTIR_TEMPORAL_ENTRY);
    let entries = restir_group2_entries();
    let used = reflection.used(index, 2);
    for binding in [0, 1, 2, 8, 12, 13, 14, 15, 16, 17, 19] {
        assert!(
            used.contains(&binding),
            "group 2 binding {binding} in {used:?}"
        );
    }
    for binding in [5, 6] {
        assert!(!used.contains(&binding), "group 2 binding {binding} unused");
    }
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
    let used1 = reflection.used(index, 1);
    assert!(
        used1
            .iter()
            .all(|binding| GI_COMPUTE_GROUP1.contains(binding)),
        "group-1 bindings {used1:?}"
    );
    assert!(reflection.used(index, 0).is_empty(), "group 0 is unused");
    let (textures, buffers) = reflection.storage(index);
    assert!(textures <= 4, "{textures} storage textures");
    assert!(buffers <= 8, "{buffers} storage buffers");
}

/// The ReSTIR spatial-reuse and resolve module (contract 'area-restir-wgsl') composed by
/// crates/raa-gi/src/parts/area_shadow/restir.rs `restir_spatial_module_wgsl`; its light point,
/// slot lookup, target and LTC luminance calls are the Handoff comment of share 2 at
/// restir_common.wgsl:391 `raa_restir_light_point` (`raa_restir_ltc_luminance` at :586).
#[test]
fn restir_spatial_module_validates_and_holds_its_kernel() {
    let source = restir_spatial_module_wgsl();
    let reflection = Reflection::new(&source);
    let computes: Vec<&str> = reflection
        .module
        .entry_points
        .iter()
        .map(|entry| entry.name.as_str())
        .collect();
    assert_eq!(computes, [AREA_TRACE_ENTRY, RESTIR_SPATIAL_ENTRY]);
    let marker = format!("@compute @workgroup_size(8, 8)\nfn {RESTIR_SPATIAL_ENTRY}(");
    assert!(source.contains(&marker), "the entry follows its workgroup");
    let index = reflection.entry(RESTIR_SPATIAL_ENTRY);
    let entry = &reflection.module.entry_points[index];
    assert_eq!(entry.stage, ShaderStage::Compute);
    assert_eq!(entry.workgroup_size, [WORKGROUP, WORKGROUP, 1]);
    for name in [
        "raa_restir_unpack(raa_restir_current[index])",
        "raa_restir_spatial_offset(pixel, frame, k, raa_restir_params.spatial_radius)",
        "raa_restir_depth_similar(surface_q.view_depth, surface_p.view_depth)",
        "raa_restir_params.normal_cos",
        "raa_restir_target(surfaces[j], light, r.uv)",
        "raa_restir_wrs_accept(",
        "raa_restir_contribution_weight(w_sum, kept_target)",
        "raa_restir_history[index] = raa_restir_pack(merged)",
        "raa_restir_light_point(",
        "raa_area_ray_visibility(surface_p.position, surface_p.normal, point)",
        "s = kept_target * merged.weight * v",
        "vec4<f32>(s, u_tilde, s, u_tilde)",
    ] {
        assert!(source.contains(name), "the spatial module calls {name}");
    }
    assert!(
        source.contains("raa_restir_ltc_luminance(surface_p, raa_local_lights["),
        "the resolve sums the unshadowed LTC luminance U~"
    );
}

#[test]
fn restir_spatial_bindings_match_the_restir_layout() {
    let reflection = Reflection::new(&restir_spatial_module_wgsl());
    let index = reflection.entry(RESTIR_SPATIAL_ENTRY);
    let entries = restir_group2_entries();
    let used = reflection.used(index, 2);
    for binding in [0, 1, 2, 5, 8, 15, 16, 17, 18, 19] {
        assert!(
            used.contains(&binding),
            "group 2 binding {binding} in {used:?}"
        );
    }
    for binding in [6, 14] {
        assert!(!used.contains(&binding), "group 2 binding {binding} unused");
    }
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
    let used1 = reflection.used(index, 1);
    assert!(
        used1
            .iter()
            .all(|binding| GI_COMPUTE_GROUP1.contains(binding)),
        "group-1 bindings {used1:?}"
    );
    assert!(reflection.used(index, 0).is_empty(), "group 0 is unused");
    let (textures, buffers) = reflection.storage(index);
    assert!(textures <= 4, "{textures} storage textures");
    assert!(buffers <= 8, "{buffers} storage buffers");
}

/// The value of every float literal in `source` (WGSL comments stripped), in source order:
/// decimal literals (`1.5`, `.5`, `2e-3`, `1f`) and hex floats (`0x1.fffffep+127f`). Integer
/// literals (no fraction, exponent or `f`/`h` suffix) are skipped.
fn wgsl_float_literals(source: &str) -> Vec<(String, f64)> {
    let mut code = String::with_capacity(source.len());
    let mut chars = source.chars().peekable();
    let mut depth = 0u32;
    while let Some(c) = chars.next() {
        let next = chars.peek().copied();
        if depth == 0 && c == '/' && next == Some('/') {
            for skipped in chars.by_ref() {
                if skipped == '\n' {
                    break;
                }
            }
            code.push('\n');
        } else if c == '/' && next == Some('*') {
            chars.next();
            depth += 1;
        } else if depth > 0 && c == '*' && next == Some('/') {
            chars.next();
            depth -= 1;
            code.push(' ');
        } else if depth == 0 {
            code.push(c);
        }
    }

    let bytes = code.as_bytes();
    let ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let digits = |from: usize, hex: bool| {
        let mut end = from;
        while end < bytes.len()
            && (bytes[end].is_ascii_digit() || (hex && bytes[end].is_ascii_hexdigit()))
        {
            end += 1;
        }
        end
    };
    let mut found = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let starts = bytes[at].is_ascii_digit()
            || (bytes[at] == b'.' && bytes.get(at + 1).is_some_and(u8::is_ascii_digit));
        if !starts || (at > 0 && ident(bytes[at - 1])) {
            at += 1;
            continue;
        }
        let start = at;
        let hex = bytes[at] == b'0' && matches!(bytes.get(at + 1), Some(b'x' | b'X'));
        let body = if hex { at + 2 } else { at };
        let mut end = digits(body, hex);
        let whole_end = end;
        let mut fraction = None;
        if bytes.get(end) == Some(&b'.') {
            let fraction_end = digits(end + 1, hex);
            fraction = Some((end + 1, fraction_end));
            end = fraction_end;
        }
        let marker: &[u8] = if hex { b"pP" } else { b"eE" };
        let mut exponent = None;
        if bytes.get(end).is_some_and(|b| marker.contains(b)) {
            let sign = end + 1 + usize::from(matches!(bytes.get(end + 1), Some(b'+' | b'-')));
            let exponent_end = digits(sign, false);
            if exponent_end > sign {
                exponent = Some((end + 1, exponent_end));
                end = exponent_end;
            }
        }
        let suffix = bytes.get(end).copied();
        let float_suffix = matches!(suffix, Some(b'f' | b'h'));
        if matches!(suffix, Some(b'f' | b'h' | b'i' | b'u')) {
            end += 1;
        }
        if fraction.is_some() || exponent.is_some() || float_suffix {
            let text = &code[start..end];
            let value = if hex {
                let mut mantissa = 0.0f64;
                let mut scale = 0i32;
                let fraction_digits = fraction.map_or(body..body, |(from, to)| from..to);
                for index in (body..whole_end).chain(fraction_digits.clone()) {
                    let digit = (bytes[index] as char).to_digit(16).unwrap_or(0);
                    mantissa = mantissa * 16.0 + f64::from(digit);
                }
                scale -= 4 * fraction_digits.len() as i32;
                if let Some((from, to)) = exponent {
                    scale += code[from..to].parse::<i32>().unwrap_or(0);
                }
                mantissa * 2f64.powi(scale)
            } else {
                let number = if float_suffix {
                    &text[..text.len() - 1]
                } else {
                    text
                };
                number
                    .parse::<f64>()
                    .unwrap_or_else(|e| panic!("float literal {text}: {e}"))
            };
            found.push((text.to_string(), value));
        }
        at = end.max(at + 1);
    }
    found
}

/// Regression of run 14797 (p4-gi-area-restir-render): Chrome's Tint refused
/// `const RAA_RESTIR_F32_MAX: f32 = 3.40282347e38;` ("cannot be represented as 'f32'": the
/// decimal lies above the largest finite f32, which naga silently rounds down), so the three
/// ReSTIR shader modules, their pipelines and every frame's command buffer were invalid and
/// the frame stayed black. Every float literal of the composed ReSTIR modules (the exact text
/// the page compiles) lies inside the finite f32 range Tint accepts, and naga evaluates
/// RAA_RESTIR_F32_MAX (written as the exact hex float) to f32::MAX.
#[test]
fn restir_float_literals_stay_inside_the_f32_range_tint_accepts() {
    let modules = [
        ("restir_common", restir_common_module_wgsl()),
        ("restir_initial", restir_initial_module_wgsl()),
        ("restir_temporal", restir_temporal_module_wgsl()),
        ("restir_spatial", restir_spatial_module_wgsl()),
    ];
    let limit = f64::from(f32::MAX);
    for (name, source) in &modules {
        let literals = wgsl_float_literals(source);
        assert!(literals.len() > 10, "{name}: float literals are found");
        let outside: Vec<&(String, f64)> = literals
            .iter()
            .filter(|(_, value)| !value.is_finite() || value.abs() > limit)
            .collect();
        assert!(
            outside.is_empty(),
            "{name}: float literals outside the finite f32 range (Tint refuses them): {outside:?}"
        );
        assert!(
            literals
                .iter()
                .any(|(text, value)| text == "0x1.fffffep+127f" && *value == limit),
            "{name}: RAA_RESTIR_F32_MAX is the exact hex float of f32::MAX"
        );
    }
    assert_eq!(
        wgsl_float_literals("// 9e99\nlet a = vec3(3.40282347e38, .5, 2e-3f) + 7u + x2e5;"),
        [
            ("3.40282347e38".to_string(), 3.40282347e38),
            (".5".to_string(), 0.5),
            ("2e-3f".to_string(), 2e-3),
        ],
        "the scanner reads decimal literals, skips comments, integers and identifiers"
    );

    let reflection = Reflection::new(&restir_common_module_wgsl());
    let (_, constant) = reflection
        .module
        .constants
        .iter()
        .find(|(_, constant)| constant.name.as_deref() == Some("RAA_RESTIR_F32_MAX"))
        .unwrap_or_else(|| panic!("RAA_RESTIR_F32_MAX is declared"));
    assert!(
        matches!(
            reflection.module.global_expressions[constant.init],
            naga::Expression::Literal(naga::Literal::F32(value)) if value == f32::MAX
        ),
        "RAA_RESTIR_F32_MAX evaluates to f32::MAX"
    );
}
