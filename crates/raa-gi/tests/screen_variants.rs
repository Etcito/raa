//! P4-Q1/S39: every screen module naga-validates, its entries follow the 8x8 compute
//! workgroup, its group-2 layouts match the bindings the WGSL entries use, its records match
//! the struct offsets of contract 'gi-screen-wgsl', and the lit, transparent, fur and vg.shade
//! compositions with the three gi slot providers validate within the 16 sampled-texture budget.
//!
//! The builders, entry constants and group-2 entry functions come from the Handoff doc comment
//! at crates/raa-gi/src/parts/screen/passes.rs:100 `probe_trace_module_wgsl`; the providers come
//! from crates/raa-gi/src/parts/slots.rs:28 `indirect_provider` (with `ao_provider` and
//! `local_visibility_provider`).

#![cfg(not(target_arch = "wasm32"))]

// `default_slot_providers` is crate-private to raa-lit; the test compiles the same source file,
// as raa-oit's and raa-fur's variants tests do.
#[allow(dead_code)]
#[path = "../../raa-lit/src/slots.rs"]
mod slots;

use naga::{AddressSpace, ImageClass, ShaderStage, TypeInner};
use raa_core::wgpu::naga;
use raa_gi::parts::screen::passes::{
    PROBE_FILTER_ENTRY, PROBE_INTEGRATE_ENTRY, PROBE_PROJECT_ENTRY, PROBE_TRACE_ENTRY,
    PYRAMID_COPY_ENTRY, PYRAMID_REDUCE_ENTRY, depth_pyramid_module_wgsl, filter_group2_entries,
    integrate_group2_entries, probe_integrate_module_wgsl, probe_project_module_wgsl,
    probe_trace_module_wgsl, project_group2_entries, pyramid_copy_group2_entries,
    pyramid_reduce_group2_entries, trace_group2_entries,
};
use raa_gi::parts::slots::{ao_provider, indirect_provider, local_visibility_provider};
use raa_shader::{Composer, MaterialFnId, MaterialFnRegistry, PassKind, VariantKey};

const PROBE_RAYS_LINE: &str = "override PROBE_RAYS: u32 = 8u;";

fn numbers(entries: &[raa_core::wgpu::BindGroupLayoutEntry]) -> Vec<u32> {
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

    /// The binding numbers of `group` that entry `index` statically uses, sorted.
    fn used(&self, index: usize, group: u32) -> Vec<u32> {
        let usage = self.info.get_entry_point(index);
        let mut out = Vec::new();
        for (handle, global) in self.module.global_variables.iter() {
            let Some(binding) = &global.binding else {
                continue;
            };
            if binding.group == group && !usage[handle].is_empty() {
                out.push(binding.binding);
            }
        }
        out.sort_unstable();
        out
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

    /// Sampled and depth image globals the fragment entry uses.
    fn fragment_textures(&self) -> u32 {
        let index = self
            .module
            .entry_points
            .iter()
            .position(|entry| entry.stage == ShaderStage::Fragment)
            .expect("fragment entry");
        let usage = self.info.get_entry_point(index);
        let mut count = 0;
        for (handle, global) in self.module.global_variables.iter() {
            if usage[handle].is_empty() {
                continue;
            }
            if let TypeInner::Image { class, .. } = &self.module.types[global.ty].inner
                && !matches!(class, ImageClass::Storage { .. })
            {
                count += 1;
            }
        }
        count
    }
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
fn screen_modules_validate_at_both_probe_ray_options() {
    let pyramid = depth_pyramid_module_wgsl();
    assert!(!pyramid.contains("override PROBE_RAYS"), "pyramid module");
    if let Err(e) = Composer::validate(&pyramid) {
        panic!("pyramid: {e}");
    }
    for (label, source) in [
        ("trace", probe_trace_module_wgsl()),
        ("project", probe_project_module_wgsl()),
        ("integrate", probe_integrate_module_wgsl()),
    ] {
        assert_eq!(source.matches(PROBE_RAYS_LINE).count(), 1, "{label} line");
        if let Err(e) = Composer::validate(&source) {
            panic!("{label} at 8: {e}");
        }
        let six = source.replace(PROBE_RAYS_LINE, "override PROBE_RAYS: u32 = 6u;");
        if let Err(e) = Composer::validate(&six) {
            panic!("{label} at 6: {e}");
        }
    }
}

#[test]
fn entries_follow_the_compute_workgroup() {
    let cases: [(String, &[&str]); 4] = [
        (
            depth_pyramid_module_wgsl(),
            &[PYRAMID_COPY_ENTRY, PYRAMID_REDUCE_ENTRY],
        ),
        (probe_trace_module_wgsl(), &[PROBE_TRACE_ENTRY]),
        (
            probe_project_module_wgsl(),
            &[PROBE_PROJECT_ENTRY, PROBE_FILTER_ENTRY],
        ),
        (probe_integrate_module_wgsl(), &[PROBE_INTEGRATE_ENTRY]),
    ];
    for (source, names) in cases {
        for name in names {
            let marker = format!("@compute @workgroup_size(8, 8)\nfn {name}(");
            assert!(source.contains(&marker), "{name} follows its workgroup");
        }
        let reflection = Reflection::new(&source);
        let computes: Vec<&str> = reflection
            .module
            .entry_points
            .iter()
            .filter(|entry| entry.stage == ShaderStage::Compute)
            .map(|entry| entry.name.as_str())
            .collect();
        assert_eq!(computes.as_slice(), names);
    }
}

/// A composed module and the group-2 binding numbers its entries use.
type LayoutCase = (String, Vec<(&'static str, Vec<u32>)>);

#[test]
fn entry_group2_bindings_match_their_layouts() {
    let cases: [LayoutCase; 4] = [
        (
            depth_pyramid_module_wgsl(),
            vec![
                (PYRAMID_COPY_ENTRY, numbers(&pyramid_copy_group2_entries())),
                (
                    PYRAMID_REDUCE_ENTRY,
                    numbers(&pyramid_reduce_group2_entries()),
                ),
            ],
        ),
        (
            probe_trace_module_wgsl(),
            vec![(PROBE_TRACE_ENTRY, numbers(&trace_group2_entries()))],
        ),
        (
            probe_project_module_wgsl(),
            vec![
                (PROBE_PROJECT_ENTRY, numbers(&project_group2_entries())),
                (PROBE_FILTER_ENTRY, numbers(&filter_group2_entries())),
            ],
        ),
        (
            probe_integrate_module_wgsl(),
            vec![(PROBE_INTEGRATE_ENTRY, numbers(&integrate_group2_entries()))],
        ),
    ];
    let group1 = [3, 6, 12, 15, 16, 17, 18];
    for (source, entries) in cases {
        let reflection = Reflection::new(&source);
        for (name, mut expected) in entries {
            let index = reflection.entry(name);
            expected.sort_unstable();
            assert_eq!(
                reflection.used(index, 2),
                expected,
                "{name} group-2 bindings"
            );
            let (textures, buffers) = reflection.storage(index);
            assert!(textures <= 4, "{name}: {textures} storage textures");
            assert!(buffers <= 8, "{name}: {buffers} storage buffers");
            let used1 = reflection.used(index, 1);
            assert!(
                used1.iter().all(|binding| group1.contains(binding)),
                "{name}: group-1 bindings {used1:?}"
            );
            assert!(reflection.used(index, 0).is_empty(), "{name}: group 0");
        }
    }
}

#[test]
fn trace_records_match_contract_gi_screen_wgsl() {
    let reflection = Reflection::new(&probe_trace_module_wgsl());
    let frame = [
        ("view_proj", 0),
        ("inv_view_proj", 64),
        ("inv_proj", 128),
        ("inv_view", 192),
        ("previous_view_proj", 256),
        ("camera_position", 320),
        ("sun_direction", 336),
        ("sun_color", 352),
        ("viewport", 368),
        ("counts", 384),
        ("settings", 400),
        ("flags", 416),
    ];
    check_struct(&reflection.module, "GiFrameGpu", 432, &frame);
    let probe = [("sh", 0), ("position", 144), ("normal", 160)];
    check_struct(&reflection.module, "Probe", 176, &probe);
    let screen = [
        ("history_valid", 0),
        ("screen_traces", 4),
        ("parity", 8),
        ("pad", 12),
    ];
    check_struct(&reflection.module, "GiScreenGpu", 16, &screen);
}

fn composer() -> Composer {
    let mut composer = Composer::new();
    let providers = slots::default_slot_providers().into_iter().chain([
        indirect_provider(),
        ao_provider(),
        local_visibility_provider(),
    ]);
    for provider in providers {
        composer
            .add_slot(provider)
            .unwrap_or_else(|e| panic!("{e}"));
    }
    composer
}

#[test]
fn gi_compositions_validate_within_the_fragment_texture_budget() {
    let composer = composer();
    let registry = MaterialFnRegistry::new();
    let cases = [
        ("forward", PassKind::Forward, 4, 0),
        ("transparent", PassKind::Transparent, 1, 0),
        ("fur", PassKind::FurShell, 1, raa_lit::FLAG_FUR),
        ("vg.shade", PassKind::VgShade, 1, 0),
    ];
    for (label, pass, msaa, flags) in cases {
        let key = VariantKey {
            material_fn: MaterialFnId::STANDARD,
            pass,
            msaa,
            flags,
        };
        let wgsl = composer
            .compose(&registry, &key)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        if let Err(e) = Composer::validate(&wgsl) {
            panic!("{label}: {e}");
        }
        let textures = Reflection::new(&wgsl).fragment_textures();
        assert!(textures <= 16, "{label}: {textures} sampled textures");
    }
}

#[test]
fn only_local_visibility_ships_the_volume_trace() {
    for provider in [indirect_provider(), ao_provider()] {
        assert!(!provider.wgsl.contains("fn raa_gi_trace("));
    }
    let text = local_visibility_provider().wgsl;
    assert!(text.contains("fn raa_gi_trace("));
    for token in [
        "textureSample(",
        "textureSampleBias(",
        "textureSampleCompare(",
        "dpdx",
        "dpdy",
        "fwidth",
    ] {
        assert!(!text.contains(token), "LocalVisibility contains {token}");
    }
}
