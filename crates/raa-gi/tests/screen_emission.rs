//! The screen emission gate (contract 'gi-screen-emission-wgsl'): screen_emission.wgsl composed
//! into the probe trace and ReSTIR GI main modules, their group-2 binding 6 and the shader
//! formula that follows the CPU twins `raa_gi::emissive::hit_cells` and `emissive_word_index`.

use raa_core::wgpu::naga;
use raa_core::wgpu::{BindGroupLayoutEntry, BindingType, BufferBindingType};
use raa_gi::parts::screen::passes::{
    PROBE_TRACE_ENTRY, probe_trace_module_wgsl, trace_group2_entries,
};
use raa_gi::parts::screen::restir_gi::{
    RESTIR_GI_MAIN_ENTRY, main_group2_entries, restir_gi_main_module_wgsl,
};
use raa_shader::Composer;

const OVERRIDE_LINE: &str = "override RAA_GI_SCREEN_EMISSIVE: f32 = 1.0;";
const EMISSION_WGSL: &str = include_str!("../src/wgsl/screen/screen_emission.wgsl");

/// The group-`group` bindings entry `name` of `source` statically uses, sorted.
fn used(source: &str, name: &str, group: u32) -> Vec<u32> {
    let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|e| panic!("parse: {e}"));
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("validate: {e}"));
    let index = module
        .entry_points
        .iter()
        .position(|entry| entry.name == name)
        .unwrap_or_else(|| panic!("entry {name}"));
    let usage = info.get_entry_point(index);
    let mut out = Vec::new();
    for (handle, global) in module.global_variables.iter() {
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

fn numbers(entries: &[BindGroupLayoutEntry]) -> Vec<u32> {
    entries.iter().map(|entry| entry.binding).collect()
}

fn assert_read_only_storage_6(entries: &[BindGroupLayoutEntry]) {
    let entry = entries.iter().find(|e| e.binding == 6).expect("binding 6");
    assert!(matches!(
        entry.ty,
        BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: true },
            ..
        }
    ));
}

#[test]
fn modules_compose_the_gate_and_use_binding_6() {
    for (source, entry) in [
        (probe_trace_module_wgsl(), PROBE_TRACE_ENTRY),
        (restir_gi_main_module_wgsl(), RESTIR_GI_MAIN_ENTRY),
    ] {
        if let Err(e) = Composer::validate(&source) {
            panic!("{entry}: {e}");
        }
        assert_eq!(source.matches(OVERRIDE_LINE).count(), 1, "{entry}");
        assert!(
            used(&source, entry, 2).contains(&6),
            "{entry} uses binding 6"
        );
    }
}

#[test]
fn group2_entries_carry_the_voxel_words() {
    let trace = trace_group2_entries();
    assert_eq!(numbers(&trace), [0, 1, 2, 3, 4, 5, 6, 7]);
    assert_read_only_storage_6(&trace);
    let main = main_group2_entries();
    assert_eq!(numbers(&main), [0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12]);
    assert_read_only_storage_6(&main);
}

#[test]
fn shader_follows_the_cpu_twin() {
    assert!(EMISSION_WGSL.contains("floor((world - grid.xyz) / grid.w - 0.5)"));
    assert!(EMISSION_WGSL.contains("* 3u + 2u"));
    assert!(EMISSION_WGSL.contains("rgbm.rgb * rgbm.a * 128.0"));
}
