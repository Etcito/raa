//! P4-Q1/M8: the SSGI trace module naga-validates, its entry follows the 8x8 compute workgroup,
//! its group-2 bindings match `ssgi_trace_group2_entries`, group 1 stays inside the GI compute
//! set, it writes one storage texture, and `SsgiParamsGpu` is `SSGI_PARAMS_SIZE` bytes
//! (contract 'ssgi-trace-wgsl').

#![cfg(not(target_arch = "wasm32"))]

use naga::{AddressSpace, ImageClass, ShaderStage, TypeInner};
use raa_core::wgpu::naga;
use raa_gi::parts::ssgi::passes::{
    SSGI_PARAMS_SIZE, SSGI_TRACE_ENTRY, WORKGROUP, ssgi_trace_group2_entries,
    ssgi_trace_module_wgsl,
};

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
        let mut out: Vec<u32> = self
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

    /// The storage textures entry `index` uses.
    fn storage_textures(&self, index: usize) -> usize {
        let usage = self.info.get_entry_point(index);
        self.module
            .global_variables
            .iter()
            .filter(|(handle, global)| {
                !usage[*handle].is_empty()
                    && global.space == AddressSpace::Handle
                    && matches!(
                        self.module.types[global.ty].inner,
                        TypeInner::Image {
                            class: ImageClass::Storage { .. },
                            ..
                        }
                    )
            })
            .count()
    }
}

#[test]
fn trace_module_validates_with_its_bindings() {
    let reflection = Reflection::new(&ssgi_trace_module_wgsl());
    let index = reflection.entry(SSGI_TRACE_ENTRY);
    assert_eq!(SSGI_TRACE_ENTRY, "raa_gi_ssgi_trace");
    let entry = &reflection.module.entry_points[index];
    assert_eq!(entry.stage, ShaderStage::Compute);
    assert_eq!(entry.workgroup_size, [WORKGROUP, WORKGROUP, 1]);
    assert_eq!(entry.workgroup_size, [8, 8, 1]);

    let numbers: Vec<u32> = ssgi_trace_group2_entries()
        .iter()
        .map(|entry| entry.binding)
        .collect();
    assert_eq!(numbers, [0, 1, 2, 3, 5, 8]);
    assert_eq!(reflection.used(index, 2), numbers);

    let group1 = reflection.used(index, 1);
    let compute_set = [3, 6, 12, 15, 16, 17, 18];
    assert!(
        group1.iter().all(|binding| compute_set.contains(binding)),
        "group 1 {group1:?} outside {compute_set:?}"
    );
    assert!(reflection.used(index, 0).is_empty());
    assert_eq!(reflection.storage_textures(index), 1);
}

#[test]
fn ssgi_params_struct_is_32_bytes() {
    let reflection = Reflection::new(&ssgi_trace_module_wgsl());
    let module = &reflection.module;
    let (handle, _) = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some("SsgiParamsGpu"))
        .unwrap_or_else(|| panic!("SsgiParamsGpu is declared"));
    let mut layouter = naga::proc::Layouter::default();
    layouter
        .update(module.to_ctx())
        .unwrap_or_else(|e| panic!("layout: {e}"));
    assert_eq!(u64::from(layouter[handle].size), SSGI_PARAMS_SIZE);
    assert_eq!(SSGI_PARAMS_SIZE, 32);
}
