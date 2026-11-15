//! Contract `taa-bindings`: the TAA resolve module validates, defines the single `cs_taa`
//! compute entry at 8x8, binds exactly group 0 entries 0-8 with the contract types, and
//! reads the shared reprojection helpers instead of the dropped reactive mask.

use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::{
    AddressSpace, ImageClass, ImageDimension, ScalarKind, ShaderStage, StorageAccess,
    StorageFormat, TypeInner,
};
use raa_shader::Composer;
use raa_taa::{TAA_ENTRY, TAA_PARAMS_SIZE, TAA_WGSL, TAA_WORKGROUP, taa_module_source};

fn parsed_module() -> naga::Module {
    let module = naga::front::wgsl::parse_str(&taa_module_source()).expect("the TAA module parses");
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .expect("the TAA module validates");
    module
}

#[test]
fn taa_module_validates_through_the_composer() {
    assert_eq!(Composer::validate(&taa_module_source()), Ok(()));
}

#[test]
fn one_compute_entry_named_cs_taa_at_eight_by_eight() {
    let module = parsed_module();
    assert_eq!(module.entry_points.len(), 1, "exactly one entry point");
    let entry = &module.entry_points[0];
    assert_eq!(entry.name, TAA_ENTRY);
    assert_eq!(entry.name, "cs_taa");
    assert_eq!(entry.stage, ShaderStage::Compute);
    assert_eq!(entry.workgroup_size, [TAA_WORKGROUP, TAA_WORKGROUP, 1]);
    assert_eq!(entry.workgroup_size, [8, 8, 1]);
}

#[test]
fn group_zero_bindings_match_the_contract() {
    let module = parsed_module();
    let mut bindings: Vec<(u32, u32, naga::Handle<naga::GlobalVariable>)> = module
        .global_variables
        .iter()
        .filter_map(|(handle, global)| global.binding.map(|b| (b.group, b.binding, handle)))
        .collect();
    bindings.sort_by_key(|(group, binding, _)| (*group, *binding));
    let points: Vec<(u32, u32)> = bindings
        .iter()
        .map(|(group, binding, _)| (*group, *binding))
        .collect();
    let expected: Vec<(u32, u32)> = (0..=8).map(|binding| (0, binding)).collect();
    assert_eq!(points, expected, "group 0 entries are exactly 0..=8");

    for (_, binding, handle) in bindings {
        let global = &module.global_variables[handle];
        let inner = &module.types[global.ty].inner;
        match binding {
            0 => assert_eq!(global.space, AddressSpace::Uniform),
            1 => {
                assert_eq!(global.space, AddressSpace::Uniform);
                let TypeInner::Struct { span, .. } = *inner else {
                    panic!("binding 1 is not the RaaTaaParams struct");
                };
                assert_eq!(u64::from(span), TAA_PARAMS_SIZE, "RaaTaaParams span");
                assert_eq!(TAA_PARAMS_SIZE, 16);
            }
            2 | 4 | 5 => assert!(
                matches!(
                    inner,
                    TypeInner::Image {
                        dim: ImageDimension::D2,
                        arrayed: false,
                        class: ImageClass::Sampled {
                            kind: ScalarKind::Float,
                            multi: false,
                        },
                    }
                ),
                "binding {binding} is not a sampled D2 f32 image"
            ),
            3 => assert!(
                matches!(
                    inner,
                    TypeInner::Image {
                        dim: ImageDimension::D2,
                        arrayed: false,
                        class: ImageClass::Depth { .. },
                    }
                ),
                "binding 3 is not a D2 depth image"
            ),
            6 => assert!(
                matches!(inner, TypeInner::Sampler { comparison: false }),
                "binding 6 is not a filtering sampler"
            ),
            7 | 8 => assert!(
                matches!(
                    inner,
                    TypeInner::Image {
                        dim: ImageDimension::D2,
                        arrayed: false,
                        class: ImageClass::Storage {
                            format: StorageFormat::Rgba16Float,
                            access: StorageAccess::STORE,
                        },
                    }
                ),
                "binding {binding} is not an Rgba16Float write storage image"
            ),
            other => panic!("unexpected binding {other}"),
        }
    }
}

#[test]
fn resolve_calls_the_shared_helpers_and_drops_the_reactive_mask() {
    for helper in [
        "raa_temporal_surface_uv(",
        "raa_temporal_sky_uv(",
        "raa_temporal_history_usable(",
    ] {
        assert!(TAA_WGSL.contains(helper), "TAA_WGSL is missing `{helper}`");
    }
    assert!(
        !TAA_WGSL.contains("reactive"),
        "TAA_WGSL names a reactive mask"
    );
}
