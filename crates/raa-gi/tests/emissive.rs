//! Contract 'gi-emissive-sources': the flags, the override values, the voxel cells and emissive
//! word index of a hit, the radiance without emission, the emission signature and the
//! voxel-light override. Expected numbers are derived by hand.

use raa_gi::EmissiveSources;
use raa_gi::emissive::{
    emission_signature, emissive_word_index, hit_cells, override_value, without_emission,
};

#[test]
fn all_sources_are_on_and_override_values_are_one_and_zero() {
    let all = EmissiveSources::ALL;
    assert!(all.volume && all.screen && all.ssgi);
    assert_eq!(override_value(true), 1.0);
    assert_eq!(override_value(false), 0.0);
}

#[test]
fn hit_cells_surround_the_point() {
    // (1.2 / 0.5 - 0.5, 0.3 / 0.5 - 0.5, 1.9 / 0.5 - 0.5) = (1.9, 0.1, 3.3): base [1, 0, 3].
    let cells = hit_cells([0.0, 0.0, 0.0], 0.5, [1.2, 0.3, 1.9]);
    assert_eq!(cells[0], [1, 0, 3]);
    assert_eq!(cells[7], [2, 1, 4]);
}

#[test]
fn emissive_word_index_is_word_two_of_the_voxel() {
    // (1 * 64 + 3 * 16 + 0 * 4 + 1) * 3 + 2 = 341.
    assert_eq!(emissive_word_index(4, 1, [1, 0, 3]), Some(341));
    assert_eq!(emissive_word_index(4, 1, [2, 1, 4]), None);
    assert_eq!(emissive_word_index(4, 0, [-1, 0, 0]), None);
}

#[test]
fn without_emission_subtracts_only_when_off() {
    assert_eq!(
        without_emission([2.0, 2.0, 2.0], [1.5, 3.0, 0.0], false),
        [0.5, 0.0, 2.0]
    );
    assert_eq!(
        without_emission([2.0, 2.0, 2.0], [1.5, 3.0, 0.0], true),
        [2.0, 2.0, 2.0]
    );
}

#[test]
fn emission_signature_follows_the_emission_and_the_volume_flag() {
    let input = [[4.0, 3.0, 1.5], [0.0, 0.0, 0.0]];
    let base = emission_signature(input, true);
    assert_eq!(base, emission_signature(input, true));
    assert_ne!(
        base,
        emission_signature([[4.0, 3.0, 1.0], [0.0, 0.0, 0.0]], true)
    );
    assert_ne!(base, emission_signature(input, false));
}

#[test]
fn voxel_light_declares_and_applies_the_emissive_override() {
    let wgsl = raa_gi::shaders::voxel_light_module_wgsl();
    assert!(wgsl.contains("override RAA_GI_EMISSIVE: f32 = 1.0;"));
    assert!(wgsl.contains("emissive * RAA_GI_EMISSIVE"));
    raa_shader::Composer::validate(&wgsl).expect("voxel_light validates");
}
