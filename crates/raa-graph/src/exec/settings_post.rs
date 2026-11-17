//! Settings accessors of the post features (contract `rg-settings-accessors`): Ssr, Fog, Shafts,
//! Oit, Dof, Bloom, Grading and Cel. Each function reads or writes one key of its descriptor. The
//! key tables and the dispatch that calls these functions are in `settings.rs`.
//!
//! DoF `autofocus` (packages/raa/types/m14-camera-effects.d.ts:29, `[number, number] | null`) is
//! not a key: no `ParamValue` variant holds `null`. It is kept out of the key tables, and the live
//! value stays in the descriptor that `write_values` builds from the getter.

use std::ops::RangeInclusive;

use raa_bloom::BloomDesc;
use raa_cel::{CelSettings, Dither};
use raa_core::RaaError;
use raa_dof::DofDesc;
use raa_fog::FogDesc;
use raa_grading::GradingDesc;
use raa_oit::OitSettings;
use raa_shafts::ShaftsDesc;
use raa_ssr::SsrDesc;

use super::settings::{choice, count, expected, f32_value, flag, float, named, u32_value, unknown};
use crate::catalog::ParamValue;

pub(super) fn get_ssr(d: &SsrDesc, key: &str) -> Option<ParamValue> {
    match key {
        "intensity" => Some(f32_value(d.intensity)),
        "maxRoughness" => Some(f32_value(d.max_roughness)),
        "thickness" => Some(f32_value(d.thickness_m)),
        _ => None,
    }
}

pub(super) fn set_ssr(d: &mut SsrDesc, key: &str, value: &ParamValue) -> Result<(), String> {
    match key {
        "intensity" => d.intensity = float(key, value)?,
        "maxRoughness" => d.max_roughness = float(key, value)?,
        "thickness" => d.thickness_m = float(key, value)?,
        _ => return Err(unknown(key)),
    }
    Ok(())
}

pub(super) fn get_fog(d: &FogDesc, key: &str) -> Option<ParamValue> {
    match key {
        "density" => Some(f32_value(d.density)),
        "heightFalloff" => Some(f32_value(d.height_falloff)),
        "baseHeight" => Some(f32_value(d.base_height_m)),
        "start" => Some(f32_value(d.start_m)),
        "color" => Some(vec3_value(d.color)),
        "inscatter" => Some(f32_value(d.inscatter)),
        "aerialStrength" => Some(f32_value(d.aerial_strength)),
        "aerialDistance" => Some(f32_value(d.aerial_distance_m)),
        _ => None,
    }
}

pub(super) fn set_fog(d: &mut FogDesc, key: &str, value: &ParamValue) -> Result<(), String> {
    match key {
        "density" => d.density = float(key, value)?,
        "heightFalloff" => d.height_falloff = float(key, value)?,
        "baseHeight" => d.base_height_m = float(key, value)?,
        "start" => d.start_m = float(key, value)?,
        "color" => d.color = vec3(key, value)?,
        "inscatter" => d.inscatter = float(key, value)?,
        "aerialStrength" => d.aerial_strength = float(key, value)?,
        "aerialDistance" => d.aerial_distance_m = float(key, value)?,
        _ => return Err(unknown(key)),
    }
    Ok(())
}

pub(super) fn get_shafts(d: &ShaftsDesc, key: &str) -> Option<ParamValue> {
    match key {
        "intensity" => Some(f32_value(d.intensity)),
        "density" => Some(f32_value(d.density)),
        "maxDistance" => Some(f32_value(d.max_distance_m)),
        _ => None,
    }
}

pub(super) fn set_shafts(d: &mut ShaftsDesc, key: &str, value: &ParamValue) -> Result<(), String> {
    match key {
        "intensity" => d.intensity = float(key, value)?,
        "density" => d.density = float(key, value)?,
        "maxDistance" => d.max_distance_m = float(key, value)?,
        _ => return Err(unknown(key)),
    }
    Ok(())
}

pub(super) fn get_oit(d: &OitSettings, key: &str) -> Option<ParamValue> {
    match key {
        "maxFragments" => Some(u32_value(d.max_fragments)),
        "poolFactor" => Some(u32_value(d.pool_factor)),
        _ => None,
    }
}

pub(super) fn set_oit(d: &mut OitSettings, key: &str, value: &ParamValue) -> Result<(), String> {
    match key {
        "maxFragments" => d.max_fragments = count(key, value)?,
        "poolFactor" => d.pool_factor = count(key, value)?,
        _ => return Err(unknown(key)),
    }
    Ok(())
}

/// The range checks of `OitSettings::validate` (crates/raa-oit/src/feature.rs:62, ranges at
/// :37 and :39). raa-oit keeps that method private, so the same ranges are checked here: the
/// device-free merge refuses what the handle's `set` refuses, under the same `oit.invalid` code.
pub(super) fn oit_validate(s: &OitSettings) -> Result<(), RaaError> {
    oit_in_range("max_fragments", s.max_fragments, 4..=32)?;
    oit_in_range("pool_factor", s.pool_factor, 2..=32)
}

fn oit_in_range(field: &str, v: u32, range: RangeInclusive<u32>) -> Result<(), RaaError> {
    if range.contains(&v) {
        return Ok(());
    }
    Err(RaaError::FrameGraph {
        code: "oit.invalid",
        message: format!(
            "OitSettings.{field} must be in {}..={} (got {v})",
            range.start(),
            range.end()
        ),
    })
}

/// DoF `focalLength` is the raw `focal_length_mm` field, as the catalog declares it (mm).
pub(super) fn get_dof(d: &DofDesc, key: &str) -> Option<ParamValue> {
    match key {
        "focusDistance" => Some(f32_value(d.focus_distance_m)),
        "fStop" => Some(f32_value(d.f_stop)),
        "focalLength" => Some(f32_value(d.focal_length_mm)),
        "maxCoc" => Some(f32_value(d.max_coc_px)),
        _ => None,
    }
}

pub(super) fn set_dof(d: &mut DofDesc, key: &str, value: &ParamValue) -> Result<(), String> {
    match key {
        "focusDistance" => d.focus_distance_m = float(key, value)?,
        "fStop" => d.f_stop = float(key, value)?,
        "focalLength" => d.focal_length_mm = float(key, value)?,
        "maxCoc" => d.max_coc_px = float(key, value)?,
        _ => return Err(unknown(key)),
    }
    Ok(())
}

pub(super) fn get_bloom(d: &BloomDesc, key: &str) -> Option<ParamValue> {
    match key {
        "strength" => Some(f32_value(d.strength)),
        "radius" => Some(f32_value(d.radius)),
        _ => None,
    }
}

pub(super) fn set_bloom(d: &mut BloomDesc, key: &str, value: &ParamValue) -> Result<(), String> {
    match key {
        "strength" => d.strength = float(key, value)?,
        "radius" => d.radius = float(key, value)?,
        _ => return Err(unknown(key)),
    }
    Ok(())
}

pub(super) fn get_grading(d: &GradingDesc, key: &str) -> Option<ParamValue> {
    match key {
        "temperature" => Some(f32_value(d.temperature_k)),
        "tint" => Some(f32_value(d.tint)),
        "exposure" => Some(f32_value(d.exposure_ev)),
        "contrast" => Some(f32_value(d.contrast)),
        "saturation" => Some(f32_value(d.saturation)),
        "vibrance" => Some(f32_value(d.vibrance)),
        "lift" => Some(vec3_value(d.lift)),
        "gamma" => Some(vec3_value(d.gamma)),
        "gain" => Some(vec3_value(d.gain)),
        "shadowsTint" => Some(vec3_value(d.shadows_tint)),
        "highlightsTint" => Some(vec3_value(d.highlights_tint)),
        "splitBalance" => Some(f32_value(d.split_balance)),
        "vignette" => Some(f32_value(d.vignette)),
        _ => None,
    }
}

pub(super) fn set_grading(
    d: &mut GradingDesc,
    key: &str,
    value: &ParamValue,
) -> Result<(), String> {
    match key {
        "temperature" => d.temperature_k = float(key, value)?,
        "tint" => d.tint = float(key, value)?,
        "exposure" => d.exposure_ev = float(key, value)?,
        "contrast" => d.contrast = float(key, value)?,
        "saturation" => d.saturation = float(key, value)?,
        "vibrance" => d.vibrance = float(key, value)?,
        "lift" => d.lift = vec3(key, value)?,
        "gamma" => d.gamma = vec3(key, value)?,
        "gain" => d.gain = vec3(key, value)?,
        "shadowsTint" => d.shadows_tint = vec3(key, value)?,
        "highlightsTint" => d.highlights_tint = vec3(key, value)?,
        "splitBalance" => d.split_balance = float(key, value)?,
        "vignette" => d.vignette = float(key, value)?,
        _ => return Err(unknown(key)),
    }
    Ok(())
}

/// Writes only the Cel keys. The tone mapping that the JS `setCel` also changes
/// (crates/raa-js/src/api/m27_cel.rs:144) is a different feature and is not written here.
pub(super) fn get_cel(d: &CelSettings, key: &str) -> Option<ParamValue> {
    match key {
        "bands" => Some(u32_value(d.bands)),
        "hues" => Some(u32_value(d.hues)),
        "chroma" => Some(u32_value(d.chroma)),
        "dither" => Some(named(d.dither.name())),
        "ditherAmount" => Some(f32_value(d.dither_amount)),
        "outlineWidth" => Some(f32_value(d.outline_width)),
        "outlineSharpness" => Some(f32_value(d.outline_sharpness)),
        "interiorEdges" => Some(ParamValue::Bool(d.interior_edges)),
        "edgeRetention" => Some(f32_value(d.edge_retention)),
        "worldSpace" => Some(ParamValue::Bool(d.world_space)),
        "worldScale" => Some(f32_value(d.world_scale)),
        "softness" => Some(f32_value(d.softness)),
        "specular" => Some(f32_value(d.specular)),
        "toon" => Some(ParamValue::Bool(d.toon)),
        "palette" => Some(ParamValue::Bool(d.palette)),
        "outlines" => Some(ParamValue::Bool(d.outlines)),
        "ditherOutlines" => Some(ParamValue::Bool(d.dither_outlines)),
        _ => None,
    }
}

pub(super) fn set_cel(d: &mut CelSettings, key: &str, value: &ParamValue) -> Result<(), String> {
    match key {
        "bands" => d.bands = count(key, value)?,
        "hues" => d.hues = count(key, value)?,
        "chroma" => d.chroma = count(key, value)?,
        "dither" => d.dither = choice(key, value, |text| Dither::from_name(text).ok())?,
        "ditherAmount" => d.dither_amount = float(key, value)?,
        "outlineWidth" => d.outline_width = float(key, value)?,
        "outlineSharpness" => d.outline_sharpness = float(key, value)?,
        "interiorEdges" => d.interior_edges = flag(key, value)?,
        "edgeRetention" => d.edge_retention = float(key, value)?,
        "worldSpace" => d.world_space = flag(key, value)?,
        "worldScale" => d.world_scale = float(key, value)?,
        "softness" => d.softness = float(key, value)?,
        "specular" => d.specular = float(key, value)?,
        "toon" => d.toon = flag(key, value)?,
        "palette" => d.palette = flag(key, value)?,
        "outlines" => d.outlines = flag(key, value)?,
        "ditherOutlines" => d.dither_outlines = flag(key, value)?,
        _ => return Err(unknown(key)),
    }
    Ok(())
}

/// A colour key as the `vec3` `ParamValue`, each component widened from the f32 field.
fn vec3_value(v: [f32; 3]) -> ParamValue {
    ParamValue::Vec3(v.map(f64::from))
}

/// A `vec3` value narrowed to the f32 field; any other kind is refused under its key.
fn vec3(key: &str, value: &ParamValue) -> Result<[f32; 3], String> {
    match value {
        ParamValue::Vec3(v) => Ok(v.map(|x| x as f32)),
        other => Err(expected(key, "a vec3", other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::FeatureId;
    use crate::exec::{default_values, merge_values, snapshot_keys};

    const POST: [FeatureId; 8] = [
        FeatureId::Ssr,
        FeatureId::Fog,
        FeatureId::Shafts,
        FeatureId::Oit,
        FeatureId::Dof,
        FeatureId::Bloom,
        FeatureId::Grading,
        FeatureId::Cel,
    ];

    fn pair(key: &str, value: ParamValue) -> (String, ParamValue) {
        (key.to_owned(), value)
    }

    #[test]
    fn bloom_defaults_are_the_bloom_desc_defaults() {
        let d = BloomDesc::default();
        let keys: Vec<String> = default_values(&FeatureId::Bloom)
            .into_iter()
            .map(|(key, _)| key)
            .collect();
        assert_eq!(keys, ["strength", "radius"]);
        assert_eq!(
            default_values(&FeatureId::Bloom),
            vec![
                pair("strength", ParamValue::Number(f64::from(d.strength))),
                pair("radius", ParamValue::Number(f64::from(d.radius))),
            ]
        );
    }

    #[test]
    fn oit_refuses_max_fragments_outside_its_range_under_its_key() {
        let changes = [pair("maxFragments", ParamValue::Int(2))];
        match merge_values(&FeatureId::Oit, &[], &changes) {
            Err((key, message)) => {
                assert_eq!(key, "maxFragments");
                assert!(message.contains("4..=32"), "message: {message}");
            }
            other => panic!("maxFragments 2 must be refused, got {other:?}"),
        }
    }

    #[test]
    fn grading_colour_key_round_trips_as_vec3() {
        // Exactly representable in f32, so the widened values compare equal.
        let lift = ParamValue::Vec3([0.25, 0.5, 0.75]);
        let changes = [pair("lift", lift.clone())];
        let merged = merge_values(&FeatureId::Grading, &[], &changes);
        assert_eq!(
            merged
                .ok()
                .and_then(|values| values.into_iter().find(|(key, _)| key == "lift")),
            Some(pair("lift", lift))
        );
    }

    #[test]
    fn dof_autofocus_is_not_a_key() {
        assert!(!snapshot_keys(&FeatureId::Dof).contains(&"autofocus"));
    }

    #[test]
    fn every_post_feature_round_trips_its_defaults() {
        for feature in &POST {
            let defaults = default_values(feature);
            let keys: Vec<&str> = defaults.iter().map(|(key, _)| key.as_str()).collect();
            assert_eq!(keys, snapshot_keys(feature), "{feature:?}");
            assert_eq!(
                merge_values(feature, &[], &defaults),
                Ok(defaults.clone()),
                "{feature:?}"
            );
        }
    }
}
