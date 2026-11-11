//! m08 sky transactions (S24): `setSky` parses a JS patch over the preset (or the current
//! sky) and hands `raa_sky::SkyTransaction` to the accepted state exactly once. The pure
//! parser runs natively; the wasm32 `Reflect` reader and the Engine methods are pinned by
//! `tests/api_registry.rs` and the browser render.
//!
//! The decisive boundary is intent, never resolved equality: an explicit preset selection
//! stays `Some(preset)` even when the merged `SkyDesc` equals the current one (raa-sky then
//! resets the preset moment), while an empty patch stays `None` and retains the authored
//! moment. Shared cloud fields are recorded individually through `LegacyCloudFields`, so an
//! omitted cloud value is not mistaken for an explicitly supplied one.

#![cfg(not(target_arch = "wasm32"))]

#[path = "../src/api/m08_sky_ibl.rs"]
mod m08;

use std::rc::Rc;

use m08::{Properties, Raw, parse_sky_transaction};
use raa_core::RaaError;
use raa_sky::{EnvUpdate, LegacyCloudFields, SkyDesc, SkyModel, SkyPreset, SkyTransaction};

/// A native stand-in for a JS description object: a key is present exactly when it is
/// listed, and `Raw` distinguishes `undefined`, `null` and a value.
#[derive(Clone)]
struct Obj(Vec<(&'static str, Raw)>);

impl Obj {
    fn new(fields: &[(&'static str, Raw)]) -> Self {
        Self(fields.to_vec())
    }

    fn empty() -> Self {
        Self(Vec::new())
    }

    fn into_rc(self) -> Rc<dyn Properties> {
        Rc::new(self)
    }
}

impl Properties for Obj {
    fn keys(&self) -> Result<Vec<String>, RaaError> {
        Ok(self.0.iter().map(|(key, _)| key.to_string()).collect())
    }

    fn read(&self, key: &str) -> Result<Raw, RaaError> {
        Ok(self
            .0
            .iter()
            .find(|(name, _)| *name == key)
            .map(|(_, value)| value.clone())
            .unwrap_or(Raw::Absent))
    }
}

fn obj(fields: &[(&'static str, Raw)]) -> Raw {
    Raw::Object(Obj::new(fields).into_rc())
}

fn parse(current: &SkyDesc, fields: &[(&'static str, Raw)]) -> Result<SkyTransaction, RaaError> {
    parse_sky_transaction(current, &Obj::new(fields))
}

fn sky_with_clouds(fields: &[(&'static str, Raw)]) -> Result<SkyTransaction, RaaError> {
    parse(&SkyDesc::preset("day").unwrap(), &[("clouds", obj(fields))])
}

fn expect_invalid(result: Result<SkyTransaction, RaaError>, path: &str) {
    let error = result.expect_err("the patch must be refused");
    assert_eq!(error.code(), "sky.invalid", "{error}");
    assert!(
        error.message().contains(&format!("'{path}'")),
        "message must name '{path}': {error}"
    );
}

/// Day preset values are the actual `desc.rs` defaults documented in m08.
#[test]
fn day_preset_defaults_match_the_documented_values() {
    let day = SkyDesc::preset("day").unwrap();
    assert_eq!(day.profile, SkyPreset::Day);
    assert_eq!(day.model, SkyModel::Physical);
    assert_eq!(day.zenith, [0.15, 0.35, 0.85]);
    assert_eq!(day.horizon, [0.85, 0.80, 0.70]);
    assert_eq!(day.ground, [0.35, 0.30, 0.25]);
    assert_eq!(day.intensity, 1.6);
    assert_eq!(day.horizon_softness, 4.0);
    assert_eq!(day.sun_disc_angular_radius, 0.0047);
    assert_eq!(day.sun_disc_intensity, 40.0);
    assert_eq!(day.clouds.coverage, 0.45);
    assert_eq!(day.clouds.density, 1.0);
    assert_eq!(day.clouds.altitude_m, 1500.0);
    assert_eq!(day.clouds.scale_m, 2500.0);
    assert_eq!(day.clouds.wind_mps, [4.0, 1.5]);
    assert_eq!(day.clouds.color, [0.95, 0.95, 0.97]);
    assert_eq!(day.clouds.silver_lining, 0.8);
    assert_eq!(day.atmosphere.ground_albedo, [0.3; 3]);
    assert_eq!(day.atmosphere.mie_multiplier, 1.0);
    assert_eq!(day.env_update, EnvUpdate::OnChange);
}

/// A time edit moves only the raa-sky runtime cursor; the authored descriptor is unchanged.
/// An explicit day preset must therefore remain an explicit event even though
/// `setSky({ preset: 'day' })` and `setSky({})` resolve to the identical descriptor here.
#[test]
fn explicit_day_preset_is_not_inferred_from_equal_resolved_values() {
    let current = SkyDesc::preset("day").unwrap();
    let explicit = parse(&current, &[("preset", Raw::String("day".into()))]).unwrap();
    let empty = parse(&current, &Obj::empty().0).unwrap();
    assert_eq!(explicit.preset, Some(SkyPreset::Day));
    assert_eq!(empty.preset, None);
    assert_eq!(
        explicit.desc, empty.desc,
        "the resolved descriptors are equal"
    );
    assert_eq!(explicit.desc, current);
    assert_eq!(explicit.cloud_fields, LegacyCloudFields::NONE);
    assert_eq!(empty.cloud_fields, LegacyCloudFields::NONE);
}

#[test]
fn empty_patch_retains_the_accepted_descriptor() {
    let mut current = SkyDesc::preset("night").unwrap();
    current.model = SkyModel::Legacy;
    current.zenith = [0.4, 0.1, 0.2];
    current.clouds.coverage = 0.9;
    current.atmosphere.mie_multiplier = 1.7;
    let transaction = parse(&current, &[]).unwrap();
    assert_eq!(transaction.desc, current);
    assert_eq!(transaction.preset, None);
    assert_eq!(transaction.cloud_fields, LegacyCloudFields::NONE);
}

#[test]
fn preset_is_applied_before_overrides_and_presence_is_explicit() {
    let current = SkyDesc::preset("night").unwrap();
    let transaction = parse(
        &current,
        &[
            ("preset", Raw::String("day".into())),
            ("zenith", Raw::Numbers(vec![0.2, 0.3, 0.4])),
            (
                "clouds",
                obj(&[
                    ("coverage", Raw::Number(0.9)),
                    ("density", Raw::Number(2.0)),
                ]),
            ),
        ],
    )
    .unwrap();
    assert_eq!(transaction.preset, Some(SkyPreset::Day));
    assert_eq!(transaction.desc.profile, SkyPreset::Day);
    assert_eq!(transaction.desc.zenith, [0.2, 0.3, 0.4]);
    assert_eq!(transaction.desc.clouds.coverage, 0.9);
    assert_eq!(transaction.desc.clouds.density, 2.0);
    // The preset supplied the rest of the day base, not the night current values.
    assert_eq!(transaction.desc.horizon, [0.85, 0.80, 0.70]);
    assert_eq!(transaction.desc.atmosphere.ground_albedo, [0.3; 3]);
    assert_eq!(
        transaction.cloud_fields,
        LegacyCloudFields(LegacyCloudFields::COVERAGE.0 | LegacyCloudFields::DENSITY.0)
    );
}

#[test]
fn shared_cloud_omissions_and_later_setters_keep_the_base() {
    let day = SkyDesc::preset("day").unwrap();
    let first = parse(
        &day,
        &[(
            "clouds",
            obj(&[
                ("coverage", Raw::Number(0.9)),
                ("density", Raw::Number(2.0)),
                ("altitude", Raw::Number(1800.0)),
                ("wind", Raw::Numbers(vec![3.0, 4.0])),
            ]),
        )],
    )
    .unwrap();
    assert_eq!(first.cloud_fields, LegacyCloudFields::ALL);

    // A later patch over the accepted descriptor changes only what it supplies.
    let later = parse(
        &first.desc,
        &[("clouds", obj(&[("density", Raw::Number(3.0))]))],
    )
    .unwrap();
    assert_eq!(later.cloud_fields, LegacyCloudFields::DENSITY);
    assert_eq!(later.desc.clouds.density, 3.0);
    assert_eq!(later.desc.clouds.coverage, 0.9);
    assert_eq!(later.desc.clouds.altitude_m, 1800.0);
    assert_eq!(later.desc.clouds.wind_mps, [3.0, 4.0]);

    // `null` is an old-field omission, so no shared bit and no value change.
    let omitted = parse(
        &later.desc,
        &[(
            "clouds",
            obj(&[
                ("coverage", Raw::Null),
                ("altitude", Raw::Undefined),
                ("wind", Raw::Null),
            ]),
        )],
    )
    .unwrap();
    assert_eq!(omitted.cloud_fields, LegacyCloudFields::NONE);
    assert_eq!(omitted.desc, later.desc);

    // Artistic-only cloud controls never set a shared bit.
    let artistic = parse(
        &day,
        &[(
            "clouds",
            obj(&[
                ("scale", Raw::Number(9000.0)),
                ("color", Raw::Numbers(vec![0.5, 0.6, 0.9])),
                ("silverLining", Raw::Number(0.2)),
            ]),
        )],
    )
    .unwrap();
    assert_eq!(artistic.cloud_fields, LegacyCloudFields::NONE);
    assert_eq!(artistic.desc.clouds.scale_m, 9000.0);
    assert_eq!(artistic.desc.clouds.color, [0.5, 0.6, 0.9]);
    assert_eq!(artistic.desc.clouds.silver_lining, 0.2);
}

#[test]
fn old_fields_keep_omission_and_null_compatibility() {
    let mut current = SkyDesc::preset("day").unwrap();
    current.zenith = [0.9, 0.8, 0.7];
    current.intensity = 2.5;
    current.env_update = EnvUpdate::EveryFrame;
    let transaction = parse(
        &current,
        &[
            ("preset", Raw::Null),
            ("zenith", Raw::Null),
            ("horizon", Raw::Undefined),
            ("ground", Raw::Null),
            ("intensity", Raw::Null),
            ("horizonSoftness", Raw::Null),
            ("sunDiscSize", Raw::Null),
            ("sunDiscIntensity", Raw::Null),
            ("clouds", Raw::Null),
            ("envUpdate", Raw::Null),
        ],
    )
    .unwrap();
    assert_eq!(transaction.desc, current);
    assert_eq!(transaction.preset, None);
    assert_eq!(transaction.cloud_fields, LegacyCloudFields::NONE);
}

#[test]
fn model_switch_alone_keeps_the_other_authored_values() {
    let current = SkyDesc::preset("jungle").unwrap();
    let transaction = parse(&current, &[("model", Raw::String("legacy".into()))]).unwrap();
    assert_eq!(transaction.preset, None);
    assert_eq!(transaction.cloud_fields, LegacyCloudFields::NONE);
    assert_eq!(transaction.desc.model, SkyModel::Legacy);
    assert_eq!(transaction.desc.profile, current.profile);
    assert_eq!(transaction.desc.zenith, current.zenith);
    assert_eq!(transaction.desc.clouds, current.clouds);
}

#[test]
fn atmosphere_overrides_apply_after_the_preset() {
    let current = SkyDesc::preset("day").unwrap();
    let transaction = parse(
        &current,
        &[
            ("preset", Raw::String("snow".into())),
            (
                "atmosphere",
                obj(&[
                    ("groundAlbedo", Raw::Numbers(vec![0.1, 0.2, 0.3])),
                    ("mieMultiplier", Raw::Number(2.2)),
                ]),
            ),
        ],
    )
    .unwrap();
    assert_eq!(transaction.preset, Some(SkyPreset::Snow));
    assert_eq!(transaction.desc.profile, SkyPreset::Snow);
    assert_eq!(transaction.desc.atmosphere.ground_albedo, [0.1, 0.2, 0.3]);
    assert_eq!(transaction.desc.atmosphere.mie_multiplier, 2.2);
}

#[test]
fn new_fields_reject_null_wrong_types_overflow_and_unknown_keys() {
    let day = SkyDesc::preset("day").unwrap();
    expect_invalid(parse(&day, &[("model", Raw::Null)]), "model");
    expect_invalid(parse(&day, &[("model", Raw::Number(1.0))]), "model");
    expect_invalid(parse(&day, &[("model", Raw::Other)]), "model");
    expect_invalid(
        parse(&day, &[("model", Raw::String("cloudy".into()))]),
        "model",
    );
    expect_invalid(parse(&day, &[("atmosphere", Raw::Null)]), "atmosphere");
    expect_invalid(
        parse(&day, &[("atmosphere", Raw::Number(1.0))]),
        "atmosphere",
    );
    expect_invalid(
        parse(
            &day,
            &[("atmosphere", obj(&[("mieMultiplier", Raw::Null)]))],
        ),
        "atmosphere.mieMultiplier",
    );
    expect_invalid(
        parse(&day, &[("atmosphere", obj(&[("groundAlbedo", Raw::Null)]))]),
        "atmosphere.groundAlbedo",
    );
    expect_invalid(
        parse(
            &day,
            &[(
                "atmosphere",
                obj(&[("groundAlbedo", Raw::Numbers(vec![0.0, 1.0, 2.0]))]),
            )],
        ),
        "atmosphere.groundAlbedo[2]",
    );
    expect_invalid(
        parse(
            &day,
            &[("atmosphere", obj(&[("mieMultiplier", Raw::Number(0.0))]))],
        ),
        "atmosphere.mieMultiplier",
    );
    expect_invalid(
        parse(
            &day,
            &[("atmosphere", obj(&[("mieMultiplier", Raw::Number(1e300))]))],
        ),
        "atmosphere.mieMultiplier",
    );
    expect_invalid(
        parse(&day, &[("atmosphere", obj(&[("bogus", Raw::Number(1.0))]))]),
        "atmosphere.bogus",
    );
    expect_invalid(parse(&day, &[("bogus", Raw::Number(1.0))]), "bogus");
    expect_invalid(
        parse(&day, &[("clouds", obj(&[("bogus", Raw::Number(1.0))]))]),
        "clouds.bogus",
    );
    expect_invalid(
        parse(&day, &[("clouds", Raw::String("none".into()))]),
        "clouds",
    );
}

#[test]
fn old_field_domains_and_paths_stay_explicit() {
    let day = SkyDesc::preset("day").unwrap();
    expect_invalid(
        parse(&day, &[("preset", Raw::String("dusk".into()))]),
        "preset",
    );
    expect_invalid(parse(&day, &[("preset", Raw::Number(1.0))]), "preset");
    expect_invalid(
        parse(&day, &[("sunDiscSize", Raw::Number(0.3))]),
        "sunDiscSize",
    );
    expect_invalid(
        parse(&day, &[("horizonSoftness", Raw::Number(0.0))]),
        "horizonSoftness",
    );
    expect_invalid(
        parse(&day, &[("intensity", Raw::Number(1e300))]),
        "intensity",
    );
    expect_invalid(
        parse(&day, &[("envUpdate", Raw::String("sometimes".into()))]),
        "envUpdate",
    );
    expect_invalid(
        sky_with_clouds(&[("coverage", Raw::Number(1.5))]),
        "clouds.coverage",
    );
    expect_invalid(
        sky_with_clouds(&[("wind", Raw::Numbers(vec![1.0]))]),
        "clouds.wind",
    );
    expect_invalid(
        sky_with_clouds(&[("color", Raw::Numbers(vec![0.5, -1.0, 0.5]))]),
        "clouds.color[1]",
    );
    expect_invalid(
        sky_with_clouds(&[("silverLining", Raw::Number(2.0))]),
        "clouds.silverLining",
    );
    expect_invalid(
        sky_with_clouds(&[("altitude", Raw::Number(0.0))]),
        "clouds.altitude",
    );
}

#[test]
fn preset_with_a_shared_override_keeps_both_pieces_of_intent() {
    // raa-sky applies the preset before the explicitly supplied shared cloud values; the
    // parser keeps only the supplied bits and the preset identity.
    let day = SkyDesc::preset("day").unwrap();
    let transaction = parse(
        &day,
        &[
            ("preset", Raw::String("day".into())),
            ("clouds", obj(&[("coverage", Raw::Number(0.7))])),
            ("model", Raw::String("physical".into())),
        ],
    )
    .unwrap();
    assert_eq!(transaction.preset, Some(SkyPreset::Day));
    assert_eq!(transaction.cloud_fields, LegacyCloudFields::COVERAGE);
    assert_eq!(transaction.desc.clouds.coverage, 0.7);
}
