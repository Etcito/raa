//! m08 sky/time transactions (S24): `setSky` parses a JS patch over the preset (or the
//! current sky) and hands `raa_sky::SkyTransaction` to the accepted state exactly once;
//! `setTimeOfDay` merges and validates an authored clock patch; `skyWeatherDiagnostics`
//! serializes the exact typed lifecycle observation. The pure parsers and serializers run
//! natively; the wasm32 `Reflect` reader and the Engine methods are pinned here, by
//! `tests/api_registry.rs` and by the managed browser scenes.
//!
//! The decisive boundary is intent, never resolved equality: an explicit preset selection
//! stays `Some(preset)` even when the merged `SkyDesc` equals the current one (raa-sky then
//! resets the preset moment), while an empty patch stays `None` and retains the authored
//! moment. Shared cloud fields are recorded individually through `LegacyCloudFields`, so an
//! omitted cloud value is not mistaken for an explicitly supplied one.

#![cfg(not(target_arch = "wasm32"))]

#[path = "../src/api/m08_sky_ibl.rs"]
mod m08;

use std::fs;
use std::path::Path;
use std::rc::Rc;

use m08::{
    MAX_SAFE_COUNTER, Properties, Raw, parse_sky_transaction, parse_time_of_day,
    parse_weather_update, serialize_diagnostics, serialize_time_of_day,
};
use raa_core::RaaError;
use raa_ibl::IblPublication;
use raa_sky::{
    CloudResetReasons, DominantLight, EnvUpdate, LegacyCloudFields, SkyDesc, SkyDiagnostics,
    SkyModel, SkyPreset, SkyReadiness, SkyReadinessStage, SkyReadinessStatus, SkyTransaction,
    SkyViewDiagnostics, TimeOfDay, TimeOfDaySettings, TimeOfDayState, Weather, WeatherPreset,
    WeatherSettings, WeatherUpdate,
};

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

/// The m08 registry entry and every public method name; the frame hook observes the
/// engine's single clock advance and never performs a second one.
#[test]
fn registry_exports_the_sky_time_and_diagnostics_methods() {
    let listed = fs::read_to_string(Path::new(env!("OUT_DIR")).join("api_files.txt"))
        .expect("api_files.txt");
    assert!(
        listed.lines().any(|l| l == "m08_sky_ibl"),
        "m08_sky_ibl not registered"
    );
    let reg = fs::read_to_string(Path::new(env!("OUT_DIR")).join("api_registry.rs"))
        .expect("api_registry.rs");
    assert!(reg.contains("m08_sky_ibl::install(ctx)?;"));
    assert!(reg.contains("m08_sky_ibl::frame(st, gpu, input)?;"));
    let src = include_str!("../src/api/m08_sky_ibl.rs");
    for name in [
        "setSky",
        "setEnvironmentLighting",
        "setTimeOfDay",
        "timeOfDay",
        "skyWeatherDiagnostics",
    ] {
        assert!(src.contains(&format!("js_name = {name})]")), "no {name}");
    }
    assert!(
        src.contains("pub fn sky(&self)"),
        "the sky getter keeps its name"
    );
    assert_eq!(src.matches("sky.mark_dirty();").count(), 1);
    assert!(
        !src.contains("advance_frame("),
        "the m08 frame hook must not advance the clock a second time"
    );
    assert!(
        !src.contains("scene_invocation"),
        "no submitted-frame counter in m08"
    );
}

fn time(fields: &[(&'static str, Raw)]) -> Result<TimeOfDaySettings, RaaError> {
    parse_time_of_day(&TimeOfDaySettings::default(), &Obj::new(fields))
}

fn expect_time_invalid(result: Result<TimeOfDaySettings, RaaError>, path: &str) {
    let error = result.expect_err("the patch must be refused");
    assert_eq!(error.code(), "sky.invalid", "{error}");
    assert!(
        error.message().contains(&format!("'{path}'")),
        "message must name '{path}': {error}"
    );
}

#[test]
fn time_patch_merges_over_the_authored_settings_and_keeps_omissions() {
    let current = TimeOfDaySettings::default();
    assert_eq!(
        current.time,
        TimeOfDay {
            day: 172,
            hours: 9.0
        }
    );
    assert_eq!((current.place.latitude, current.place.north), (38.0, 0.0));
    assert!(!current.enabled && !current.playing);
    assert_eq!(current.speed, 1.0);

    let merged = parse_time_of_day(
        &current,
        &Obj::new(&[
            ("day", Raw::Number(200.0)),
            ("hours", Raw::Number(18.5)),
            ("latitude", Raw::Number(-12.5)),
            ("north", Raw::Number(1.25)),
            ("enabled", Raw::Bool(true)),
            ("playing", Raw::Bool(true)),
            ("speed", Raw::Number(60.0)),
        ]),
    )
    .expect("valid patch");
    assert_eq!(
        merged.time,
        TimeOfDay {
            day: 200,
            hours: 18.5
        }
    );
    assert_eq!((merged.place.latitude, merged.place.north), (-12.5, 1.25));
    assert!(merged.enabled && merged.playing);
    assert_eq!(merged.speed, 60.0);

    // Undefined (and absence) omit; the authored moment is not the runtime cursor here.
    let omitted = parse_time_of_day(
        &merged,
        &Obj::new(&[
            ("day", Raw::Undefined),
            ("hours", Raw::Undefined),
            ("latitude", Raw::Undefined),
            ("north", Raw::Undefined),
            ("enabled", Raw::Undefined),
            ("playing", Raw::Undefined),
            ("speed", Raw::Undefined),
        ]),
    )
    .expect("omissions");
    assert_eq!(omitted, merged);
    assert_eq!(parse_time_of_day(&merged, &Obj::empty()).unwrap(), merged);
}

#[test]
fn author_time_and_runtime_cursor_stay_separate_in_the_state_view() {
    let mut state = TimeOfDayState::default();
    state.settings.time = TimeOfDay {
        day: 200,
        hours: 18.25,
    };
    state.current = TimeOfDay {
        day: 201,
        hours: 6.5,
    };
    state.sky = state.current.sky(&state.settings.place);
    state.dominant = DominantLight::Moon;
    let v = serialize_time_of_day(&state);
    assert_eq!((v.day, v.hours), (200, 18.25));
    assert_eq!((v.current_day, v.current_hours), (201, 6.5));
    assert_eq!((v.latitude, v.north), (38.0, 0.0));
    assert!(!v.enabled && !v.playing);
    assert_eq!(v.speed, 1.0);
    assert_eq!(
        v.sun_direction,
        [state.sky.sun.x, state.sky.sun.y, state.sky.sun.z]
    );
    assert_eq!(
        v.moon_direction,
        [state.sky.moon.x, state.sky.moon.y, state.sky.moon.z]
    );
    assert_eq!(v.moon_lit, state.sky.moon_lit);
    assert_eq!(
        v.star_pole,
        [state.sky.pole.x, state.sky.pole.y, state.sky.pole.z]
    );
    assert_eq!(
        v.star_equinox,
        [
            state.sky.equinox.x,
            state.sky.equinox.y,
            state.sky.equinox.z
        ]
    );
    assert_eq!(v.dominant, "moon");
    state.dominant = DominantLight::Sun;
    assert_eq!(serialize_time_of_day(&state).dominant, "sun");
    state.dominant = DominantLight::Manual;
    assert_eq!(serialize_time_of_day(&state).dominant, "manual");
}

#[test]
fn time_patch_refuses_null_wrong_types_unknown_keys_and_domains() {
    expect_time_invalid(time(&[("day", Raw::Null)]), "day");
    expect_time_invalid(time(&[("day", Raw::Number(0.0))]), "day");
    expect_time_invalid(time(&[("day", Raw::Number(366.0))]), "day");
    expect_time_invalid(time(&[("day", Raw::Number(1.5))]), "day");
    expect_time_invalid(time(&[("day", Raw::Number(f64::NAN))]), "day");
    expect_time_invalid(time(&[("day", Raw::String("172".into()))]), "day");
    expect_time_invalid(time(&[("hours", Raw::Null)]), "hours");
    expect_time_invalid(time(&[("hours", Raw::Number(-0.5))]), "hours");
    expect_time_invalid(time(&[("hours", Raw::Number(24.0))]), "hours");
    expect_time_invalid(time(&[("hours", Raw::Number(1e300))]), "hours");
    expect_time_invalid(time(&[("latitude", Raw::Null)]), "latitude");
    expect_time_invalid(time(&[("latitude", Raw::Number(90.5))]), "latitude");
    expect_time_invalid(time(&[("latitude", Raw::Number(-90.5))]), "latitude");
    expect_time_invalid(time(&[("latitude", Raw::Number(1e300))]), "latitude");
    expect_time_invalid(time(&[("north", Raw::Null)]), "north");
    expect_time_invalid(time(&[("north", Raw::Number(1e300))]), "north");
    expect_time_invalid(time(&[("enabled", Raw::Null)]), "enabled");
    expect_time_invalid(time(&[("enabled", Raw::Number(1.0))]), "enabled");
    expect_time_invalid(time(&[("playing", Raw::Null)]), "playing");
    expect_time_invalid(time(&[("playing", Raw::String("yes".into()))]), "playing");
    expect_time_invalid(time(&[("speed", Raw::Null)]), "speed");
    expect_time_invalid(time(&[("speed", Raw::Number(-1.0))]), "speed");
    expect_time_invalid(time(&[("speed", Raw::Number(f64::INFINITY))]), "speed");
    expect_time_invalid(time(&[("altitude", Raw::Number(1.0))]), "altitude");
}

fn weather(
    current: &WeatherSettings,
    fields: &[(&'static str, Raw)],
) -> Result<WeatherUpdate, RaaError> {
    parse_weather_update(current, &Obj::new(fields))
}

fn expect_weather_invalid(result: Result<WeatherUpdate, RaaError>, path: &str) {
    let error = result.expect_err("the weather patch must be refused");
    assert_eq!(error.code(), "sky.invalid", "{error}");
    assert!(
        error.message().contains(&format!("'{path}'")),
        "message must name '{path}': {error}"
    );
}

/// An explicit preset supplies the base before every override, keeps its named identity and
/// requests a restart even when the structured values happen to equal the base. Text has no
/// preset identity, so it supplies fair as the cycle start.
#[test]
fn weather_preset_and_text_supply_explicit_start_before_overrides() {
    let current = WeatherSettings::default();
    let storm = Weather::by_name("storm").unwrap();
    let update = weather(
        &current,
        &[
            ("preset", Raw::String("storm".into())),
            ("coverage", Raw::Number(0.22)),
            ("changing", Raw::Bool(true)),
        ],
    )
    .expect("preset plus overrides");
    assert_eq!(update.cycle_start, Some(WeatherPreset::Storm));
    assert!(update.restart_cycle, "an explicit base selection restarts");
    assert!(update.settings.changing);
    assert_eq!(update.settings.target.coverage, 0.22);
    assert_eq!(update.settings.target.thickness, storm.thickness);
    assert_eq!(update.settings.target.wind_speed, storm.wind_speed);
    assert_eq!(update.settings.target.base, storm.base);
    assert_eq!(update.settings.seed, current.seed);
    assert_eq!(
        update.settings.transition_seconds,
        current.transition_seconds
    );

    let parsed = Weather::parse("coverage 0.8\nrain 1\n").unwrap();
    let text = weather(
        &current,
        &[("text", Raw::String("coverage 0.8\nrain 1\n".into()))],
    )
    .expect("text weather");
    assert_eq!(text.cycle_start, Some(WeatherPreset::Fair));
    assert!(text.restart_cycle);
    assert_eq!(text.settings.target, parsed);

    // Text that describes storm values still reports fair as its start identity.
    let storm_text = weather(&current, &[("text", Raw::String(storm.to_text()))]).unwrap();
    assert_eq!(storm_text.cycle_start, Some(WeatherPreset::Fair));
    assert_eq!(storm_text.settings.target, storm);
}

/// The parser returns intent, never an inference from resolved equality: equal-valued
/// patches differ in cycle start and restart, and an empty patch retains the current values.
#[test]
fn weather_equal_values_keep_distinct_cycle_intent() {
    let current = WeatherSettings::default();
    let explicit = weather(&current, &[("preset", Raw::String("fair".into()))]).unwrap();
    let empty = weather(&current, &[]).unwrap();
    assert_eq!(explicit.settings.target, current.target);
    assert_eq!(empty.settings.target, current.target);
    assert_eq!(explicit.cycle_start, Some(WeatherPreset::Fair));
    assert!(explicit.restart_cycle);
    assert_eq!(empty.cycle_start, None);
    assert!(!empty.restart_cycle);
    assert_eq!(empty.settings, current);

    // The values override the preset but stay equal to both bases; intent is still explicit.
    let overridden = weather(
        &current,
        &[
            ("preset", Raw::String("fair".into())),
            ("coverage", Raw::Number(f64::from(current.target.coverage))),
            (
                "wind",
                obj(&[("speed", Raw::Number(f64::from(current.target.wind_speed)))]),
            ),
        ],
    )
    .unwrap();
    assert_eq!(overridden.settings.target, current.target);
    assert_eq!(overridden.cycle_start, Some(WeatherPreset::Fair));
    assert!(overridden.restart_cycle);
}

/// A manual target edit stops the cycle unless the same patch explicitly re-enables it. A
/// changed seed or a false-to-true changing edit restarts; unchanged `changing: true` alone
/// keeps the running cycle and does not restart it.
#[test]
fn weather_manual_target_edits_and_explicit_restart_intent() {
    let current = WeatherSettings {
        changing: true,
        seed: 7,
        ..WeatherSettings::default()
    };
    let edit = weather(&current, &[("coverage", Raw::Number(0.4))]).unwrap();
    assert!(
        !edit.settings.changing,
        "a manual target edit stops changing"
    );
    assert!(!edit.restart_cycle);
    assert_eq!(edit.settings.target.coverage, 0.4);
    assert_eq!(edit.settings.seed, 7);

    let kept = weather(
        &current,
        &[
            ("coverage", Raw::Number(0.4)),
            ("changing", Raw::Bool(true)),
        ],
    )
    .unwrap();
    assert!(
        kept.settings.changing,
        "an explicit true wins over the stop"
    );
    assert!(
        !kept.restart_cycle,
        "an unchanged changing:true never restarts"
    );

    let idle = weather(&current, &[("changing", Raw::Bool(true))]).unwrap();
    assert!(idle.settings.changing);
    assert!(!idle.restart_cycle);
    assert_eq!(idle.settings, current);

    let stopped = WeatherSettings {
        changing: false,
        ..current
    };
    let enabled = weather(&stopped, &[("changing", Raw::Bool(true))]).unwrap();
    assert!(enabled.settings.changing);
    assert!(enabled.restart_cycle, "false-to-true restarts");

    let reseeded = weather(&current, &[("seed", Raw::Number(8.0))]).unwrap();
    assert_eq!(reseeded.settings.seed, 8);
    assert!(reseeded.restart_cycle, "a changed seed restarts");

    // Changing false with no base selection or seed change never restarts.
    let stopped = weather(&stopped, &[("changing", Raw::Bool(false))]).unwrap();
    assert!(!stopped.settings.changing);
    assert!(!stopped.restart_cycle);
}

/// Omitted and `undefined` fields merge over the accepted authored settings, including each
/// wind component separately; null is a refusal, never an omission.
#[test]
fn weather_omissions_merge_nested_wind() {
    let current = WeatherSettings {
        enabled: false,
        shadows: false,
        changing: true,
        seed: 42,
        transition_seconds: 2.5,
        target: Weather::by_name("broken").unwrap(),
    };
    let update = weather(&current, &[("wind", obj(&[("speed", Raw::Number(3.0))]))]).unwrap();
    assert_eq!(update.settings.target.wind_speed, 3.0);
    assert_eq!(update.settings.target.wind_from, current.target.wind_from);
    assert_eq!(update.settings.target.coverage, current.target.coverage);
    assert!(!update.settings.enabled);
    assert!(!update.settings.shadows);
    assert_eq!(update.settings.seed, 42);
    assert_eq!(update.settings.transition_seconds, 2.5);
    assert!(!update.settings.changing, "the wind edit is manual");

    // A wind object whose components are undefined is an omission of both components.
    let undefined_wind = weather(
        &current,
        &[(
            "wind",
            obj(&[("speed", Raw::Undefined), ("from", Raw::Undefined)]),
        )],
    )
    .unwrap();
    assert_eq!(undefined_wind.settings, current);
    assert!(undefined_wind.settings.changing, "undefined is not an edit");

    let omitted = weather(
        &current,
        &[
            ("preset", Raw::Undefined),
            ("text", Raw::Undefined),
            ("enabled", Raw::Undefined),
            ("shadows", Raw::Undefined),
            ("changing", Raw::Undefined),
            ("seed", Raw::Undefined),
            ("transitionSeconds", Raw::Undefined),
            ("coverage", Raw::Undefined),
            ("type", Raw::Undefined),
            ("base", Raw::Undefined),
            ("thickness", Raw::Undefined),
            ("density", Raw::Undefined),
            ("cirrus", Raw::Undefined),
            ("wind", Raw::Undefined),
            ("evolution", Raw::Undefined),
            ("rain", Raw::Undefined),
        ],
    )
    .unwrap();
    assert_eq!(omitted.settings, current);
    assert_eq!(omitted.cycle_start, None);
    assert!(!omitted.restart_cycle);
}

/// Every structured refusal names its complete field path with `sky.invalid`, and the
/// accepted current settings are never mutated by a refused patch.
#[test]
fn weather_candidate_refusals_name_complete_paths() {
    let current = WeatherSettings::default();
    let before = current;
    expect_weather_invalid(weather(&current, &[("preset", Raw::Null)]), "preset");
    expect_weather_invalid(
        weather(&current, &[("preset", Raw::String("drizzle".into()))]),
        "preset",
    );
    expect_weather_invalid(weather(&current, &[("preset", Raw::Number(1.0))]), "preset");
    expect_weather_invalid(weather(&current, &[("text", Raw::Null)]), "text");
    expect_weather_invalid(
        weather(&current, &[("text", Raw::String("coverage nope\n".into()))]),
        "text",
    );
    expect_weather_invalid(
        weather(
            &current,
            &[
                ("preset", Raw::String("fair".into())),
                ("text", Raw::String(String::new())),
            ],
        ),
        "text",
    );
    expect_weather_invalid(weather(&current, &[("enabled", Raw::Null)]), "enabled");
    expect_weather_invalid(
        weather(&current, &[("enabled", Raw::Number(1.0))]),
        "enabled",
    );
    expect_weather_invalid(weather(&current, &[("shadows", Raw::Null)]), "shadows");
    expect_weather_invalid(
        weather(&current, &[("shadows", Raw::String("on".into()))]),
        "shadows",
    );
    expect_weather_invalid(weather(&current, &[("changing", Raw::Null)]), "changing");
    expect_weather_invalid(
        weather(&current, &[("changing", Raw::Number(1.0))]),
        "changing",
    );

    expect_weather_invalid(weather(&current, &[("seed", Raw::Null)]), "seed");
    expect_weather_invalid(weather(&current, &[("seed", Raw::Number(-1.0))]), "seed");
    expect_weather_invalid(weather(&current, &[("seed", Raw::Number(1.5))]), "seed");
    expect_weather_invalid(
        weather(
            &current,
            &[("seed", Raw::Number(f64::from(u32::MAX) + 1.0))],
        ),
        "seed",
    );
    expect_weather_invalid(
        weather(&current, &[("seed", Raw::Number(f64::NAN))]),
        "seed",
    );
    expect_weather_invalid(
        weather(&current, &[("seed", Raw::Number(f64::INFINITY))]),
        "seed",
    );
    expect_weather_invalid(
        weather(&current, &[("seed", Raw::String("1".into()))]),
        "seed",
    );
    expect_weather_invalid(
        weather(&current, &[("transitionSeconds", Raw::Null)]),
        "transitionSeconds",
    );
    expect_weather_invalid(
        weather(&current, &[("transitionSeconds", Raw::Number(-0.5))]),
        "transitionSeconds",
    );
    expect_weather_invalid(
        weather(
            &current,
            &[("transitionSeconds", Raw::Number(f64::INFINITY))],
        ),
        "transitionSeconds",
    );
    expect_weather_invalid(
        weather(&current, &[("transitionSeconds", Raw::Number(1e300))]),
        "transitionSeconds",
    );

    expect_weather_invalid(weather(&current, &[("coverage", Raw::Null)]), "coverage");
    expect_weather_invalid(
        weather(&current, &[("coverage", Raw::Number(1.5))]),
        "coverage",
    );
    expect_weather_invalid(weather(&current, &[("type", Raw::Null)]), "type");
    expect_weather_invalid(weather(&current, &[("type", Raw::Number(-0.1))]), "type");
    expect_weather_invalid(weather(&current, &[("base", Raw::Null)]), "base");
    expect_weather_invalid(weather(&current, &[("base", Raw::Number(-1.0))]), "base");
    expect_weather_invalid(weather(&current, &[("thickness", Raw::Null)]), "thickness");
    expect_weather_invalid(
        weather(&current, &[("thickness", Raw::Number(99.0))]),
        "thickness",
    );
    expect_weather_invalid(weather(&current, &[("density", Raw::Null)]), "density");
    expect_weather_invalid(
        weather(&current, &[("density", Raw::Number(-1.0))]),
        "density",
    );
    expect_weather_invalid(weather(&current, &[("cirrus", Raw::Null)]), "cirrus");
    expect_weather_invalid(weather(&current, &[("cirrus", Raw::Number(1.1))]), "cirrus");
    expect_weather_invalid(weather(&current, &[("evolution", Raw::Null)]), "evolution");
    expect_weather_invalid(
        weather(&current, &[("evolution", Raw::Number(-1.0))]),
        "evolution",
    );
    expect_weather_invalid(weather(&current, &[("rain", Raw::Null)]), "rain");
    expect_weather_invalid(weather(&current, &[("rain", Raw::Number(2.0))]), "rain");

    expect_weather_invalid(weather(&current, &[("wind", Raw::Null)]), "wind");
    expect_weather_invalid(
        weather(&current, &[("wind", Raw::String("north".into()))]),
        "wind",
    );
    expect_weather_invalid(
        weather(&current, &[("wind", obj(&[("speed", Raw::Null)]))]),
        "wind.speed",
    );
    expect_weather_invalid(
        weather(&current, &[("wind", obj(&[("speed", Raw::Number(-1.0))]))]),
        "wind.speed",
    );
    expect_weather_invalid(
        weather(&current, &[("wind", obj(&[("from", Raw::Null)]))]),
        "wind.from",
    );
    expect_weather_invalid(
        weather(
            &current,
            &[("wind", obj(&[("from", Raw::Number(f64::INFINITY))]))],
        ),
        "wind.from",
    );
    expect_weather_invalid(
        weather(&current, &[("wind", obj(&[("gust", Raw::Number(1.0))]))]),
        "wind.gust",
    );

    // A parsed text base does not hide a later structured refusal, and the derived
    // base + thickness stays finite in f32.
    expect_weather_invalid(
        weather(
            &current,
            &[
                ("text", Raw::String("coverage 0.5\n".into())),
                ("thickness", Raw::Number(10.0)),
            ],
        ),
        "thickness",
    );
    expect_weather_invalid(
        weather(
            &current,
            &[
                ("base", Raw::Number(f64::from(f32::MAX))),
                ("thickness", Raw::Number(f64::from(f32::MAX))),
            ],
        ),
        "base",
    );
    expect_weather_invalid(weather(&current, &[("bogus", Raw::Number(1.0))]), "bogus");
    assert_eq!(
        current, before,
        "a refusal never mutates the current settings"
    );
}

/// The JS boundary keeps the required private parser signature and the preceding sky/time
/// methods; setWeather is the separate consumer that applies the returned candidate once.
#[test]
fn weather_candidate_parser_exists_with_the_js_description_signature() {
    let src = include_str!("../src/api/m08_sky_ibl.rs");
    assert!(src.contains("fn parse_weather_candidate("));
    assert!(src.contains("desc: &JsValue,"));
    assert!(src.contains("current: &raa_sky::WeatherSettings,"));
    assert!(src.contains("-> Result<raa_sky::WeatherUpdate, RaaError>"));
    for name in [
        "setSky",
        "setEnvironmentLighting",
        "setTimeOfDay",
        "timeOfDay",
        "skyWeatherDiagnostics",
    ] {
        assert!(src.contains(&format!("js_name = {name})]")), "no {name}");
    }
}

fn view(view: u32, reasons: CloudResetReasons) -> SkyViewDiagnostics {
    SkyViewDiagnostics {
        view,
        width: 1280,
        height: 720,
        invocation: Some(7),
        submitted_invocation: None,
        sky_view_generation: 2,
        history_valid: true,
        reset_reasons: reasons,
        cloud_screen_valid: false,
        cloud_panorama_valid: true,
    }
}

#[test]
fn diagnostics_defaults_are_empty_null_and_pending_install() {
    let d = serialize_diagnostics(&SkyDiagnostics::default()).expect("defaults");
    assert_eq!(d.invocation, None);
    assert_eq!(d.submitted_invocation, None);
    assert_eq!((d.weather_generation, d.atmosphere_generation), (0, 0));
    assert_eq!(
        d.environment,
        IblPublication {
            requested_generation: 0,
            published_generation: None,
            pending: false,
        }
    );
    assert!(d.views.is_empty(), "no prepared view serializes as []");
    assert_eq!(d.readiness.status, "pending");
    assert_eq!(d.readiness.stage, "install");
    assert_eq!(d.readiness.error, None);
    assert_eq!(SkyReadinessStatus::Pending.as_str(), "pending");
}

#[test]
fn diagnostics_keep_slot_order_reason_order_and_null_options() {
    let diagnostics = SkyDiagnostics {
        invocation: Some(9),
        submitted_invocation: None,
        weather_generation: 3,
        atmosphere_generation: 4,
        environment: IblPublication {
            requested_generation: 2,
            published_generation: Some(2),
            pending: false,
        },
        views: [
            None,
            Some(view(
                1,
                CloudResetReasons::VIEW_REMOVED
                    .union(CloudResetReasons::WEATHER)
                    .union(CloudResetReasons::RESIZE),
            )),
            None,
            Some(view(3, CloudResetReasons::NONE)),
        ],
        readiness: SkyReadiness::pending(SkyReadinessStage::Atmosphere),
    };
    let d = serialize_diagnostics(&diagnostics).expect("pending diagnostics");
    assert_eq!(
        d.views.iter().map(|v| v.view).collect::<Vec<_>>(),
        vec![1, 3],
        "present slots in ascending slot order"
    );
    assert_eq!(
        d.views[0].reset_reasons,
        vec!["Resize", "Weather", "ViewRemoved"],
        "ordered reset bitset"
    );
    assert_eq!(d.views[1].reset_reasons, Vec::<&str>::new());
    assert_eq!((d.views[0].width, d.views[0].height), (1280, 720));
    assert_eq!(d.views[0].invocation, Some(7));
    assert_eq!(d.views[0].submitted_invocation, None);
    assert_eq!(d.views[0].sky_view_generation, 2);
    assert!(d.views[0].history_valid);
    assert!(!d.views[0].cloud_screen_valid);
    assert!(d.views[0].cloud_panorama_valid);
    assert_eq!(d.submitted_invocation, None);
    assert_eq!(d.environment.published_generation, Some(2));
    assert_eq!(d.readiness.status, "pending");
    assert_eq!(d.readiness.stage, "atmosphere");
    assert_eq!(d.readiness.error, None);
}

#[test]
fn diagnostics_failed_readiness_retains_the_originating_error() {
    let diagnostics = SkyDiagnostics {
        readiness: SkyReadiness::failed(SkyReadinessStage::CloudScreen, "gpu.device_lost"),
        ..SkyDiagnostics::default()
    };
    let d = serialize_diagnostics(&diagnostics).expect("failed diagnostics");
    assert_eq!(d.readiness.status, "failed");
    assert_eq!(d.readiness.stage, "cloudScreen");
    assert_eq!(d.readiness.error.as_deref(), Some("gpu.device_lost"));
    let ready = SkyDiagnostics {
        readiness: SkyReadiness {
            status: SkyReadinessStatus::Ready,
            stage: SkyReadinessStage::Submission,
            error: None,
        },
        ..SkyDiagnostics::default()
    };
    let d = serialize_diagnostics(&ready).expect("ready diagnostics");
    assert_eq!(
        (d.readiness.status, d.readiness.stage),
        ("ready", "submission")
    );
    assert_eq!(d.readiness.error, None);
}

#[test]
fn diagnostics_refuse_counters_beyond_safe_js_integers() {
    let mut diagnostics = SkyDiagnostics::default();
    diagnostics.weather_generation = MAX_SAFE_COUNTER;
    diagnostics.invocation = Some(MAX_SAFE_COUNTER);
    let ok = serialize_diagnostics(&diagnostics).expect("largest safe counter");
    assert_eq!(ok.weather_generation, MAX_SAFE_COUNTER);
    assert_eq!(ok.invocation, Some(MAX_SAFE_COUNTER));

    diagnostics.weather_generation = MAX_SAFE_COUNTER + 1;
    let error = serialize_diagnostics(&diagnostics).expect_err("overflowing counter");
    assert_eq!(error.code(), "sky.invalid");
    assert!(error.message().contains("weatherGeneration"), "{error}");

    let mut too_big_view = view(0, CloudResetReasons::NONE);
    too_big_view.sky_view_generation = MAX_SAFE_COUNTER + 1;
    diagnostics.weather_generation = 0;
    diagnostics.views[0] = Some(too_big_view);
    let error = serialize_diagnostics(&diagnostics).expect_err("view counter");
    assert_eq!(error.code(), "sky.invalid");
    assert!(
        error.message().contains("views.skyViewGeneration"),
        "{error}"
    );
}

#[test]
fn repeated_diagnostics_are_fresh_equal_snapshots_of_unchanged_state() {
    let mut diagnostics = SkyDiagnostics::default();
    diagnostics.views[0] = Some(view(0, CloudResetReasons::RESIZE));
    let before = diagnostics.clone();
    let first = serialize_diagnostics(&diagnostics).expect("first read");
    let second = serialize_diagnostics(&diagnostics).expect("second read");
    assert_eq!(first, second, "a repeated read observes the same state");
    assert_eq!(diagnostics, before, "reading never mutates the observation");

    let mut changed = first;
    changed.views.clear();
    changed.readiness.error = Some("changed".into());
    assert_eq!(second.views.len(), 1, "fresh objects are independent");
    assert_eq!(second.readiness.error, None);
}

/// One invocation advances the clock once in the engine; the m08 frame hook only observes
/// the accepted light, so a repeated invocation identity cannot advance time twice.
#[test]
fn clock_advances_once_per_invocation_and_m08_only_observes_it() {
    let engine = include_str!("../src/engine.rs");
    let state = include_str!("../src/state.rs");
    let observer = include_str!("../src/api/m08_sky_ibl.rs");
    assert!(state.contains("sky.advance_frame(self.sky_invocation_id(), input.dt_ms / 1000.0)?"));
    assert_eq!(engine.matches("s.advance_sky_frame(input)?;").count(), 1);
    let advance = engine.find("s.advance_sky_frame(input)?;").unwrap();
    assert!(engine.find(".set_time(input.now_ms)").unwrap() < advance);
    assert!(advance < engine.find("api::frame_all(s, gpu, input)?;").unwrap());
    assert!(observer.contains("fn frame("));
    assert!(!observer.contains("advance_frame("));
    assert!(!observer.contains("frame_index"));
}
