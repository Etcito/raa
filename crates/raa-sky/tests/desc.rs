use raa_sky::{SkyDesc, SkyModel, SkyPreset};

const PRESETS: [&str; 4] = ["day", "snow", "jungle", "night"];

fn day() -> SkyDesc {
    SkyDesc::preset("day").expect("day preset")
}

#[test]
fn every_preset_is_valid() {
    for name in PRESETS {
        let sky = SkyDesc::preset(name).unwrap_or_else(|| panic!("preset {name} missing"));
        sky.validate()
            .unwrap_or_else(|e| panic!("preset {name} refused: {e}"));
    }
}

#[test]
fn unknown_preset_is_none() {
    assert!(SkyDesc::preset("dusk").is_none());
    assert!(SkyDesc::preset("").is_none());
    assert!(SkyDesc::preset("Day").is_none());
}

#[test]
fn each_invalid_field_is_refused() {
    type Edit = fn(&mut SkyDesc);
    let cases: [(&str, Edit); 8] = [
        ("zenith", |s| s.zenith[1] = f32::NAN),
        ("intensity", |s| s.intensity = -1.0),
        ("clouds.coverage", |s| s.clouds.coverage = 1.5),
        ("horizon_softness", |s| s.horizon_softness = 0.0),
        ("clouds.altitude_m", |s| s.clouds.altitude_m = 0.0),
        ("clouds.scale_m", |s| s.clouds.scale_m = -1.0),
        ("sun_disc_angular_radius", |s| {
            s.sun_disc_angular_radius = 0.5
        }),
        ("clouds.silver_lining", |s| s.clouds.silver_lining = 2.0),
    ];
    for (field, edit) in cases {
        let mut sky = day();
        edit(&mut sky);
        let err = sky.validate().expect_err(field);
        assert_eq!(err.code(), "sky.invalid", "{field}");
        let message = err.message();
        assert!(message.contains(field), "{field}: {message}");
    }
}

#[test]
fn presets_differ() {
    let day = day();
    let night = SkyDesc::preset("night").expect("night preset");
    assert_ne!(day, night);
    assert!(night.intensity < day.intensity * 0.1);
    let luma = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    assert!(luma(night.zenith) < luma(day.zenith));
}

#[test]
fn physical_presets_keep_the_s24_mapping_and_legacy_cases() {
    let day = day();
    assert_eq!(day.model, SkyModel::Physical);
    assert_eq!(day.profile, SkyPreset::Day);
    assert_eq!(day.atmosphere.ground_albedo, [0.3; 3]);
    assert_eq!(day.atmosphere.mie_multiplier, 1.0);

    let snow = SkyDesc::preset("snow").expect("snow");
    assert_eq!(snow.profile, SkyPreset::Snow);
    assert_eq!(snow.atmosphere.ground_albedo, [0.9; 3]);
    assert_eq!(snow.atmosphere.mie_multiplier, 1.5);

    let jungle = SkyDesc::preset("jungle").expect("jungle");
    assert_eq!(jungle.profile, SkyPreset::Jungle);
    assert_eq!(jungle.atmosphere.ground_albedo, [0.18, 0.26, 0.14]);
    assert_eq!(jungle.atmosphere.mie_multiplier, 1.2);

    let night = SkyDesc::preset("night").expect("night");
    assert_eq!(night.profile, SkyPreset::Night);
    assert_eq!(night.atmosphere.ground_albedo, [0.3; 3]);
    assert_eq!(night.atmosphere.mie_multiplier, 1.0);

    // The legacy comparison keeps the same authored artistic controls.
    let mut legacy = day.clone();
    legacy.model = SkyModel::Legacy;
    assert_eq!(legacy.zenith, day.zenith);
    assert_eq!(legacy.horizon, day.horizon);
    assert_eq!(legacy.intensity, day.intensity);
    legacy.validate().expect("legacy day");
}
