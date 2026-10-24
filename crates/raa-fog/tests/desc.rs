//! Q12: every preset validates, unknown names are `None`, every refusal is `post.invalid`
//! and names its field, and `write_env` fills the S10 fog fields only.

use raa_fog::FogDesc;
use raa_pipeline::EnvGpu;

const PRESETS: [&str; 5] = ["clear", "haze", "snow", "jungle", "night"];

#[test]
fn default_is_clear() {
    let fog = FogDesc::default();
    assert_eq!(fog.density, 0.0);
    assert_eq!(FogDesc::preset("clear"), Some(fog));
    fog.validate().expect("default");
}

#[test]
fn every_preset_validates() {
    for name in PRESETS {
        let fog = FogDesc::preset(name).unwrap_or_else(|| panic!("preset {name} missing"));
        fog.validate()
            .unwrap_or_else(|e| panic!("preset {name} refused: {e}"));
    }
}

#[test]
fn unknown_preset_is_none() {
    for name in ["", "dusk", "Clear", "HAZE", "fog", "day"] {
        assert!(FogDesc::preset(name).is_none(), "{name}");
    }
}

#[test]
fn presets_hold_the_starting_values() {
    let haze = FogDesc::preset("haze").expect("haze");
    assert_eq!(
        (haze.density, haze.height_falloff, haze.inscatter),
        (0.012, 0.12, 0.6)
    );
    assert_eq!((haze.aerial_strength, haze.aerial_distance_m), (0.5, 400.0));

    let snow = FogDesc::preset("snow").expect("snow");
    assert_eq!(snow.density, 0.02);
    assert_eq!((snow.aerial_strength, snow.aerial_distance_m), (0.6, 250.0));

    let jungle = FogDesc::preset("jungle").expect("jungle");
    assert_eq!((jungle.density, jungle.height_falloff), (0.03, 0.3));

    let night = FogDesc::preset("night").expect("night");
    assert_eq!((night.density, night.inscatter), (0.015, 0.2));
    assert!(night.color[2] > night.color[0] && night.color[2] > night.color[1]);
}

#[test]
fn every_refusal_has_code_and_names_the_field() {
    type Edit = fn(&mut FogDesc);
    let cases: [(&str, Edit); 19] = [
        ("density", |f| f.density = -0.01),
        ("density", |f| f.density = 1.01),
        ("density", |f| f.density = f32::NAN),
        ("height_falloff", |f| f.height_falloff = 0.0),
        ("height_falloff", |f| f.height_falloff = 10.01),
        ("height_falloff", |f| f.height_falloff = f32::INFINITY),
        ("base_height_m", |f| f.base_height_m = f32::NAN),
        ("base_height_m", |f| f.base_height_m = f32::NEG_INFINITY),
        ("start_m", |f| f.start_m = -0.1),
        ("start_m", |f| f.start_m = f32::NAN),
        ("color", |f| f.color = [1.0, -0.1, 1.0]),
        ("color", |f| f.color = [f32::NAN, 0.0, 0.0]),
        ("inscatter", |f| f.inscatter = -0.1),
        ("inscatter", |f| f.inscatter = 4.01),
        ("aerial_strength", |f| f.aerial_strength = -0.1),
        ("aerial_strength", |f| f.aerial_strength = 1.01),
        ("aerial_distance_m", |f| f.aerial_distance_m = 0.0),
        ("aerial_distance_m", |f| f.aerial_distance_m = -5.0),
        ("aerial_distance_m", |f| f.aerial_distance_m = f32::NAN),
    ];
    for (field, edit) in cases {
        let mut fog = FogDesc::default();
        edit(&mut fog);
        let err = fog.validate().expect_err(field);
        assert_eq!(err.code(), "post.invalid", "{field}");
        assert!(err.message().contains(field), "{field}: {err}");
    }
}

#[test]
fn range_limits_are_accepted() {
    let mut d = FogDesc {
        density: 1.0,
        height_falloff: 10.0,
        base_height_m: -120.0,
        start_m: 0.0,
        color: [0.0, 0.0, 0.0],
        inscatter: 4.0,
        aerial_strength: 1.0,
        aerial_distance_m: 0.001,
    };
    d.validate().expect("upper limits");
    (d.density, d.height_falloff, d.start_m) = (0.0, 0.000_001, 12.0);
    (d.inscatter, d.aerial_strength) = (0.0, 0.0);
    d.validate().expect("lower limits");
}

#[test]
fn write_env_places_the_s10_fields_and_keeps_the_rest() {
    let fog = FogDesc::preset("haze").expect("haze");
    let mut env = EnvGpu::default();
    env.sun_disc = [1.0, 2.0, 3.0, 4.0];
    env.zenith = [1.1, 2.1, 3.1, 4.1];
    env.horizon = [1.2, 2.2, 3.2, 4.2];
    env.ground = [1.3, 2.3, 3.3, 4.3];
    env.sky_params = [1.4, 2.4, 3.4, 4.4];
    env.background = [1.5, 2.5, 3.5, 4.5];
    env.clouds = [1.6, 2.6, 3.6, 4.6];
    env.clouds2 = [1.7, 2.7, 3.7, 4.7];
    env.cloud_color = [1.8, 2.8, 3.8, 4.8];
    env.fog = [9.0; 4];
    env.fog_color = [9.0; 4];
    env.aerial = [9.0; 4];
    let before = env.clone();

    fog.write_env(&mut env);

    assert_eq!(env.fog, [0.012, 0.12, 0.0, 0.0]);
    assert_eq!(env.fog_color, [0.84, 0.80, 0.72, 0.6]);
    assert_eq!(env.aerial, [0.5, 400.0, 0.0, 0.0]);
    assert_eq!(env.sun_disc, before.sun_disc);
    assert_eq!(env.zenith, before.zenith);
    assert_eq!(env.horizon, before.horizon);
    assert_eq!(env.ground, before.ground);
    assert_eq!(env.sky_params, before.sky_params);
    assert_eq!(env.background, before.background);
    assert_eq!(env.clouds, before.clouds);
    assert_eq!(env.clouds2, before.clouds2);
    assert_eq!(env.cloud_color, before.cloud_color);
}
