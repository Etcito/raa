use raa_grading::GradingDesc;

const PRESETS: [&str; 5] = ["neutral", "candy", "ice", "jungle", "neon"];

fn preset(name: &str) -> GradingDesc {
    GradingDesc::preset(name).unwrap_or_else(|| panic!("preset {name} missing"))
}

#[test]
fn every_preset_is_valid() {
    for name in PRESETS {
        preset(name)
            .validate()
            .unwrap_or_else(|e| panic!("preset {name} refused: {e}"));
    }
}

#[test]
fn unknown_preset_is_none() {
    assert!(GradingDesc::preset("dusk").is_none());
    assert!(GradingDesc::preset("").is_none());
    assert!(GradingDesc::preset("Candy").is_none());
}

#[test]
fn neutral_is_the_identity_default() {
    let d = GradingDesc::default();
    assert_eq!(preset("neutral"), d);
    assert_eq!(d.temperature_k, 6500.0);
    assert_eq!((d.tint, d.exposure_ev, d.vibrance), (0.0, 0.0, 0.0));
    assert_eq!((d.contrast, d.saturation), (1.0, 1.0));
    assert_eq!(d.lift, [0.0; 3]);
    assert_eq!(d.gamma, [1.0; 3]);
    assert_eq!(d.gain, [1.0; 3]);
    assert_eq!(d.shadows_tint, [0.5; 3]);
    assert_eq!(d.highlights_tint, [0.5; 3]);
    assert_eq!((d.split_balance, d.vignette), (0.0, 0.0));
}

#[test]
fn presets_keep_their_character() {
    let neutral = GradingDesc::default();
    assert!(preset("candy").temperature_k > neutral.temperature_k);
    assert!(preset("ice").temperature_k < neutral.temperature_k);
    assert!(preset("jungle").tint < 0.0);
    assert!(preset("neon").contrast > neutral.contrast);
    for name in ["candy", "ice", "jungle", "neon"] {
        assert_ne!(preset(name), neutral, "{name}");
    }
}

#[test]
fn range_edges_are_accepted() {
    let d = GradingDesc::default();
    let low = GradingDesc {
        temperature_k: 1500.0,
        tint: -1.0,
        exposure_ev: -10.0,
        contrast: 0.01,
        saturation: 0.0,
        vibrance: -1.0,
        gamma: [0.01; 3],
        gain: [0.0; 3],
        shadows_tint: [0.0; 3],
        highlights_tint: [0.0; 3],
        split_balance: -1.0,
        vignette: 0.0,
        ..d
    };
    low.validate().expect("low edges");
    let high = GradingDesc {
        temperature_k: 15000.0,
        tint: 1.0,
        exposure_ev: 10.0,
        contrast: 4.0,
        saturation: 4.0,
        vibrance: 1.0,
        split_balance: 1.0,
        vignette: 1.0,
        ..d
    };
    high.validate().expect("high edges");
}

#[test]
fn each_refusal_names_its_field() {
    type Field = fn(&mut GradingDesc) -> &mut f32;
    let cases: [(&str, Field, [f32; 2]); 13] = [
        ("temperature_k", |d| &mut d.temperature_k, [1499.0, 15001.0]),
        ("tint", |d| &mut d.tint, [-1.1, 1.1]),
        ("exposure_ev", |d| &mut d.exposure_ev, [-10.5, 10.5]),
        ("contrast", |d| &mut d.contrast, [0.0, 4.5]),
        ("saturation", |d| &mut d.saturation, [-0.1, 4.1]),
        ("vibrance", |d| &mut d.vibrance, [-1.1, 1.1]),
        (
            "lift[1]",
            |d| &mut d.lift[1],
            [f32::INFINITY, f32::NEG_INFINITY],
        ),
        ("gamma[2]", |d| &mut d.gamma[2], [0.0, -1.0]),
        ("gain[0]", |d| &mut d.gain[0], [-0.5, -1.0]),
        ("shadows_tint[1]", |d| &mut d.shadows_tint[1], [-0.1, -2.0]),
        (
            "highlights_tint[2]",
            |d| &mut d.highlights_tint[2],
            [-0.1, -2.0],
        ),
        ("split_balance", |d| &mut d.split_balance, [-1.1, 1.1]),
        ("vignette", |d| &mut d.vignette, [-0.1, 1.1]),
    ];
    for (field, slot, bad) in cases {
        for v in [f32::NAN, bad[0], bad[1]] {
            let mut d = GradingDesc::default();
            *slot(&mut d) = v;
            let err = d.validate().expect_err(field);
            assert_eq!(err.code(), "post.invalid", "{field} = {v}");
            let message = err.message();
            assert!(
                message.contains(&format!("`{field}`")),
                "{field} = {v}: {message}"
            );
        }
    }
}
