use raa_js::convert::*;

#[test]
fn handles_increase_and_are_never_reused() {
    let mut m = HandleMap::new("mesh");
    let a = m.insert(10u32);
    let b = m.insert(20u32);
    assert_eq!((a, b), (1, 2));
    assert_eq!(m.remove(a).unwrap(), 10);
    let c = m.insert(30u32);
    assert_eq!(c, 3);
    assert_eq!(m.len(), 2);
    assert_eq!(m.get(b).unwrap(), 20);
}

#[test]
fn stale_or_unknown_handle_is_refused() {
    let mut m = HandleMap::new("node");
    let a = m.insert(1u32);
    m.remove(a).unwrap();
    for h in [a, 0, 99] {
        let e = m.get(h).unwrap_err();
        assert_eq!(e.code(), UNKNOWN_HANDLE);
    }
    assert_eq!(m.remove(a).unwrap_err().code(), UNKNOWN_HANDLE);
    assert!(m.get(a).unwrap_err().to_string().contains("node"));
}

#[test]
fn trs_length_mismatch() {
    assert!(check_trs_len(2, 20).is_ok());
    assert_eq!(check_trs_len(2, 19).unwrap_err().code(), LENGTH_MISMATCH);
}

#[test]
fn decode_trs_known_vector() {
    let v: Vec<f32> = (0..20).map(|x| x as f32).collect();
    let t = decode_trs(&v, 1);
    assert_eq!(t.translation.to_array(), [10.0, 11.0, 12.0]);
    assert_eq!(t.rotation.to_array(), [13.0, 14.0, 15.0, 16.0]);
    assert_eq!(t.scale.to_array(), [17.0, 18.0, 19.0]);
}

#[test]
fn material_defaults_and_srgb() {
    let m = MaterialDesc::new([0.5, 0.5, 0.5]).to_material().unwrap();
    assert!((m.base_color[0] - 0.214).abs() < 1e-3);
    assert_eq!((m.roughness, m.metallic, m.emissive), (0.6, 0.0, [0.0; 3]));
    assert_eq!(m.model, raa_material::ShadingModel::PbrLite);
    let mut d = MaterialDesc::new([0.5; 3]);
    d.model = ModelName::Flat;
    assert_eq!(
        d.to_material().unwrap().model,
        raa_material::ShadingModel::Flat
    );
    assert!(ModelName::parse("x").is_err());
}

#[test]
fn invalid_roughness_is_refused() {
    let mut d = MaterialDesc::new([0.5; 3]);
    d.roughness = 1.5;
    assert_eq!(d.to_material().unwrap_err().code(), "material.invalid");
}

#[test]
fn quality_defaults_and_clamps() {
    assert_eq!(quality(None).unwrap().segments, 24);
    assert_eq!(quality(Some(1000.0)).unwrap().segments, 256);
    assert_eq!(quality(Some(1.0)).unwrap().segments, 3);
    assert_eq!(
        quality(Some(f64::NAN)).unwrap_err().code(),
        INVALID_ARGUMENT
    );
}

#[test]
fn api_error_keeps_code() {
    assert_eq!(api_error(DISPOSED, "gone").code(), "api.disposed");
    assert_eq!(api_error(UNKNOWN_PARAM, "x").code(), "api.unknown_param");
}

#[test]
fn engine_options_default_to_msaa_4_agx_ev_0() {
    let o = engine_options(&EngineOptionsInput::default()).unwrap();
    assert_eq!(o, EngineOptions::default());
    assert_eq!((o.max_instances, o.max_materials), (16384, 256));
    assert_eq!(o.power_preference, wgpu::PowerPreference::HighPerformance);
    assert_eq!(o.msaa, 4);
    assert_eq!(o.tone_mapper, raa_tonemap::ToneMapper::Agx);
    assert_eq!(o.exposure_ev, 0.0);
}

#[test]
fn engine_options_accept_the_lit_additions() {
    let input = EngineOptionsInput {
        msaa: Some(1.0),
        tone_mapper: Some("aces".to_string()),
        exposure_ev: Some(-1.5),
        ..EngineOptionsInput::default()
    };
    let o = engine_options(&input).unwrap();
    assert_eq!(
        (o.msaa, o.tone_mapper, o.exposure_ev),
        (1, raa_tonemap::ToneMapper::Aces, -1.5)
    );
    let input = EngineOptionsInput {
        msaa: Some(4.0),
        tone_mapper: Some("linear".to_string()),
        ..EngineOptionsInput::default()
    };
    let o = engine_options(&input).unwrap();
    assert_eq!(
        (o.msaa, o.tone_mapper),
        (4, raa_tonemap::ToneMapper::Linear)
    );
    assert_eq!(tone_mapper("agx"), Some(raa_tonemap::ToneMapper::Agx));
}

#[test]
fn engine_options_keep_the_v0_keys() {
    let input = EngineOptionsInput {
        max_instances: Some(1000.0),
        max_materials: Some(8.0),
        power_preference: Some("low-power".to_string()),
        ..EngineOptionsInput::default()
    };
    let o = engine_options(&input).unwrap();
    assert_eq!((o.max_instances, o.max_materials), (1000, 8));
    assert_eq!(o.power_preference, wgpu::PowerPreference::LowPower);
    for bad in [
        EngineOptionsInput {
            max_instances: Some(0.5),
            ..EngineOptionsInput::default()
        },
        EngineOptionsInput {
            power_preference: Some("fast".to_string()),
            ..EngineOptionsInput::default()
        },
    ] {
        assert_eq!(engine_options(&bad).unwrap_err().code(), INVALID_ARGUMENT);
    }
}

fn refusal(input: EngineOptionsInput) -> String {
    let e = engine_options(&input).unwrap_err();
    assert_eq!(e.code(), INVALID_ARGUMENT);
    e.to_string()
}

#[test]
fn engine_options_refuse_invalid_lit_values_naming_the_key() {
    let m = refusal(EngineOptionsInput {
        msaa: Some(2.0),
        ..EngineOptionsInput::default()
    });
    assert!(m.contains("msaa"), "{m}");
    let m = refusal(EngineOptionsInput {
        tone_mapper: Some("filmic".to_string()),
        ..EngineOptionsInput::default()
    });
    assert!(m.contains("toneMapper"), "{m}");
    for ev in [f64::NAN, f64::INFINITY, 1e300] {
        let m = refusal(EngineOptionsInput {
            exposure_ev: Some(ev),
            ..EngineOptionsInput::default()
        });
        assert!(m.contains("exposureEv"), "{m}");
    }
}
