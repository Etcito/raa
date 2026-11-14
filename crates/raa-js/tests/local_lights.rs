//! m28 local-light descriptor parsing (S17): the pure merge and its property-presence
//! rules run natively. The wasm32 `Reflect` reader is pinned by `tests/api_registry.rs`.

#![cfg(not(target_arch = "wasm32"))]

use raa_core::RaaError;
use raa_lit::{LocalLight, LocalLightKind, LocalLights, MAX_LIGHT_HANDLE, MAX_LOCAL_LIGHTS};

#[allow(dead_code)]
#[path = "../src/api/m28_lights.rs"]
mod m28;

use m28::{
    AREA_FORM_FACTOR_FEATURE_ID, AreaLightShading, INVALID_ARGUMENT, INVALID_DESC, LocalLightPatch,
    Patch, Properties, Raw, add, handle_arg, merge, parse, parse_area_shading, remove,
    srgb_to_linear, update,
};

/// A native stand-in for a JS description object: a key is present exactly when it is
/// listed, and a present key may hold `Raw::Undefined`. This mirrors `Reflect::has` plus
/// `Reflect::get`, so an omitted key and an explicit `undefined` are distinguishable.
struct Mock(Vec<(&'static str, Raw)>);

impl Mock {
    fn new(fields: &[(&'static str, Raw)]) -> Self {
        Self(fields.to_vec())
    }

    fn empty() -> Self {
        Self(Vec::new())
    }
}

impl Properties for Mock {
    fn read(&self, key: &str) -> Result<Raw, RaaError> {
        Ok(self
            .0
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, value)| value.clone())
            .unwrap_or(Raw::Absent))
    }
}

fn code(e: &RaaError) -> &'static str {
    match e {
        RaaError::FrameGraph { code, .. } => code,
        _ => "other",
    }
}

#[test]
fn absent_keys_use_defaults_on_add() {
    let patch = parse(&Mock::empty()).unwrap();
    assert_eq!(merge(LocalLight::default(), &patch), LocalLight::default());
}

#[test]
fn absent_keys_keep_the_base_on_update() {
    let base = LocalLight {
        intensity: 9.0,
        range: 12.0,
        ..LocalLight::default()
    };
    let patch = parse(&Mock::new(&[("intensity", Raw::Number(3.0))])).unwrap();
    let merged = merge(base, &patch);
    assert_eq!(merged.intensity, 3.0);
    assert_eq!(merged.range, base.range);
    assert_eq!(merged.kind, base.kind);
}

#[test]
fn present_undefined_restores_the_s17_default() {
    let base = LocalLight {
        kind: LocalLightKind::Area,
        position: [3.0, 4.0, 5.0],
        direction: [1.0, 0.0, 0.0],
        color: [0.2, 0.3, 0.4],
        intensity: 9.0,
        range: 12.0,
        angle_deg: 60.0,
        inner_angle_deg: Some(45.0),
        size: [2.0, 3.0],
        radius: 4.0,
        shadows: true,
    };
    let patch = parse(&Mock::new(&[
        ("kind", Raw::Undefined),
        ("position", Raw::Undefined),
        ("direction", Raw::Undefined),
        ("color", Raw::Undefined),
        ("intensity", Raw::Undefined),
        ("range", Raw::Undefined),
        ("angleDeg", Raw::Undefined),
        ("innerAngleDeg", Raw::Undefined),
        ("size", Raw::Undefined),
        ("radius", Raw::Undefined),
        ("shadows", Raw::Undefined),
    ]))
    .unwrap();
    assert_eq!(merge(base, &patch), LocalLight::default());
}

#[test]
fn parse_maps_presence_to_keep_reset_and_set() {
    let patch = parse(&Mock::new(&[
        ("intensity", Raw::Number(4.0)),
        ("innerAngleDeg", Raw::Undefined),
    ]))
    .unwrap();
    assert_eq!(patch.intensity, Patch::Set(4.0));
    assert_eq!(patch.inner_angle_deg, Patch::Reset);
    assert_eq!(patch.range, Patch::Keep);
    assert_eq!(patch.kind, Patch::Keep);
    assert_eq!(LocalLightPatch::empty(), LocalLightPatch::empty());
}

#[test]
fn inner_angle_distinguishes_omitted_from_undefined() {
    let authored = LocalLight {
        angle_deg: 60.0,
        inner_angle_deg: Some(45.0),
        ..LocalLight::default()
    };
    let omitted = parse(&Mock::empty()).unwrap();
    assert_eq!(merge(authored, &omitted).inner_angle_deg, Some(45.0));
    let cleared = parse(&Mock::new(&[("innerAngleDeg", Raw::Undefined)])).unwrap();
    let merged = merge(authored, &cleared);
    assert_eq!(merged.inner_angle_deg, None);
    assert_eq!(merged.effective_inner_angle_deg(), 48.0);
    let set = parse(&Mock::new(&[("innerAngleDeg", Raw::Number(30.0))])).unwrap();
    assert_eq!(merge(authored, &set).inner_angle_deg, Some(30.0));
}

#[test]
fn color_is_converted_from_srgb_exactly_once() {
    let patch = parse(&Mock::new(&[("color", Raw::Numbers(vec![0.5, 0.0, 1.0]))])).unwrap();
    let merged = merge(LocalLight::default(), &patch);
    let expected = [0.5f32, 0.0, 1.0].map(srgb_to_linear);
    assert_eq!(merged.color, expected);
    assert_ne!(expected, [0.5, 0.0, 1.0]);
    // A later update that omits `color` keeps the linear value: no second conversion.
    let keep = parse(&Mock::empty()).unwrap();
    assert_eq!(merge(merged, &keep).color, expected);
    // A reset goes back to the default white, again without a double conversion.
    let reset = parse(&Mock::new(&[("color", Raw::Undefined)])).unwrap();
    assert_eq!(merge(merged, &reset).color, [1.0, 1.0, 1.0]);
}

#[test]
fn kind_change_preserves_valid_authored_fields() {
    let base = LocalLight {
        angle_deg: 60.0,
        inner_angle_deg: Some(45.0),
        range: 12.0,
        ..LocalLight::default()
    };
    let patch = parse(&Mock::new(&[("kind", Raw::String("spot".into()))])).unwrap();
    let merged = merge(base, &patch);
    assert_eq!(merged.kind, LocalLightKind::Spot);
    assert_eq!(
        (merged.angle_deg, merged.inner_angle_deg, merged.range),
        (60.0, Some(45.0), 12.0)
    );
}

#[test]
fn malformed_fields_are_refused_with_light_invalid_desc() {
    let cases = [
        Mock::new(&[("position", Raw::Numbers(vec![1.0, 2.0]))]),
        Mock::new(&[("size", Raw::Numbers(vec![1.0, 2.0, 3.0]))]),
        Mock::new(&[("intensity", Raw::String("bright".into()))]),
        Mock::new(&[("shadows", Raw::Number(1.0))]),
        Mock::new(&[("kind", Raw::String("cone".into()))]),
        Mock::new(&[("color", Raw::Other)]),
    ];
    for mock in cases {
        assert_eq!(code(&parse(&mock).unwrap_err()), INVALID_DESC);
    }
}

#[test]
fn merged_angle_and_inner_are_validated_by_the_descriptor() {
    let base = LocalLight {
        angle_deg: 60.0,
        inner_angle_deg: Some(45.0),
        ..LocalLight::default()
    };
    let patch = parse(&Mock::new(&[("angleDeg", Raw::Number(40.0))])).unwrap();
    assert_eq!(
        code(&merge(base, &patch).validate().unwrap_err()),
        INVALID_DESC
    );
}

#[test]
fn handles_are_positive_safe_integers_before_narrowing() {
    assert_eq!(handle_arg("handle", 1.0).unwrap(), 1);
    assert_eq!(
        handle_arg("handle", MAX_LIGHT_HANDLE as f64).unwrap(),
        MAX_LIGHT_HANDLE
    );
    for bad in [
        0.0,
        -1.0,
        1.5,
        f64::NAN,
        f64::INFINITY,
        9_007_199_254_740_992.0, // 2^53, one past the safe-integer maximum
    ] {
        assert_eq!(
            code(&handle_arg("handle", bad).unwrap_err()),
            INVALID_ARGUMENT
        );
    }
}

// ---- area-light shading (contract `area-light-shading-api`) ----

#[test]
fn area_light_shading_parses_ltc_and_form_factor() {
    let ltc = parse_area_shading(&Mock::new(&[("mode", Raw::String("ltc".into()))])).unwrap();
    assert_eq!(ltc, AreaLightShading::Ltc);
    let form_factor =
        parse_area_shading(&Mock::new(&[("mode", Raw::String("formFactor".into()))])).unwrap();
    assert_eq!(form_factor, AreaLightShading::FormFactor);
}

#[test]
fn area_light_shading_refuses_every_other_value() {
    let cases = [
        Mock::empty(),
        Mock::new(&[("mode", Raw::Undefined)]),
        Mock::new(&[("mode", Raw::String("LTC".into()))]),
        Mock::new(&[("mode", Raw::String("formfactor".into()))]),
        Mock::new(&[("mode", Raw::Number(1.0))]),
    ];
    for mock in cases {
        let err = parse_area_shading(&mock).unwrap_err();
        assert_eq!(err.code(), INVALID_ARGUMENT);
        assert_eq!(
            err.message(),
            "setAreaLightShading: 'mode' must be 'ltc' or 'formFactor'"
        );
    }
}

#[test]
fn area_form_factor_feature_id_is_39() {
    assert_eq!(AREA_FORM_FACTOR_FEATURE_ID, 39);
}

#[test]
fn area_light_shading_as_str_round_trips() {
    for mode in [AreaLightShading::Ltc, AreaLightShading::FormFactor] {
        let parsed =
            parse_area_shading(&Mock::new(&[("mode", Raw::String(mode.as_str().into()))])).unwrap();
        assert_eq!(parsed, mode);
    }
    assert_eq!(AreaLightShading::Ltc.as_str(), "ltc");
    assert_eq!(AreaLightShading::FormFactor.as_str(), "formFactor");
}

// ---- S17 add/update/remove against the renderer-owned registry ----

#[test]
fn add_uses_s17_defaults_and_returns_a_live_handle() {
    let mut lights = LocalLights::new();
    let patch = parse(&Mock::empty()).unwrap();
    let handle = add(&mut lights, &patch).unwrap();
    assert!(handle >= 1);
    assert_eq!(*lights.get(handle).unwrap(), LocalLight::default());
}

#[test]
fn add_and_update_normalize_the_stored_direction() {
    let mut lights = LocalLights::new();
    let patch = parse(&Mock::new(&[(
        "direction",
        Raw::Numbers(vec![3.0, 4.0, 0.0]),
    )]))
    .unwrap();
    let handle = add(&mut lights, &patch).unwrap();
    let stored = *lights.get(handle).unwrap();
    assert!((stored.direction[0] - 0.6).abs() < 1e-6);
    assert!((stored.direction[1] - 0.8).abs() < 1e-6);
    assert_eq!(stored.direction[2], 0.0);
}

#[test]
fn add_converts_colour_from_srgb_exactly_once() {
    let mut lights = LocalLights::new();
    let patch = parse(&Mock::new(&[("color", Raw::Numbers(vec![0.5, 0.0, 1.0]))])).unwrap();
    let expected = [0.5f32, 0.0, 1.0].map(srgb_to_linear);
    let handle = add(&mut lights, &patch).unwrap();
    assert_eq!(lights.get(handle).unwrap().color, expected);
    // A later update that omits `color` keeps the linear value, unconverted again.
    let keep = parse(&Mock::empty()).unwrap();
    update(&mut lights, handle, &keep).unwrap();
    assert_eq!(lights.get(handle).unwrap().color, expected);
    assert_ne!(expected, [0.5, 0.0, 1.0]);
}

#[test]
fn capacity_refuses_the_257th_without_mutating_the_registry() {
    let mut lights = LocalLights::new();
    let patch = parse(&Mock::empty()).unwrap();
    let handles: Vec<_> = (0..MAX_LOCAL_LIGHTS)
        .map(|_| add(&mut lights, &patch).unwrap())
        .collect();
    assert_eq!((lights.active_count(), handles.len()), (256, 256));
    let err = add(&mut lights, &patch).unwrap_err();
    assert_eq!(code(&err), "lights.invalid");
    assert!(err.to_string().contains("256"), "{}", err);
    assert_eq!(lights.active_count(), 256);
}

#[test]
fn stale_handles_refuse_and_are_never_reused() {
    let mut lights = LocalLights::new();
    let patch = parse(&Mock::empty()).unwrap();
    let stale = add(&mut lights, &patch).unwrap();
    let fresh = add(&mut lights, &patch).unwrap();
    remove(&mut lights, stale).unwrap();
    assert_eq!(
        code(&update(&mut lights, stale, &patch).unwrap_err()),
        "api.unknown_handle"
    );
    assert_eq!(
        code(&remove(&mut lights, stale).unwrap_err()),
        "api.unknown_handle"
    );
    // The next handle is larger: the removed value is never aliased.
    let next = add(&mut lights, &patch).unwrap();
    assert!(next > fresh);
}

#[test]
fn kind_changes_retain_valid_fields_and_metadata() {
    let mut lights = LocalLights::new();
    let created = parse(&Mock::new(&[
        ("kind", Raw::String("spot".into())),
        ("angleDeg", Raw::Number(60.0)),
        ("innerAngleDeg", Raw::Number(45.0)),
        ("range", Raw::Number(12.0)),
        ("radius", Raw::Number(2.5)),
        ("shadows", Raw::Bool(true)),
    ]))
    .unwrap();
    let handle = add(&mut lights, &created).unwrap();
    let changes = parse(&Mock::new(&[("kind", Raw::String("area".into()))])).unwrap();
    update(&mut lights, handle, &changes).unwrap();
    let stored = *lights.get(handle).unwrap();
    assert_eq!(stored.kind, LocalLightKind::Area);
    assert_eq!(
        (stored.angle_deg, stored.inner_angle_deg, stored.range),
        (60.0, Some(45.0), 12.0)
    );
    assert_eq!((stored.radius, stored.shadows), (2.5, true));
}

#[test]
fn refused_updates_leave_the_stored_light_unchanged() {
    let mut lights = LocalLights::new();
    let created = parse(&Mock::new(&[
        ("angleDeg", Raw::Number(60.0)),
        ("innerAngleDeg", Raw::Number(45.0)),
        ("intensity", Raw::Number(3.0)),
    ]))
    .unwrap();
    let handle = add(&mut lights, &created).unwrap();
    let before = *lights.get(handle).unwrap();
    // The retained explicit inner angle must fit the merged outer angle.
    let narrowed = parse(&Mock::new(&[("angleDeg", Raw::Number(40.0))])).unwrap();
    assert_eq!(
        code(&update(&mut lights, handle, &narrowed).unwrap_err()),
        INVALID_DESC
    );
    // A malformed merged field is refused without clearing the others.
    let malformed = parse(&Mock::new(&[("intensity", Raw::Number(f64::NAN))])).unwrap();
    assert_eq!(
        code(&update(&mut lights, handle, &malformed).unwrap_err()),
        INVALID_DESC
    );
    assert_eq!(*lights.get(handle).unwrap(), before);
    // Clearing the explicit inner angle restores the default, so the narrower cone is valid.
    let cleared = parse(&Mock::new(&[
        ("innerAngleDeg", Raw::Undefined),
        ("angleDeg", Raw::Number(40.0)),
    ]))
    .unwrap();
    update(&mut lights, handle, &cleared).unwrap();
    let stored = *lights.get(handle).unwrap();
    assert_eq!((stored.angle_deg, stored.inner_angle_deg), (40.0, None));
    assert_eq!(stored.intensity, 3.0);
}
