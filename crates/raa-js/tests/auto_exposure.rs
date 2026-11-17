//! The m33 merge: `merge_auto_exposure` keeps unspecified keys, refuses each out-of-range key
//! with `post.invalid` naming the JS key and never changes its input.

use std::fs;
use std::path::Path;

use raa_exposure::AutoExposureDesc;

#[allow(dead_code)]
#[path = "../src/api/m33_auto_exposure.rs"]
mod m33;

use m33::{AutoExposurePatch, merge_auto_exposure};

fn table() -> AutoExposureDesc {
    AutoExposureDesc {
        enabled: false,
        compensation_ev: 0.263,
        low_percent: 0.2,
        high_percent: 0.7,
        speed_up: 2.4,
        speed_down: 1.1,
        min_ev: -6.0,
        max_ev: 6.0,
    }
}

#[test]
fn unspecified_keys_are_kept() {
    let empty = AutoExposurePatch::default();
    assert_eq!(merge_auto_exposure(table(), &empty).unwrap(), table());
    let patch = AutoExposurePatch {
        speed_up: Some(3.0),
        ..AutoExposurePatch::default()
    };
    let merged = merge_auto_exposure(table(), &patch).unwrap();
    assert_eq!(
        merged,
        AutoExposureDesc {
            speed_up: 3.0,
            ..table()
        }
    );
}

#[test]
fn out_of_range_keys_are_refused_and_input_is_kept() {
    let d = AutoExposurePatch::default;
    let cases = [
        (
            AutoExposurePatch {
                compensation_ev: Some(9.0),
                ..d()
            },
            "compensationEv",
        ),
        (
            AutoExposurePatch {
                low_percent: Some(1.2),
                ..d()
            },
            "lowPercent",
        ),
        (
            AutoExposurePatch {
                high_percent: Some(-0.5),
                ..d()
            },
            "highPercent",
        ),
        (
            AutoExposurePatch {
                speed_up: Some(0.0),
                ..d()
            },
            "speedUp",
        ),
        (
            AutoExposurePatch {
                speed_down: Some(-1.0),
                ..d()
            },
            "speedDown",
        ),
        (
            AutoExposurePatch {
                min_ev: Some(f64::NAN),
                ..d()
            },
            "minEv",
        ),
        (
            AutoExposurePatch {
                max_ev: Some(f64::INFINITY),
                ..d()
            },
            "maxEv",
        ),
        (
            AutoExposurePatch {
                low_percent: Some(0.8),
                high_percent: Some(0.3),
                ..d()
            },
            "lowPercent",
        ),
    ];
    for (patch, key) in cases {
        let input = table();
        let copy = input;
        let err = merge_auto_exposure(input, &patch).unwrap_err();
        assert_eq!(err.code(), "post.invalid", "{key}");
        assert!(err.to_string().contains(key), "{key}: {err}");
        assert_eq!(input, copy, "{key}");
    }
}

#[test]
fn registry_and_sources_name_the_api() {
    let path = Path::new(env!("OUT_DIR")).join("api_files.txt");
    let files = fs::read_to_string(&path).expect("api_files.txt");
    assert!(files.lines().any(|l| l.trim() == "m33_auto_exposure"));
    let src = include_str!("../src/api/m33_auto_exposure.rs");
    for name in ["setAutoExposure", "autoExposure", "autoExposureState"] {
        assert!(src.contains(&format!("js_name = {name})")), "no {name}");
    }
    let m07 = include_str!("../src/api/m07_pipeline.rs");
    assert!(m07.contains("AutoExposureFeature"));
    assert!(m07.contains("fn view_auto_exposure"));
}
