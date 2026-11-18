//! The m34 merge: `merge_sharpening` keeps unspecified keys, refuses an out-of-range
//! `sharpness` with `post.invalid` and never changes its input.

use std::fs;
use std::path::Path;

use raa_sharpen::SharpenDesc;

#[allow(dead_code)]
#[path = "../src/api/m34_sharpening.rs"]
mod m34;

use m34::merge_sharpening;

fn current() -> SharpenDesc {
    SharpenDesc {
        enabled: false,
        sharpness: 0.9,
    }
}

#[test]
fn unspecified_keys_are_kept() {
    let c = current();
    assert_eq!(merge_sharpening(c, None, None).unwrap(), c);
    let s = merge_sharpening(c, None, Some(0.5)).unwrap();
    assert_eq!(
        s,
        SharpenDesc {
            enabled: false,
            sharpness: 0.5
        }
    );
    let e = merge_sharpening(c, Some(true), None).unwrap();
    assert_eq!(
        e,
        SharpenDesc {
            enabled: true,
            sharpness: 0.9
        }
    );
}

#[test]
fn out_of_range_sharpness_is_refused() {
    let c = current();
    for v in [1.5, -0.1, f64::NAN, f64::INFINITY] {
        let err = merge_sharpening(c, None, Some(v)).unwrap_err();
        assert_eq!(err.code(), "post.invalid", "{v}");
        assert!(err.to_string().contains("sharpness"), "{v}: {err}");
        assert_eq!(c, current());
    }
}

#[test]
fn registry_and_sources_name_the_api() {
    let path = Path::new(env!("OUT_DIR")).join("api_files.txt");
    let files = fs::read_to_string(&path).expect("api_files.txt");
    assert!(files.lines().any(|l| l.trim() == "m34_sharpening"));
    let src = include_str!("../src/api/m34_sharpening.rs");
    for name in ["setSharpening", "sharpening"] {
        assert!(src.contains(&format!("js_name = {name})")), "no {name}");
    }
}
