//! m22 material textures: the registry exports addTexture and removeTexture; the side and
//! texel conversions of `src/api/m22_textures.rs` run natively.

use std::fs;
use std::path::Path;

use raa_lit::TexelData;

#[allow(dead_code)]
#[path = "../src/api/m22_textures.rs"]
mod m22;

use m22::{Texels, texture_side};

#[test]
fn registry_exports_add_texture_and_remove_texture() {
    let path = Path::new(env!("OUT_DIR")).join("api_registry.rs");
    let reg = fs::read_to_string(&path).expect("api_registry.rs");
    assert!(reg.contains("mod m22_textures;"));
    assert!(reg.contains("m22_textures::install(ctx)?;"));
    let src = include_str!("../src/api/m22_textures.rs");
    for name in ["addTexture", "removeTexture"] {
        assert!(src.contains(&format!("js_name = {name}")), "no {name}");
    }
}

#[test]
fn sides_are_whole_numbers() {
    assert_eq!(texture_side("width", Some(0.0)).unwrap(), 0);
    assert_eq!(texture_side("width", Some(256.0)).unwrap(), 256);
    for bad in [None, Some(-1.0), Some(1.5), Some(f64::NAN), Some(f64::INFINITY), Some(5e9)] {
        let e = texture_side("width", bad).unwrap_err();
        assert_eq!(e.code(), "api.invalid_argument", "{bad:?}");
        assert!(e.to_string().contains("width"), "{e}");
    }
}

#[test]
fn texels_keep_their_kind() {
    let bytes = Texels::Rgba8(vec![1, 2, 3, 4]);
    assert!(matches!(bytes.data(), TexelData::Rgba8(&[1, 2, 3, 4])));
    let floats = Texels::Float(vec![0.0, 0.5, 1.0, 2.0]);
    match floats.data() {
        TexelData::Float(f) => assert_eq!(f, [0.0, 0.5, 1.0, 2.0]),
        other => panic!("{other:?}"),
    }
    assert_eq!(raa_lit::quantize(2.0), 255);
    assert_eq!(raa_lit::quantize(-1.0), 0);
}
