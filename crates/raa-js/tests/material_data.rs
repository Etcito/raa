//! m25 material data words: the registry exports addMaterialData, setMaterialData and
//! removeMaterialData; the base and float conversions of `src/api/m25_material_data.rs` run
//! natively, and addSurfaceMaterial reads `data`.

use std::fs;
use std::path::Path;

#[allow(dead_code)]
#[path = "../src/api/m25_material_data.rs"]
mod m25;

use m25::{data_base, float_words};

#[test]
fn registry_exports_the_material_data_methods() {
    let path = Path::new(env!("OUT_DIR")).join("api_registry.rs");
    let reg = fs::read_to_string(&path).expect("api_registry.rs");
    assert!(reg.contains("mod m25_material_data;"));
    assert!(reg.contains("m25_material_data::install(ctx)?;"));
    let src = include_str!("../src/api/m25_material_data.rs");
    for name in ["addMaterialData", "setMaterialData", "removeMaterialData"] {
        assert!(src.contains(&format!("js_name = {name}")), "no {name}");
    }
    let materials = include_str!("../src/api/m10_materials.rs");
    assert!(materials.contains("m.data = surface_data(&desc)?;"));
}

#[test]
fn bases_are_whole_numbers() {
    assert_eq!(data_base(Some(0.0)).unwrap(), 0);
    assert_eq!(data_base(Some(4096.0)).unwrap(), 4096);
    for bad in [
        None,
        Some(-1.0),
        Some(2.5),
        Some(f64::NAN),
        Some(f64::INFINITY),
        Some(5e9),
    ] {
        let e = data_base(bad).unwrap_err();
        assert_eq!(e.code(), "api.invalid_argument", "{bad:?}");
        assert!(e.to_string().contains("base"), "{e}");
    }
}

#[test]
fn floats_are_stored_as_their_bits() {
    assert_eq!(
        float_words(&[1.0, -2.5, 0.0]),
        [0x3f80_0000, 0xc020_0000, 0]
    );
}
