use raa_geom::{MeshData, Quality};
use raa_prims::{PrimParams, generate, reference};

const FIXTURE: &str = include_str!("fixtures/sdf-leaf-reference.json");

#[test]
fn all_48_cases_match_sdf_reference() {
    let fx = reference::parse_fixture(FIXTURE).expect("fixture parses");
    assert_eq!(fx.cases.len(), 48);
    let mut fails = Vec::new();
    for case in &fx.cases {
        let name = format!("{} {} {}", case.kind.as_str(), case.set, case.quality);
        let p = PrimParams::reference_set(case.kind, case.set);
        let mut m = MeshData::new();
        if let Err(e) = generate(&p, Quality::new(case.quality), &mut m) {
            fails.push(format!("{name}: generate failed: {e}"));
            continue;
        }
        let d = reference::compare(&fx, case, &m);
        if !d.ok {
            fails.push(format!("{name}: {}", d.message));
        }
    }
    assert!(
        fails.is_empty(),
        "{} failures:\n{}",
        fails.len(),
        fails.join("\n")
    );
}
