use raa_geom::{MeshData, Quality, analysis};
use raa_prims::{PrimKind, PrimParams, generate, triangle_count};

fn cases() -> Vec<(String, PrimParams, Quality)> {
    let mut v = Vec::new();
    for kind in PrimKind::ALL {
        for set in ['A', 'B'] {
            for q in [Quality::LOW, Quality::LOD0, Quality::HIGH] {
                let name = format!("{} {set} {}", kind.as_str(), q.segments);
                v.push((name, PrimParams::reference_set(kind, set), q));
            }
        }
    }
    v
}

fn bytes(m: &MeshData) -> Vec<u8> {
    let mut b = Vec::new();
    m.write_vertex_bytes(&mut b);
    b
}

#[test]
fn two_generations_are_identical() {
    let all = cases();
    assert_eq!(all.len(), 48);
    for (name, p, q) in all {
        let (mut a, mut b) = (MeshData::new(), MeshData::new());
        generate(&p, q, &mut a).unwrap();
        generate(&p, q, &mut b).unwrap();
        assert_eq!(analysis::mesh_hash(&a), analysis::mesh_hash(&b), "{name}");
        assert_eq!(bytes(&a), bytes(&b), "{name}");
    }
}

#[test]
fn triangle_count_matches_generated() {
    let mut fails = Vec::new();
    for (name, p, q) in cases() {
        for q in [q, Quality::new(3), Quality::new(1000)] {
            let mut m = MeshData::new();
            generate(&p, q, &mut m).unwrap();
            let want = m.triangle_count() as u32;
            let got = triangle_count(&p, q);
            if got != want {
                fails.push(format!(
                    "{name} q{}: formula {got}, generated {want}",
                    q.segments
                ));
            }
        }
    }
    assert!(fails.is_empty(), "{}", fails.join("\n"));
}

#[test]
fn dirty_mesh_reuse_gives_same_hash() {
    let mut dirty = MeshData::new();
    generate(
        &PrimParams::reference_set(PrimKind::Torus, 'B'),
        Quality::HIGH,
        &mut dirty,
    )
    .unwrap();
    for (name, p, q) in cases() {
        let mut fresh = MeshData::new();
        generate(&p, q, &mut fresh).unwrap();
        generate(&p, q, &mut dirty).unwrap();
        assert_eq!(
            analysis::mesh_hash(&fresh),
            analysis::mesh_hash(&dirty),
            "{name}"
        );
    }
}
