use raa_deform::{squash_normal, squash_point};

const AMOUNTS: [f32; 5] = [-0.9, -0.5, -0.2, 0.3, 1.0];

fn axes() -> [[f32; 3]; 3] {
    let r = std::f32::consts::FRAC_1_SQRT_2;
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [r, r, 0.0]]
}

const CUBE_VERTS: [[f32; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
];

// Twelve triangles with outward (counter-clockwise seen from outside) winding.
const CUBE_TRIS: [[usize; 3]; 12] = [
    [0, 3, 2],
    [0, 2, 1],
    [4, 5, 6],
    [4, 6, 7],
    [0, 1, 5],
    [0, 5, 4],
    [3, 7, 6],
    [3, 6, 2],
    [0, 4, 7],
    [0, 7, 3],
    [1, 2, 6],
    [1, 6, 5],
];

fn signed_volume(verts: &[[f32; 3]; 8]) -> f64 {
    CUBE_TRIS
        .iter()
        .map(|t| {
            let p = |i: usize| verts[t[i]].map(f64::from);
            let (a, b, c) = (p(0), p(1), p(2));
            let cross = [
                b[1] * c[2] - b[2] * c[1],
                b[2] * c[0] - b[0] * c[2],
                b[0] * c[1] - b[1] * c[0],
            ];
            (a[0] * cross[0] + a[1] * cross[1] + a[2] * cross[2]) / 6.0
        })
        .sum()
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn length(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}

fn normalise(a: [f32; 3]) -> [f32; 3] {
    let l = length(a);
    [a[0] / l, a[1] / l, a[2] / l]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn perpendicular_basis(n: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let helper = if n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let t1 = normalise(cross(n, helper));
    let t2 = cross(n, t1);
    (t1, t2)
}

fn unit_vectors() -> Vec<[f32; 3]> {
    let golden = 2.399_963_f32;
    (0..32)
        .map(|i| {
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / 32.0;
            let r = (1.0 - z * z).sqrt();
            let phi = i as f32 * golden;
            [r * phi.cos(), z, r * phi.sin()]
        })
        .collect()
}

#[test]
fn squashed_cube_keeps_its_volume() {
    for axis in axes() {
        for s in AMOUNTS {
            let mut verts = CUBE_VERTS;
            for v in verts.iter_mut() {
                *v = squash_point(*v, axis, s);
            }
            let volume = signed_volume(&verts);
            assert!(
                (volume - 1.0).abs() < 1e-4,
                "axis {axis:?} s {s}: volume {volume}"
            );
        }
    }
}

#[test]
fn squashed_normals_are_unit_and_perpendicular_to_squashed_edges() {
    let face_normals: [[f32; 3]; 6] = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    let mut normals: Vec<[f32; 3]> = face_normals.to_vec();
    normals.extend(unit_vectors());

    for axis in axes() {
        for s in AMOUNTS {
            for n in &normals {
                let out = squash_normal(*n, axis, s);
                assert!(
                    (length(out) - 1.0).abs() < 1e-5,
                    "axis {axis:?} s {s} normal {n:?}: length {}",
                    length(out)
                );
            }

            for n in face_normals {
                let (t1, t2) = perpendicular_basis(n);
                let out_n = squash_normal(n, axis, s);
                for t in [t1, t2] {
                    let edge = normalise(squash_point(t, axis, s));
                    assert!(
                        dot(out_n, edge).abs() < 1e-4,
                        "axis {axis:?} s {s} normal {n:?}: dot {}",
                        dot(out_n, edge)
                    );
                }
            }
        }
    }
}

#[test]
fn amounts_below_the_clamp_behave_as_the_clamp() {
    let p = [0.3, -0.7, 1.1];
    let n = normalise([0.2, 0.9, -0.4]);
    for axis in axes() {
        assert_eq!(squash_point(p, axis, -2.0), squash_point(p, axis, -0.9));
        assert_eq!(squash_normal(n, axis, -2.0), squash_normal(n, axis, -0.9));
    }
}
