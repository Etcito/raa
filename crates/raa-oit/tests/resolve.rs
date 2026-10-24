use raa_oit::node::{END, OitNode, f16_to_f32, f32_to_f16, pool_nodes};
use raa_oit::resolve_cpu::{collect_nearest, resolve_pixel};

type Frag = ([f32; 3], f32, f32);

fn list(frags: &[Frag]) -> (Vec<OitNode>, u32) {
    let mut nodes = Vec::new();
    let mut head = END;
    for (c, a, d) in frags {
        nodes.push(OitNode::pack(*c, *a, *d, head));
        head = (nodes.len() - 1) as u32;
    }
    (nodes, head)
}

#[test]
fn three_fragments_any_order_same_result() {
    let f: [Frag; 3] = [
        ([1.0, 0.0, 0.0], 0.5, 0.2),
        ([0.0, 1.0, 0.0], 0.25, 0.5),
        ([0.0, 0.0, 1.0], 0.75, 0.8),
    ];
    let perms = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let bg = [0.1, 0.2, 0.3];
    let mut first: Option<[f32; 3]> = None;
    for p in perms {
        let order: Vec<Frag> = p.iter().map(|&i| f[i]).collect();
        let (nodes, head) = list(&order);
        let r = resolve_pixel(&nodes, head, bg, 16);
        match first {
            None => first = Some(r),
            Some(x) => assert_eq!(x, r),
        }
    }
}

#[test]
fn far_to_near_composite_matches_hand_value() {
    // far: blue a=0.5 over bg (0,0,0) -> (0,0,0.5); near: red a=0.5 -> (0.5,0,0.25)
    let (nodes, head) = list(&[([1.0, 0.0, 0.0], 0.5, 0.1), ([0.0, 0.0, 1.0], 0.5, 0.9)]);
    let r = resolve_pixel(&nodes, head, [0.0; 3], 16);
    assert_eq!(r, [0.5, 0.0, 0.25]);
}

#[test]
fn twenty_fragments_drop_the_four_farthest() {
    let frags: Vec<Frag> = (0..20).map(|i| ([1.0; 3], 0.1, i as f32 * 0.01)).collect();
    let (nodes, head) = list(&frags);
    let kept = collect_nearest(&nodes, head, 16);
    assert_eq!(kept.len(), 16);
    for (i, k) in kept.iter().enumerate() {
        assert_eq!(k.depth, i as f32 * 0.01);
    }
    let mut rev = frags.clone();
    rev.reverse();
    let (n2, h2) = list(&rev);
    assert_eq!(collect_nearest(&n2, h2, 16), kept);
}

#[test]
fn pack_unpack_round_trips_within_f16_precision() {
    let c = [0.3, 1.7, 12.5];
    let n = OitNode::pack(c, 0.42, 0.73, 7);
    let back = OitNode::from_le_bytes(n.to_le_bytes());
    assert_eq!(back, n);
    let (cc, a) = back.color_alpha();
    for (x, y) in c.iter().zip(cc) {
        assert!((x - y).abs() <= x.abs() * 1.0 / 1024.0, "{x} vs {y}");
    }
    assert!((a - 0.42).abs() < 1e-3);
    assert_eq!(back.depth(), 0.73);
    assert_eq!(back.next, 7);
    assert_eq!(f32_to_f16(1.0), 0x3c00);
    assert_eq!(f32_to_f16(-2.0), 0xc000);
    assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
    assert_eq!(f16_to_f32(f32_to_f16(1e-6)), 17.0 * 2f32.powi(-24));
    assert!(f16_to_f32(f32_to_f16(1e6)).is_infinite());
}

#[test]
fn pool_nodes_sizing() {
    let big = 1u64 << 40;
    assert_eq!(pool_nodes(1280, 720, 8, big), 7_372_800);
    assert_eq!(pool_nodes(1280, 720, 8, 1600), 100);
    assert_eq!(pool_nodes(0, 720, 8, big), 1);
    assert_eq!(pool_nodes(1280, 720, 8, 0), 1);
    assert_eq!(pool_nodes(u32::MAX, u32::MAX, u32::MAX, u64::MAX), u32::MAX);
}
