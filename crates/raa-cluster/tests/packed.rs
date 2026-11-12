use std::mem::size_of;

use raa_cluster::packed::{
    PACKED_CLUSTER_SIZE, PACKED_NODE_SIZE, PACKED_VERTEX_SIZE, PackedCluster, PackedNode,
    PackedVertex, pack_counts, pack_triangle, unpack_triangle,
};

fn sample_vertex() -> PackedVertex {
    PackedVertex {
        data: [1, 2, 3, 4, 5, 6, 7, 8],
    }
}

fn sample_cluster() -> PackedCluster {
    PackedCluster {
        center: [1.5, 2.5, 3.5],
        radius: 4.5,
        lod_center: [5.5, 6.5, 7.5],
        lod_radius: 8.5,
        lod_error: 9.5,
        vertex_offset: 0x1122_3344,
        triangle_offset: 0x5566_7788,
        counts: 0x99aa_bbcc,
    }
}

fn sample_node() -> PackedNode {
    PackedNode {
        center: [1.5, 2.5, 3.5],
        radius: 4.5,
        lod_center: [5.5, 6.5, 7.5],
        lod_radius: 8.5,
        max_parent_error: 9.5,
        first: 0x1122_3344,
        count: 0x5566_7788,
        is_leaf: 0x99aa_bbcc,
    }
}

fn assert_f32_at(bytes: &[u8], offset: usize, expected: f32) {
    assert_eq!(&bytes[offset..offset + 4], &expected.to_le_bytes());
}

fn assert_u32_at(bytes: &[u8], offset: usize, expected: u32) {
    assert_eq!(&bytes[offset..offset + 4], &expected.to_le_bytes());
}

#[test]
fn packed_record_sizes_are_the_s29_sizes() {
    assert_eq!(PACKED_VERTEX_SIZE, 16);
    assert_eq!(PACKED_CLUSTER_SIZE, 48);
    assert_eq!(PACKED_NODE_SIZE, 48);
    assert_eq!(size_of::<PackedVertex>(), PACKED_VERTEX_SIZE);
    assert_eq!(size_of::<PackedCluster>(), PACKED_CLUSTER_SIZE);
    assert_eq!(size_of::<PackedNode>(), PACKED_NODE_SIZE);
}

#[test]
fn packed_vertex_field_offsets() {
    let mut bytes = Vec::new();
    sample_vertex().write_le_bytes(&mut bytes);
    assert_eq!(bytes.len(), PACKED_VERTEX_SIZE);
    for (index, expected) in [1u16, 2, 3, 4, 5, 6, 7, 8].iter().enumerate() {
        let offset = index * 2;
        assert_eq!(&bytes[offset..offset + 2], &expected.to_le_bytes());
    }
}

#[test]
fn packed_cluster_field_offsets() {
    let mut bytes = Vec::new();
    sample_cluster().write_le_bytes(&mut bytes);
    assert_eq!(bytes.len(), PACKED_CLUSTER_SIZE);
    assert_f32_at(&bytes, 0, 1.5);
    assert_f32_at(&bytes, 4, 2.5);
    assert_f32_at(&bytes, 8, 3.5);
    assert_f32_at(&bytes, 12, 4.5);
    assert_f32_at(&bytes, 16, 5.5);
    assert_f32_at(&bytes, 20, 6.5);
    assert_f32_at(&bytes, 24, 7.5);
    assert_f32_at(&bytes, 28, 8.5);
    assert_f32_at(&bytes, 32, 9.5);
    assert_u32_at(&bytes, 36, 0x1122_3344);
    assert_u32_at(&bytes, 40, 0x5566_7788);
    assert_u32_at(&bytes, 44, 0x99aa_bbcc);
}

#[test]
fn packed_node_field_offsets() {
    let mut bytes = Vec::new();
    sample_node().write_le_bytes(&mut bytes);
    assert_eq!(bytes.len(), PACKED_NODE_SIZE);
    assert_f32_at(&bytes, 0, 1.5);
    assert_f32_at(&bytes, 4, 2.5);
    assert_f32_at(&bytes, 8, 3.5);
    assert_f32_at(&bytes, 12, 4.5);
    assert_f32_at(&bytes, 16, 5.5);
    assert_f32_at(&bytes, 20, 6.5);
    assert_f32_at(&bytes, 24, 7.5);
    assert_f32_at(&bytes, 28, 8.5);
    assert_f32_at(&bytes, 32, 9.5);
    assert_u32_at(&bytes, 36, 0x1122_3344);
    assert_u32_at(&bytes, 40, 0x5566_7788);
    assert_u32_at(&bytes, 44, 0x99aa_bbcc);
}

#[test]
fn packed_records_round_trip_and_append() {
    let vertex = sample_vertex();
    let mut bytes = Vec::new();
    vertex.write_le_bytes(&mut bytes);
    vertex.write_le_bytes(&mut bytes);
    assert_eq!(bytes.len(), 2 * PACKED_VERTEX_SIZE);
    assert_eq!(PackedVertex::read_le_bytes(&bytes), Some(vertex));

    let cluster = sample_cluster();
    let mut bytes = Vec::new();
    cluster.write_le_bytes(&mut bytes);
    cluster.write_le_bytes(&mut bytes);
    assert_eq!(bytes.len(), 2 * PACKED_CLUSTER_SIZE);
    assert_eq!(PackedCluster::read_le_bytes(&bytes), Some(cluster));

    let node = sample_node();
    let mut bytes = Vec::new();
    node.write_le_bytes(&mut bytes);
    node.write_le_bytes(&mut bytes);
    assert_eq!(bytes.len(), 2 * PACKED_NODE_SIZE);
    assert_eq!(PackedNode::read_le_bytes(&bytes), Some(node));
}

#[test]
fn packed_records_refuse_short_slices() {
    let mut bytes = Vec::new();
    sample_vertex().write_le_bytes(&mut bytes);
    assert_eq!(PackedVertex::read_le_bytes(&bytes[..15]), None);

    let mut bytes = Vec::new();
    sample_cluster().write_le_bytes(&mut bytes);
    assert_eq!(PackedCluster::read_le_bytes(&bytes[..47]), None);

    let mut bytes = Vec::new();
    sample_node().write_le_bytes(&mut bytes);
    assert_eq!(PackedNode::read_le_bytes(&bytes[..47]), None);
}

#[test]
fn triangle_packing_round_trips() {
    let packed = pack_triangle(0, 127, 254);
    assert_eq!(packed, 127 << 8 | 254 << 16);
    assert_eq!(unpack_triangle(packed), [0, 127, 254]);
}

#[test]
fn packed_counts_read_back_the_three_fields() {
    let counts = pack_counts(255, 128, 7);
    assert_eq!(counts, 255 | 128 << 8 | 7 << 16);
    assert_eq!(counts >> 24, 0);
    let cluster = PackedCluster {
        counts,
        ..PackedCluster::default()
    };
    assert_eq!(cluster.vertex_count(), 255);
    assert_eq!(cluster.triangle_count(), 128);
    assert_eq!(cluster.level(), 7);
    let saturated = PackedCluster {
        counts: pack_counts(1, 2, 300),
        ..PackedCluster::default()
    };
    assert_eq!(saturated.level(), 255);
    assert_eq!(saturated.vertex_count(), 1);
    assert_eq!(saturated.triangle_count(), 2);
}
