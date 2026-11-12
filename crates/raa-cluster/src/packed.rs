//! Packed S29 GPU records (`PackedVertex`, `PackedCluster`, `PackedNode`) with their
//! little-endian writers and readers, the triangle packing and the section constants.
//! Adapted from solstice (MIT), Copyright (c) 2026 SamG-Coder:
//! `crates/solstice-asset/src/gpu.rs`, revision 7e465b5.

pub const MAX_CLUSTER_TRIANGLES: usize = 128;
pub const MAX_CLUSTER_VERTICES: usize = 255;
pub const ROOT_PARENT_ERROR: f32 = 1.0e30;
pub const CLUSTER_ALGORITHM: u32 = 1;
pub const CLUSTER_SECTION_VERSION: u32 = 1;
pub const PACKED_VERTEX_SIZE: usize = 16;
pub const PACKED_CLUSTER_SIZE: usize = 48;
pub const PACKED_NODE_SIZE: usize = 48;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PackedVertex {
    pub data: [u16; 8],
}

impl PackedVertex {
    pub fn write_le_bytes(&self, out: &mut Vec<u8>) {
        for word in self.data {
            out.extend_from_slice(&word.to_le_bytes());
        }
    }

    pub fn read_le_bytes(bytes: &[u8]) -> Option<Self> {
        let bytes = bytes.get(..PACKED_VERTEX_SIZE)?;
        let mut data = [0u16; 8];
        for (index, word) in data.iter_mut().enumerate() {
            *word = read_u16(bytes, index * 2)?;
        }
        Some(Self { data })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PackedCluster {
    pub center: [f32; 3],
    pub radius: f32,
    pub lod_center: [f32; 3],
    pub lod_radius: f32,
    pub lod_error: f32,
    pub vertex_offset: u32,
    pub triangle_offset: u32,
    pub counts: u32,
}

impl PackedCluster {
    pub fn write_le_bytes(&self, out: &mut Vec<u8>) {
        for value in self.center {
            push_f32(out, value);
        }
        push_f32(out, self.radius);
        for value in self.lod_center {
            push_f32(out, value);
        }
        push_f32(out, self.lod_radius);
        push_f32(out, self.lod_error);
        push_u32(out, self.vertex_offset);
        push_u32(out, self.triangle_offset);
        push_u32(out, self.counts);
    }

    pub fn read_le_bytes(bytes: &[u8]) -> Option<Self> {
        let bytes = bytes.get(..PACKED_CLUSTER_SIZE)?;
        Some(Self {
            center: [
                read_f32(bytes, 0)?,
                read_f32(bytes, 4)?,
                read_f32(bytes, 8)?,
            ],
            radius: read_f32(bytes, 12)?,
            lod_center: [
                read_f32(bytes, 16)?,
                read_f32(bytes, 20)?,
                read_f32(bytes, 24)?,
            ],
            lod_radius: read_f32(bytes, 28)?,
            lod_error: read_f32(bytes, 32)?,
            vertex_offset: read_u32(bytes, 36)?,
            triangle_offset: read_u32(bytes, 40)?,
            counts: read_u32(bytes, 44)?,
        })
    }

    pub fn vertex_count(&self) -> u32 {
        self.counts & 0xff
    }

    pub fn triangle_count(&self) -> u32 {
        (self.counts >> 8) & 0xff
    }

    pub fn level(&self) -> u32 {
        (self.counts >> 16) & 0xff
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PackedNode {
    pub center: [f32; 3],
    pub radius: f32,
    pub lod_center: [f32; 3],
    pub lod_radius: f32,
    pub max_parent_error: f32,
    pub first: u32,
    pub count: u32,
    pub is_leaf: u32,
}

impl PackedNode {
    pub fn write_le_bytes(&self, out: &mut Vec<u8>) {
        for value in self.center {
            push_f32(out, value);
        }
        push_f32(out, self.radius);
        for value in self.lod_center {
            push_f32(out, value);
        }
        push_f32(out, self.lod_radius);
        push_f32(out, self.max_parent_error);
        push_u32(out, self.first);
        push_u32(out, self.count);
        push_u32(out, self.is_leaf);
    }

    pub fn read_le_bytes(bytes: &[u8]) -> Option<Self> {
        let bytes = bytes.get(..PACKED_NODE_SIZE)?;
        Some(Self {
            center: [
                read_f32(bytes, 0)?,
                read_f32(bytes, 4)?,
                read_f32(bytes, 8)?,
            ],
            radius: read_f32(bytes, 12)?,
            lod_center: [
                read_f32(bytes, 16)?,
                read_f32(bytes, 20)?,
                read_f32(bytes, 24)?,
            ],
            lod_radius: read_f32(bytes, 28)?,
            max_parent_error: read_f32(bytes, 32)?,
            first: read_u32(bytes, 36)?,
            count: read_u32(bytes, 40)?,
            is_leaf: read_u32(bytes, 44)?,
        })
    }
}

pub fn pack_triangle(a: u8, b: u8, c: u8) -> u32 {
    u32::from(a) | u32::from(b) << 8 | u32::from(c) << 16
}

pub fn unpack_triangle(packed: u32) -> [u8; 3] {
    [packed as u8, (packed >> 8) as u8, (packed >> 16) as u8]
}

pub fn pack_counts(vertex_count: u32, triangle_count: u32, level: u32) -> u32 {
    (vertex_count & 0xff) | (triangle_count & 0xff) << 8 | level.min(255) << 16
}

fn push_f32(out: &mut Vec<u8>, value: f32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn read_array<const N: usize>(bytes: &[u8], offset: usize) -> Option<[u8; N]> {
    bytes.get(offset..offset.checked_add(N)?)?.try_into().ok()
}

fn read_f32(bytes: &[u8], offset: usize) -> Option<f32> {
    read_array::<4>(bytes, offset).map(f32::from_le_bytes)
}

fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    read_array::<4>(bytes, offset).map(u32::from_le_bytes)
}

fn read_u16(bytes: &[u8], offset: usize) -> Option<u16> {
    read_array::<2>(bytes, offset).map(u16::from_le_bytes)
}
