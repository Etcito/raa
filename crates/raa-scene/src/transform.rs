//! Local transforms and the generational node/instance handles of the scene arena.

use raa_math::{Mat4, Quat, Vec3};

/// A local transform: apply scale, then rotation, then translation.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Transform {
    /// Position of the node's origin in its parent's space.
    pub translation: Vec3,
    /// Orientation of the node in its parent's space.
    pub rotation: Quat,
    /// Scale along the node's own axes.
    pub scale: Vec3,
}

impl Transform {
    /// No translation, no rotation, unit scale.
    pub const IDENTITY: Transform = Transform {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    /// Builds a transform from its parts.
    pub const fn from_trs(translation: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Transform {
            translation,
            rotation,
            scale,
        }
    }

    /// The 4x4 matrix `T * R * S` of this transform.
    pub fn to_mat4(&self) -> Mat4 {
        Mat4::from_trs(self.translation, self.rotation, self.scale)
    }
}

/// Generational handle of a node in the scene arena.
///
/// The index says which slot, the generation says which occupant of that slot. A handle
/// whose slot was removed (or reused) no longer matches the arena and is refused as
/// stale.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct NodeId {
    index: u32,
    generation: u32,
}

impl NodeId {
    /// Packs the generation in the high half and the index in the low half.
    pub fn to_bits(self) -> u64 {
        ((self.generation() as u64) << 32) | self.index() as u64
    }

    /// Crate-private: the arena builds handles with this.
    #[allow(dead_code, reason = "the scene arena lands in the next tasks")]
    pub(crate) fn new(index: u32, generation: u32) -> Self {
        NodeId { index, generation }
    }

    /// Crate-private: slot of this handle in the arena.
    pub(crate) fn index(self) -> u32 {
        self.index
    }

    /// Crate-private: generation this handle expects in its slot.
    pub(crate) fn generation(self) -> u32 {
        self.generation
    }
}

/// Generational handle of an instance in the scene arena. See [`NodeId`].
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct InstanceId {
    index: u32,
    generation: u32,
}

impl InstanceId {
    /// Packs the generation in the high half and the index in the low half.
    pub fn to_bits(self) -> u64 {
        ((self.generation() as u64) << 32) | self.index() as u64
    }

    /// Crate-private: the arena builds handles with this.
    #[allow(dead_code, reason = "the scene arena lands in the next tasks")]
    pub(crate) fn new(index: u32, generation: u32) -> Self {
        InstanceId { index, generation }
    }

    /// Crate-private: slot of this handle in the arena.
    pub(crate) fn index(self) -> u32 {
        self.index
    }

    /// Crate-private: generation this handle expects in its slot.
    pub(crate) fn generation(self) -> u32 {
        self.generation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_bits_packs_generation_high_and_index_low() {
        let node = NodeId::new(0x1234_5678, 0x9abc_def0);
        assert_eq!(node.to_bits(), 0x9abc_def0_1234_5678);
        assert_eq!(node.index(), 0x1234_5678);
        assert_eq!(node.generation(), 0x9abc_def0);

        let instance = InstanceId::new(7, 1);
        assert_eq!(instance.to_bits(), 0x0000_0001_0000_0007);
        assert_eq!(instance.index(), 7);
        assert_eq!(instance.generation(), 1);
    }

    #[test]
    fn identity_transform_maps_to_identity_matrix() {
        assert_eq!(Transform::IDENTITY.to_mat4(), Mat4::IDENTITY);
    }

    #[test]
    fn from_trs_takes_the_parts_as_given() {
        let t = Transform::from_trs(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY, Vec3::splat(2.0));
        assert_eq!(t.translation, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(t.scale, Vec3::splat(2.0));
        assert_eq!(
            t.to_mat4(),
            Mat4::from_trs(t.translation, t.rotation, t.scale)
        );
    }
}
