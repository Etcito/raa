//! The flat, sorted list of visible instances the renderer draws each frame.

use raa_core::{MaterialId, MeshId};
use raa_math::Mat4;

/// One visible instance: what to draw, with which material, where.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RenderItem {
    /// Mesh to draw.
    pub mesh: MeshId,
    /// Material to shade it with.
    pub material: MaterialId,
    /// World matrix of the instance's node.
    pub world: Mat4,
}

/// A contiguous run of [`RenderItem`]s that share one mesh.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Batch {
    /// Mesh shared by every item of the run.
    pub mesh: MeshId,
    /// Index of the first item of the run in [`RenderList::items`].
    pub first: u32,
    /// Number of items in the run.
    pub count: u32,
}

/// Items and their per-mesh batches. Reused frame after frame: [`RenderList::clear`]
/// keeps the capacity, so refilling it does not allocate.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RenderList {
    /// Visible instances, sorted by (mesh, material, instance slot).
    pub items: Vec<RenderItem>,
    /// Contiguous runs of `items` per mesh, in item order.
    pub batches: Vec<Batch>,
}

impl RenderList {
    /// An empty list with room for `n` items and `n` batches.
    pub fn with_capacity(n: usize) -> Self {
        RenderList {
            items: Vec::with_capacity(n),
            batches: Vec::with_capacity(n),
        }
    }

    /// Empties the list and keeps its capacity.
    pub fn clear(&mut self) {
        self.items.clear();
        self.batches.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_capacity_reserves_items_and_batches() {
        let list = RenderList::with_capacity(16);
        assert!(list.items.capacity() >= 16);
        assert!(list.batches.capacity() >= 16);
        assert!(list.items.is_empty() && list.batches.is_empty());
    }

    #[test]
    fn clear_empties_and_keeps_capacity() {
        let mut list = RenderList::with_capacity(4);
        list.items.push(RenderItem {
            mesh: MeshId(1),
            material: MaterialId(2),
            world: Mat4::IDENTITY,
        });
        list.batches.push(Batch {
            mesh: MeshId(1),
            first: 0,
            count: 1,
        });
        let (items_cap, batches_cap) = (list.items.capacity(), list.batches.capacity());
        list.clear();
        assert!(list.items.is_empty() && list.batches.is_empty());
        assert_eq!(list.items.capacity(), items_cap);
        assert_eq!(list.batches.capacity(), batches_cap);
    }
}
