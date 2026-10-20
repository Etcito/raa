//! Transforms, instances, camera and lights.
//!
//! This crate owns the scene side of the engine: generic node and instance handles,
//! local/world transforms and the two lights the forward renderer shades with. It
//! depends on [`raa_math`] for math and [`raa_core`] for engine-wide ids.

use std::fmt;

mod camera;
mod graph;
mod light;
mod render_list;
mod transform;

pub use camera::{Camera, OrbitController};
pub use graph::Scene;
pub use light::{Ambient, DirectionalLight};
pub use render_list::{Batch, RenderItem, RenderList};
pub use transform::{InstanceId, NodeId, Transform};

/// Every refusal raa-scene can produce. Each variant has a stable code (see
/// [`SceneError::code`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SceneError {
    /// The node handle does not belong to this scene (removed, or from another scene).
    StaleNode,
    /// The instance handle does not belong to this scene (removed, or from another scene).
    StaleInstance,
    /// Reserved. A parent change would make a node an ancestor of itself.
    ///
    /// M2's API has no reparenting (a node's parent is fixed when it is added), so this
    /// variant is never produced yet. It exists so the code is stable when reparenting
    /// arrives.
    ParentCycle,
}

impl SceneError {
    /// Stable machine-readable code of the form `<area>.<snake_case>`.
    pub fn code(&self) -> &'static str {
        match self {
            SceneError::StaleNode => "scene.stale_node",
            SceneError::StaleInstance => "scene.stale_instance",
            SceneError::ParentCycle => "scene.parent_cycle",
        }
    }

    /// Human-readable message, without the code.
    pub fn message(&self) -> &'static str {
        match self {
            SceneError::StaleNode => "node handle is stale or belongs to another scene",
            SceneError::StaleInstance => "instance handle is stale or belongs to another scene",
            SceneError::ParentCycle => "parent change would create a cycle",
        }
    }
}

impl fmt::Display for SceneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code(), self.message())
    }
}

impl std::error::Error for SceneError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_variant_has_its_stable_code() {
        assert_eq!(SceneError::StaleNode.code(), "scene.stale_node");
        assert_eq!(SceneError::StaleInstance.code(), "scene.stale_instance");
        assert_eq!(SceneError::ParentCycle.code(), "scene.parent_cycle");
    }

    #[test]
    fn display_shows_code_then_message() {
        assert_eq!(
            SceneError::StaleNode.to_string(),
            "scene.stale_node: node handle is stale or belongs to another scene"
        );
    }
}
