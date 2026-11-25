//! Optional grid inputs shared by mesh reflection planning and validation.

use raa_loader::json::JsonValue;

use crate::catalog::PortSpec;
use crate::format::{GraphDoc, NodeDoc};

pub(crate) const GRID_DIFFUSE_TYPE: &str = "mesh.grid-diffuse";

pub(crate) const SPHERICAL_INPUT: &str = "sphericalLookup";
pub(crate) const SPHERICAL_OUTPUT: &str = "lookup";
pub(crate) const SPHERICAL_SOURCE: &[&str] = &["pass.gi.spherical.lookup"];
pub(crate) const VOLUME_PORT: &str = "volumeAtlas";
pub(crate) const VOLUME_SOURCE: &[&str] = &["pass.gi.volume.volumeAtlas"];
pub(crate) const VOLUME_TYPE: &str = "pass.gi.volume";
pub(crate) const INHERIT_GI: &str = "inheritGi";

pub(crate) fn mesh_grid_consumer(ty: &str) -> bool {
    matches!(
        ty,
        "mesh.reflection" | "mesh.fitted-specular" | GRID_DIFFUSE_TYPE
    )
}

pub(crate) fn mesh_grid_port(port: &str) -> bool {
    matches!(port, SPHERICAL_INPUT | VOLUME_PORT)
}

/// An authored grid link selects the grid set for this consumer. Disabling inheritance
/// also allows the last authored link to be disconnected without restoring a default.
pub(crate) fn input_defaults(
    doc: &GraphDoc,
    node: &NodeDoc,
    port: &PortSpec,
) -> &'static [&'static str] {
    if mesh_grid_consumer(&node.type_id)
        && mesh_grid_port(port.key)
        && (node
            .params
            .iter()
            .any(|(key, value)| key == INHERIT_GI && *value == JsonValue::Bool(false))
            || doc
                .links
                .iter()
                .any(|link| link.to.node == node.id && mesh_grid_port(&link.to.port)))
    {
        &[]
    } else {
        port.default_source
    }
}
