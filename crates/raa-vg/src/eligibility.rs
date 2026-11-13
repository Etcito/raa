//! Path eligibility of one instance (S31, S32): which instances may be drawn virtualized, the
//! first refusal of a rank table, and the path an instance is reported on.
//!
//! The rank table is evaluated in order and the first match wins: rank 0 skinned, rank 1
//! deformed or with a material vertex function, rank 2 a material class other than Opaque.
//! Skinned, deformed, transparent and fur content stays on the forward path.

use raa_lit::{DrawClass, InstanceShape};

use crate::error::{DEFORMED_UNSUPPORTED, SKINNED_UNSUPPORTED, TRANSPARENT_MATERIAL};

/// The refusal code of `shape` by the rank table, or `None` when it is eligible.
pub fn refusal(shape: &InstanceShape) -> Option<&'static str> {
    if shape.skinned {
        Some(SKINNED_UNSUPPORTED)
    } else if shape.deformed || shape.vertex_fn {
        Some(DEFORMED_UNSUPPORTED)
    } else if shape.class != DrawClass::Opaque {
        Some(TRANSPARENT_MATERIAL)
    } else {
        None
    }
}

/// The reason for the code [`refusal`] gives `shape`, or `None` when it is eligible. Rank 2
/// names the class: `transparent` or `fur`.
pub fn refusal_message(shape: &InstanceShape) -> Option<String> {
    let code = refusal(shape)?;
    Some(match code {
        SKINNED_UNSUPPORTED => {
            "skinned instance: posed skins stay on the forward path".to_string()
        }
        DEFORMED_UNSUPPORTED if shape.deformed => {
            "deformed instance (squash or jiggle): deformed instances stay on the forward path"
                .to_string()
        }
        DEFORMED_UNSUPPORTED => {
            "material vertex function: materials that define material_vertex stay on the forward path"
                .to_string()
        }
        _ => {
            let class = match shape.class {
                DrawClass::Fur => "fur",
                _ => "transparent",
            };
            format!("material class {class}: only opaque materials are virtualized")
        }
    })
}

/// The path an instance is reported on: `Virtual`, or `Forward` with its refusal code and
/// message. `pending` is true while the instance is requested and eligible but a mesh has no
/// ready cluster data yet (it draws on forward until every mesh is ready).
#[derive(Clone, Debug, PartialEq)]
pub enum InstancePath {
    /// Drawn through the virtualized path.
    Virtual,
    /// Drawn on the forward path.
    Forward {
        pending: bool,
        code: Option<&'static str>,
        message: Option<String>,
    },
}
