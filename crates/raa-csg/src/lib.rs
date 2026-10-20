//! Simple booleans of a primitive against a convex host: exact convex face cutting, otherwise
//! a typed refusal (`CsgError`).

use std::fmt;

use raa_geom::MeshData;

mod cases;
mod clip;
mod plane;
mod resolver;
mod weld;

pub use cases::{CsgCase, cases};
pub use clip::{intersect_convex, subtract_convex};
pub use plane::{Plane, convex_planes};
pub use resolver::CsgResolver;

/// Why a boolean was refused or failed.
#[derive(Debug, Clone, PartialEq)]
pub enum CsgError {
    HostNotConvex {
        kind: String,
    },
    ToolNotConvex {
        kind: String,
    },
    ToolHasChildren {
        node: String,
    },
    MultipleIntersect,
    OverlappingTools {
        a: String,
        b: String,
    },
    Coplanar {
        detail: String,
    },
    EmptyResult,
    NotWatertight {
        boundary_edges: u32,
        nonmanifold_edges: u32,
        inconsistent_edges: u32,
    },
    Degenerate(String),
}

impl CsgError {
    pub fn code(&self) -> &'static str {
        match self {
            CsgError::HostNotConvex { .. } => "csg.host_not_convex",
            CsgError::ToolNotConvex { .. } => "csg.tool_not_convex",
            CsgError::ToolHasChildren { .. } => "csg.tool_has_children",
            CsgError::MultipleIntersect => "csg.multiple_intersect",
            CsgError::OverlappingTools { .. } => "csg.overlapping_tools",
            CsgError::Coplanar { .. } => "csg.coplanar",
            CsgError::EmptyResult => "csg.empty_result",
            CsgError::NotWatertight { .. } => "csg.not_watertight",
            CsgError::Degenerate(_) => "csg.degenerate",
        }
    }
}

impl fmt::Display for CsgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CsgError::HostNotConvex { kind } => {
                write!(f, "host '{kind}' is not convex")
            }
            CsgError::ToolNotConvex { kind } => {
                write!(f, "tool '{kind}' is not convex")
            }
            CsgError::ToolHasChildren { node } => {
                write!(
                    f,
                    "tool '{node}' has children, a boolean tool must be a leaf"
                )
            }
            CsgError::MultipleIntersect => {
                write!(f, "more than one intersect tool on the same host")
            }
            CsgError::OverlappingTools { a, b } => {
                write!(f, "subtract tools '{a}' and '{b}' overlap")
            }
            CsgError::Coplanar { detail } => write!(f, "coplanar faces: {detail}"),
            CsgError::EmptyResult => write!(f, "the boolean result is empty"),
            CsgError::NotWatertight {
                boundary_edges,
                nonmanifold_edges,
                inconsistent_edges,
            } => {
                write!(
                    f,
                    "result is not watertight: {boundary_edges} boundary, \
                     {nonmanifold_edges} non-manifold, {inconsistent_edges} inconsistent edges"
                )
            }
            CsgError::Degenerate(m) => write!(f, "degenerate geometry: {m}"),
        }
    }
}

impl std::error::Error for CsgError {}

#[allow(dead_code)]
/// CSG_EPS: 1e-6 x max(1, diagonal).
pub(crate) fn csg_eps(diagonal: f64) -> f64 {
    1e-6 * diagonal.max(1.0)
}

#[allow(dead_code)]
/// Diagonal of the union of the bounding boxes of `meshes`, in f64 (0 when there are no points).
pub(crate) fn bbox_diagonal_union(meshes: &[&MeshData]) -> f64 {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for m in meshes {
        for p in &m.positions {
            for k in 0..3 {
                lo[k] = lo[k].min(f64::from(p[k]));
                hi[k] = hi[k].max(f64::from(p[k]));
            }
        }
    }
    if lo[0] > hi[0] {
        return 0.0;
    }
    let d = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// A convex polygon with per-vertex normals, counter-clockwise seen from outside.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Poly {
    pub pos: Vec<[f64; 3]>,
    pub nrm: Vec<[f64; 3]>,
}
