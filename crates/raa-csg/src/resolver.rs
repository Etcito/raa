//! `BooleanResolver` backed by exact convex cutting.
//!
//! Fixed order: no tools copies the host; then the refusals ToolHasChildren, MultipleIntersect,
//! HostNotConvex and ToolNotConvex; then the single intersect tool; then every subtract tool
//! at once against the post-intersect host planes.

use raa_geom::MeshData;
use raa_loader::{BoolOp, BooleanHost, BooleanResolver, BooleanTool, LoadError};

use crate::{
    CsgError, bbox_diagonal_union, convex_planes, csg_eps, intersect_convex, subtract_convex,
};

/// Resolves subtract/intersect tools on a convex host, or refuses with `boolean.csg.*`.
#[derive(Debug, Clone, Copy, Default)]
pub struct CsgResolver;

/// `boolean.` + `CsgError::code()`, as a static string.
fn boolean_code(e: &CsgError) -> &'static str {
    match e {
        CsgError::HostNotConvex { .. } => "boolean.csg.host_not_convex",
        CsgError::ToolNotConvex { .. } => "boolean.csg.tool_not_convex",
        CsgError::ToolHasChildren { .. } => "boolean.csg.tool_has_children",
        CsgError::MultipleIntersect => "boolean.csg.multiple_intersect",
        CsgError::OverlappingTools { .. } => "boolean.csg.overlapping_tools",
        CsgError::Coplanar { .. } => "boolean.csg.coplanar",
        CsgError::EmptyResult => "boolean.csg.empty_result",
        CsgError::NotWatertight { .. } => "boolean.csg.not_watertight",
        CsgError::Degenerate(_) => "boolean.csg.degenerate",
    }
}

/// The path stays empty: the loader fills it with the first tool's JSON pointer, so the
/// message names the offending node(s) itself.
fn refuse(e: &CsgError, context: &str) -> LoadError {
    LoadError {
        code: boolean_code(e),
        path: String::new(),
        message: format!("{e} ({context})"),
    }
}

fn copy_mesh(src: &MeshData, out: &mut MeshData) {
    out.clear();
    out.positions.extend_from_slice(&src.positions);
    out.normals.extend_from_slice(&src.normals);
    out.indices.extend_from_slice(&src.indices);
}

fn tool_list(host: &BooleanHost<'_>, tools: &[&BooleanTool<'_>]) -> String {
    let mut s = format!("host node '{}'", host.node_id);
    for (i, t) in tools.iter().enumerate() {
        s.push_str(if i == 0 { ", tool node(s) " } else { ", " });
        s.push('\'');
        s.push_str(t.node_id);
        s.push('\'');
    }
    s
}

impl BooleanResolver for CsgResolver {
    fn resolve(
        &self,
        host: &BooleanHost<'_>,
        tools: &[BooleanTool<'_>],
        out: &mut MeshData,
    ) -> Result<(), LoadError> {
        // (1) No tools: same as NoBooleans.
        if tools.is_empty() {
            copy_mesh(host.mesh, out);
            return Ok(());
        }

        // (2) A tool must be a leaf.
        if let Some(t) = tools.iter().find(|t| t.has_children) {
            let e = CsgError::ToolHasChildren {
                node: t.node_id.to_string(),
            };
            return Err(refuse(&e, &format!("on host node '{}'", host.node_id)));
        }

        // (3) At most one intersect tool.
        let intersects: Vec<&BooleanTool<'_>> =
            tools.iter().filter(|t| t.op == BoolOp::Intersect).collect();
        if let [a, b, ..] = intersects.as_slice() {
            return Err(refuse(
                &CsgError::MultipleIntersect,
                &format!(
                    "intersect nodes '{}' and '{}' on host node '{}'",
                    a.node_id, b.node_id, host.node_id
                ),
            ));
        }
        let subtracts: Vec<&BooleanTool<'_>> =
            tools.iter().filter(|t| t.op == BoolOp::Subtract).collect();

        let mut all: Vec<&MeshData> = Vec::with_capacity(tools.len() + 1);
        all.push(host.mesh);
        all.extend(tools.iter().map(|t| t.mesh));
        let eps = csg_eps(bbox_diagonal_union(&all));

        // (4) Host convexity: declared by the kind, then checked on the mesh.
        let host_err = || CsgError::HostNotConvex {
            kind: host.kind.to_string(),
        };
        if !host.convex {
            return Err(refuse(
                &host_err(),
                &format!("host node '{}'", host.node_id),
            ));
        }
        let mut host_planes = convex_planes(host.mesh, eps)
            .map_err(|e| refuse(&host_err(), &format!("host node '{}': {e}", host.node_id)))?;

        // (5) Tool convexity, in document order.
        for t in tools {
            let tool_err = CsgError::ToolNotConvex {
                kind: t.kind.to_string(),
            };
            if !t.convex {
                return Err(refuse(
                    &tool_err,
                    &format!("tool node '{}' on host node '{}'", t.node_id, host.node_id),
                ));
            }
            if let Err(e) = convex_planes(t.mesh, eps) {
                return Err(refuse(
                    &tool_err,
                    &format!(
                        "tool node '{}' on host node '{}': {e}",
                        t.node_id, host.node_id
                    ),
                ));
            }
        }

        // (6) The intersect first: the result stays convex.
        let mut scratch = MeshData::new();
        let current: &MeshData = match intersects.first() {
            Some(t) => {
                intersect_convex(host.mesh, t.mesh, &mut scratch)
                    .map_err(|e| refuse(&e, &tool_list(host, &[t])))?;
                &scratch
            }
            None => host.mesh,
        };

        // (7) All subtract tools at once against the post-intersect planes.
        if subtracts.is_empty() {
            copy_mesh(current, out);
            return Ok(());
        }
        if !intersects.is_empty() {
            host_planes = convex_planes(current, eps).map_err(|e| {
                refuse(
                    &e,
                    &format!(
                        "planes of host node '{}' after intersect: {}",
                        host.node_id,
                        tool_list(host, &intersects)
                    ),
                )
            })?;
        }
        let meshes: Vec<&MeshData> = subtracts.iter().map(|t| t.mesh).collect();
        subtract_convex(current, &host_planes, &meshes, out)
            .map_err(|e| refuse(&e, &tool_list(host, &subtracts)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_geom::builtin::DebugCube;
    use raa_geom::{GeometryProvider, ParamValue, Params, Quality};

    fn cube(half: f32) -> MeshData {
        let mut m = MeshData::new();
        let params = Params::new().with("half", ParamValue::Scalar(half));
        DebugCube
            .generate(&params, Quality::LOD0, &mut m)
            .expect("cube");
        m
    }

    fn host(mesh: &MeshData, convex: bool) -> BooleanHost<'_> {
        BooleanHost {
            node_id: "body",
            kind: "box",
            convex,
            mesh,
        }
    }

    fn tool<'a>(id: &'a str, op: BoolOp, mesh: &'a MeshData) -> BooleanTool<'a> {
        BooleanTool {
            node_id: id,
            kind: "sphere",
            op,
            convex: true,
            has_children: false,
            mesh,
        }
    }

    fn refusal(h: &BooleanHost<'_>, tools: &[BooleanTool<'_>]) -> LoadError {
        let mut out = MeshData::new();
        CsgResolver.resolve(h, tools, &mut out).unwrap_err()
    }

    fn assert_names(e: &LoadError, ids: &[&str]) {
        assert!(!e.message.is_empty());
        assert_eq!(e.path, "");
        for id in ids {
            assert!(e.message.contains(id), "{id} not in {:?}", e.message);
        }
    }

    #[test]
    fn tool_with_children_is_refused_first() {
        let (h, t) = (cube(1.0), cube(0.5));
        let mut child = tool("hole", BoolOp::Subtract, &t);
        child.has_children = true;
        child.convex = false;
        let other = tool("cut", BoolOp::Intersect, &t);
        let e = refusal(&host(&h, false), &[other, other, child]);
        assert_eq!(e.code, "boolean.csg.tool_has_children");
        assert_names(&e, &["hole", "body"]);
    }

    #[test]
    fn two_intersect_tools_are_refused() {
        let (h, t) = (cube(1.0), cube(0.5));
        let e = refusal(
            &host(&h, false),
            &[
                tool("a", BoolOp::Intersect, &t),
                tool("dent", BoolOp::Subtract, &t),
                tool("b", BoolOp::Intersect, &t),
            ],
        );
        assert_eq!(e.code, "boolean.csg.multiple_intersect");
        assert_names(&e, &["'a'", "'b'", "body"]);
    }

    #[test]
    fn non_convex_host_is_refused() {
        let (h, t) = (cube(1.0), cube(0.5));
        let mut bad = tool("hole", BoolOp::Subtract, &t);
        bad.convex = false;
        let e = refusal(&host(&h, false), &[bad]);
        assert_eq!(e.code, "boolean.csg.host_not_convex");
        assert_names(&e, &["body", "box"]);
    }

    #[test]
    fn non_convex_tool_is_refused() {
        let (h, t) = (cube(1.0), cube(0.5));
        let mut bad = tool("hole", BoolOp::Subtract, &t);
        bad.convex = false;
        bad.kind = "torus";
        let e = refusal(&host(&h, true), &[tool("ok", BoolOp::Subtract, &t), bad]);
        assert_eq!(e.code, "boolean.csg.tool_not_convex");
        assert_names(&e, &["hole", "torus"]);
    }

    #[test]
    fn no_tools_copies_host_byte_identical() {
        let h = cube(1.0);
        let mut out = MeshData {
            positions: vec![[9.0; 3]],
            normals: vec![[0.0, 1.0, 0.0]],
            indices: vec![0, 0, 0],
        };
        CsgResolver
            .resolve(&host(&h, false), &[], &mut out)
            .unwrap();
        assert_eq!(out, h);
    }

    #[test]
    fn every_refusal_code_has_the_boolean_prefix() {
        let errs = [
            CsgError::HostNotConvex { kind: "k".into() },
            CsgError::ToolNotConvex { kind: "k".into() },
            CsgError::ToolHasChildren { node: "n".into() },
            CsgError::MultipleIntersect,
            CsgError::OverlappingTools {
                a: "a".into(),
                b: "b".into(),
            },
            CsgError::Coplanar { detail: "d".into() },
            CsgError::EmptyResult,
            CsgError::NotWatertight {
                boundary_edges: 1,
                nonmanifold_edges: 0,
                inconsistent_edges: 0,
            },
            CsgError::Degenerate("x".into()),
        ];
        for e in &errs {
            let l = refuse(e, "host node 'body'");
            assert_eq!(l.code, format!("boolean.{}", e.code()));
            assert!(l.message.contains("body"));
        }
    }
}
