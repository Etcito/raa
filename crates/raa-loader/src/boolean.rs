//! Boolean (subtract/intersect) delegation.

use raa_geom::MeshData;

use crate::LoadError;

/// The boolean operation a tool node applies to its host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolOp {
    Subtract,
    Intersect,
}

impl BoolOp {
    fn name(self) -> &'static str {
        match self {
            BoolOp::Subtract => "subtract",
            BoolOp::Intersect => "intersect",
        }
    }
}

/// The node whose mesh the tools modify, in its own local space.
#[derive(Debug, Clone, Copy)]
pub struct BooleanHost<'a> {
    pub node_id: &'a str,
    pub kind: &'a str,
    pub convex: bool,
    pub mesh: &'a MeshData,
}

/// A subtract/intersect child, its mesh already in host local space.
#[derive(Debug, Clone, Copy)]
pub struct BooleanTool<'a> {
    pub node_id: &'a str,
    pub kind: &'a str,
    pub op: BoolOp,
    pub convex: bool,
    pub has_children: bool,
    pub mesh: &'a MeshData,
}

/// Evaluates a host and its tools into `out`.
pub trait BooleanResolver {
    fn resolve(
        &self,
        host: &BooleanHost<'_>,
        tools: &[BooleanTool<'_>],
        out: &mut MeshData,
    ) -> Result<(), LoadError>;
}

/// The resolver used when no CSG is installed: copies tool-less hosts, refuses any tool.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoBooleans;

impl BooleanResolver for NoBooleans {
    fn resolve(
        &self,
        host: &BooleanHost<'_>,
        tools: &[BooleanTool<'_>],
        out: &mut MeshData,
    ) -> Result<(), LoadError> {
        match tools.first() {
            None => {
                out.clear();
                out.positions.extend_from_slice(&host.mesh.positions);
                out.normals.extend_from_slice(&host.mesh.normals);
                out.indices.extend_from_slice(&host.mesh.indices);
                Ok(())
            }
            Some(tool) => Err(LoadError::new(
                "boolean.unsupported",
                "",
                format!(
                    "no boolean resolver installed: {} by node '{}' on host '{}'",
                    tool.op.name(),
                    tool.node_id,
                    host.node_id
                ),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri() -> MeshData {
        MeshData {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            indices: vec![0, 1, 2],
        }
    }

    fn host(mesh: &MeshData) -> BooleanHost<'_> {
        BooleanHost {
            node_id: "body",
            kind: "box",
            convex: true,
            mesh,
        }
    }

    #[test]
    fn no_booleans_refuses_tools_with_unsupported() {
        let m = tri();
        let tools = [
            BooleanTool {
                node_id: "hole",
                kind: "cylinder",
                op: BoolOp::Intersect,
                convex: true,
                has_children: false,
                mesh: &m,
            },
            BooleanTool {
                node_id: "other",
                kind: "sphere",
                op: BoolOp::Subtract,
                convex: true,
                has_children: false,
                mesh: &m,
            },
        ];
        let mut out = MeshData::new();
        let e = NoBooleans.resolve(&host(&m), &tools, &mut out).unwrap_err();
        assert_eq!(e.code, "boolean.unsupported");
        assert_eq!(e.path, "");
        assert_eq!(
            e.message,
            "no boolean resolver installed: intersect by node 'hole' on host 'body'"
        );
    }

    #[test]
    fn no_booleans_copies_host_without_tools() {
        let m = tri();
        let mut out = MeshData {
            positions: vec![[9.0; 3]],
            normals: vec![[0.0, 1.0, 0.0]],
            indices: vec![0, 0, 0],
        };
        NoBooleans.resolve(&host(&m), &[], &mut out).unwrap();
        assert_eq!(out, m);
    }
}
