//! Build step of prims v0: provider resolution, triangle estimate and budget, mesh
//! generation with booleans, materials, anchors, colliders, and the `PrimsLoader`.
//! Deterministic: the same asset and context give the same output.

use raa_geom::{GeometryProvider, MeshData, ParamValue, Params, Quality};
use raa_material::Material;
use raa_math::Mat4;

use super::model::{NodeOp, PrimNode, PrimsAsset};
use super::{DEFAULT_TRIANGLE_BUDGET, FORMAT, parse};
use crate::boolean::{BoolOp, BooleanHost, BooleanTool};
use crate::collider::{collider_from_doc, derive_collider};
use crate::error::LoadError;
use crate::json::JsonValue;
use crate::registry::AssetLoader;
use crate::{DEFAULT_MATERIAL, LoadContext, LoadedAsset, LoadedMesh};

/// Child pointer `<base>/<key>` with `~` and `/` escaped (RFC 6901).
fn ptr(base: &str, key: &str) -> String {
    let mut s = String::with_capacity(base.len() + key.len() + 1);
    s.push_str(base);
    s.push('/');
    for c in key.chars() {
        match c {
            '~' => s.push_str("~0"),
            '/' => s.push_str("~1"),
            _ => s.push(c),
        }
    }
    s
}

/// A node that becomes one mesh (a root or a union child), with its boolean tools.
struct Item<'a> {
    node: &'a PrimNode,
    pointer: String,
    world: Mat4,
    material: String,
    /// Subtract/intersect children and their pointers, in document order.
    tools: Vec<(&'a PrimNode, String)>,
}

/// Walk state: mesh items in pre-order and the world matrix of every node.
#[derive(Default)]
struct Plan<'a> {
    items: Vec<Item<'a>>,
    worlds: Vec<(&'a str, Mat4)>,
}

fn local_matrix(node: &PrimNode) -> Mat4 {
    node.transform
        .as_ref()
        .map_or(Mat4::IDENTITY, |t| t.to_mat4())
}

fn plan_node<'a>(
    node: &'a PrimNode,
    pointer: String,
    parent_world: &Mat4,
    parent_material: &str,
    plan: &mut Plan<'a>,
) {
    let world = parent_world.mul(&local_matrix(node));
    let material = node
        .material
        .clone()
        .unwrap_or_else(|| parent_material.to_string());
    plan.worlds.push((node.id.as_str(), world));
    let children = ptr(&pointer, "children");
    let mut tools = Vec::new();
    for (i, c) in node.children.iter().enumerate() {
        if c.op != NodeOp::Union {
            tools.push((c, format!("{children}/{i}")));
        }
    }
    plan.items.push(Item {
        node,
        pointer,
        world,
        material: material.clone(),
        tools,
    });
    for (i, c) in node.children.iter().enumerate() {
        let cp = format!("{children}/{i}");
        if c.op == NodeOp::Union {
            plan_node(c, cp, &world, &material, plan);
        } else {
            // A tool adds no mesh; its world is still known for collider overrides.
            plan.worlds
                .push((c.id.as_str(), world.mul(&local_matrix(c))));
        }
    }
}

fn plan(a: &PrimsAsset) -> Plan<'_> {
    let mut p = Plan::default();
    for (i, n) in a.nodes.iter().enumerate() {
        plan_node(
            n,
            format!("/nodes/{i}"),
            &Mat4::IDENTITY,
            DEFAULT_MATERIAL,
            &mut p,
        );
    }
    p
}

fn provider<'c>(
    ctx: &LoadContext<'c>,
    node: &PrimNode,
    pointer: &str,
) -> Result<&'c dyn GeometryProvider, LoadError> {
    ctx.geometry.get(&node.kind).ok_or_else(|| {
        LoadError::new(
            "kind.unknown",
            ptr(pointer, "kind"),
            format!("kind '{}' is not registered", node.kind),
        )
    })
}

fn params(node: &PrimNode, pointer: &str) -> Result<Params, LoadError> {
    let mut out = Params::new();
    for (k, v) in &node.params {
        let value = match v {
            JsonValue::Number(x) => Some(ParamValue::Scalar(*x as f32)),
            JsonValue::Array(a) => match a.as_slice() {
                [
                    JsonValue::Number(x),
                    JsonValue::Number(y),
                    JsonValue::Number(z),
                ] => Some(ParamValue::Vec3([*x as f32, *y as f32, *z as f32])),
                _ => None,
            },
            _ => None,
        };
        let value = value.ok_or_else(|| {
            LoadError::new(
                "kind.params",
                ptr(&ptr(pointer, "params"), k),
                format!("param '{k}' must be a number or an array of 3 numbers"),
            )
        })?;
        out.set(k, value);
    }
    Ok(out)
}

/// Provider and converted params of a node.
fn resolve<'c>(
    ctx: &LoadContext<'c>,
    node: &PrimNode,
    pointer: &str,
) -> Result<(&'c dyn GeometryProvider, Params), LoadError> {
    let p = provider(ctx, node, pointer)?;
    Ok((p, params(node, pointer)?))
}

fn generate(
    p: &dyn GeometryProvider,
    params: &Params,
    quality: Quality,
    pointer: &str,
    out: &mut MeshData,
) -> Result<(), LoadError> {
    p.generate(params, quality, out)
        .map_err(|e| LoadError::from_geom(&e, ptr(pointer, "params")))
}

fn saturate(n: u64) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

fn estimate_plan(plan: &Plan<'_>, ctx: &LoadContext<'_>) -> Result<u32, LoadError> {
    let mut total: u64 = 0;
    for item in &plan.items {
        let (p, params) = resolve(ctx, item.node, &item.pointer)?;
        let n = p
            .triangle_count(&params, ctx.quality)
            .map_err(|e| LoadError::from_geom(&e, ptr(&item.pointer, "params")))?;
        total += u64::from(n);
    }
    Ok(saturate(total))
}

/// Triangles of the meshes the asset produces at `ctx.quality` (roots and union children;
/// boolean tools excluded), before booleans, from the providers' `triangle_count`.
pub fn estimate_triangles(a: &PrimsAsset, ctx: &LoadContext<'_>) -> Result<u32, LoadError> {
    estimate_plan(&plan(a), ctx)
}

fn budget_error(path: &str, n: u32, budget: u32) -> LoadError {
    LoadError::new(
        "cap.budget",
        path,
        format!("{n} triangles > budget {budget}"),
    )
}

/// Builds the meshes, materials, anchors and colliders of a parsed asset at `ctx.quality`.
pub fn build(a: &PrimsAsset, ctx: &LoadContext<'_>) -> Result<LoadedAsset, LoadError> {
    let budget = a.triangle_budget.unwrap_or(DEFAULT_TRIANGLE_BUDGET);
    let budget_path = if a.triangle_budget.is_some() {
        "/triangleBudget"
    } else {
        ""
    };
    let plan = plan(a);
    let estimate = estimate_plan(&plan, ctx)?;
    if estimate > budget {
        return Err(budget_error(budget_path, estimate, budget));
    }

    let mut meshes = Vec::with_capacity(plan.items.len());
    let mut total: u64 = 0;
    let mut scratch = MeshData::new();
    for item in &plan.items {
        let (p, host_params) = resolve(ctx, item.node, &item.pointer)?;
        let mut mesh = MeshData::new();
        generate(p, &host_params, ctx.quality, &item.pointer, &mut mesh)?;
        if !item.tools.is_empty() {
            let mut tool_meshes = Vec::with_capacity(item.tools.len());
            let mut tool_info = Vec::with_capacity(item.tools.len());
            for (tool, tool_ptr) in &item.tools {
                let (tp, tparams) = resolve(ctx, tool, tool_ptr)?;
                generate(tp, &tparams, ctx.quality, tool_ptr, &mut scratch)?;
                let mut in_host = MeshData::new();
                in_host.append_transformed(&scratch, &local_matrix(tool));
                tool_meshes.push(in_host);
                let op = if tool.op == NodeOp::Intersect {
                    BoolOp::Intersect
                } else {
                    BoolOp::Subtract
                };
                tool_info.push((tool, op, tp.is_convex()));
            }
            let tools: Vec<BooleanTool<'_>> = tool_info
                .iter()
                .zip(&tool_meshes)
                .map(|((tool, op, convex), m)| BooleanTool {
                    node_id: &tool.id,
                    kind: &tool.kind,
                    op: *op,
                    convex: *convex,
                    has_children: !tool.children.is_empty(),
                    mesh: m,
                })
                .collect();
            let host = BooleanHost {
                node_id: &item.node.id,
                kind: &item.node.kind,
                convex: p.is_convex(),
                mesh: &mesh,
            };
            let mut out = MeshData::new();
            ctx.booleans
                .resolve(&host, &tools, &mut out)
                .map_err(|mut e| {
                    if e.path.is_empty()
                        && let Some((_, first)) = item.tools.first()
                    {
                        e.path = first.clone();
                    }
                    e
                })?;
            mesh = out;
        }
        total += mesh.triangle_count() as u64;
        meshes.push(LoadedMesh {
            node_id: item.node.id.clone(),
            kind: item.node.kind.clone(),
            mesh,
            transform: item.world,
            material: item.material.clone(),
        });
    }
    let triangles = saturate(total);
    if triangles > budget {
        return Err(budget_error(budget_path, triangles, budget));
    }

    let mut declared: Vec<_> = a.materials.iter().collect();
    declared.sort_by(|x, y| x.0.as_bytes().cmp(y.0.as_bytes()));
    let mut materials = Vec::with_capacity(declared.len() + 1);
    materials.push((DEFAULT_MATERIAL.to_string(), Material::default()));
    for (name, m) in declared {
        let f = |v: [f64; 3]| v.map(|x| x as f32);
        let mat = Material::from_srgb(
            f(m.color),
            m.roughness as f32,
            m.metallic as f32,
            f(m.emissive),
        );
        mat.validate()
            .map_err(|e| LoadError::from_material(&e, ptr("/materials", name)))?;
        materials.push((name.clone(), mat));
    }

    let mut anchors: Vec<(String, Mat4)> = a
        .anchors
        .iter()
        .map(|(k, t)| (k.clone(), t.to_mat4()))
        .collect();
    anchors.sort_by(|x, y| x.0.as_bytes().cmp(y.0.as_bytes()));

    let mut colliders = Vec::new();
    if let Some(docs) = &a.collision {
        let world_of = |id: &str| plan.worlds.iter().find(|(n, _)| *n == id).map(|(_, w)| *w);
        for (i, doc) in docs.iter().enumerate() {
            let pointer = format!("/collision/colliders/{i}");
            colliders.push(collider_from_doc(doc, &pointer, world_of)?);
        }
    } else {
        for item in &plan.items {
            let low = || {
                let (p, params) = resolve(ctx, item.node, &item.pointer)?;
                let mut m = MeshData::new();
                generate(p, &params, Quality::LOW, &item.pointer, &mut m)?;
                Ok(m)
            };
            colliders.push(derive_collider(
                &item.node.id,
                &item.node.kind,
                &item.node.params,
                &item.world,
                &item.pointer,
                low,
            )?);
        }
    }

    Ok(LoadedAsset {
        format: FORMAT.to_string(),
        id: a.id.clone(),
        name: a.name.clone(),
        meshes,
        materials,
        anchors,
        colliders,
        triangles,
        budget,
    })
}

/// Loader of `raa.prims` documents (parse, then build).
#[derive(Debug, Clone, Copy, Default)]
pub struct PrimsLoader;

/// `json.syntax` error at the first invalid UTF-8 byte (`@<offset> line:col`).
fn utf8_error(bytes: &[u8], off: usize) -> LoadError {
    let head = bytes.get(..off).unwrap_or(&[]);
    let line = 1 + head.iter().filter(|&&c| c == b'\n').count();
    let line_start = head.iter().rposition(|&c| c == b'\n').map_or(0, |i| i + 1);
    let col = 1 + head
        .get(line_start..)
        .unwrap_or(&[])
        .iter()
        .filter(|&&c| (c & 0xC0) != 0x80)
        .count();
    LoadError::new(
        "json.syntax",
        format!("@{off} {line}:{col}"),
        "invalid UTF-8",
    )
}

impl AssetLoader for PrimsLoader {
    fn format(&self) -> &'static str {
        FORMAT
    }

    fn sniff(&self, bytes: &[u8]) -> bool {
        let head = bytes.get(..bytes.len().min(512)).unwrap_or(&[]);
        head.windows(FORMAT.len()).any(|w| w == FORMAT.as_bytes())
    }

    fn load(&self, bytes: &[u8], ctx: &LoadContext<'_>) -> Result<LoadedAsset, LoadError> {
        let text = std::str::from_utf8(bytes).map_err(|e| utf8_error(bytes, e.valid_up_to()))?;
        let asset = parse::parse_with(text, ctx.strict_kinds)?;
        build(&asset, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NoBooleans;
    use crate::prims::samples::{invalid_cases, sample_assets};
    use raa_geom::GeometryRegistry;
    use std::cell::RefCell;

    fn registry() -> GeometryRegistry {
        let mut g = GeometryRegistry::new();
        raa_prims::register_all(&mut g).expect("register_all");
        g
    }

    fn text(name: &str) -> String {
        sample_assets()
            .into_iter()
            .find(|(n, _)| *n == name)
            .map(|(_, t)| t)
            .expect("sample")
    }

    fn count_union(nodes: &[PrimNode]) -> usize {
        nodes
            .iter()
            .filter(|n| n.op == NodeOp::Union)
            .map(|n| 1 + count_union(&n.children))
            .sum()
    }

    #[test]
    fn crate_sample_builds_at_lod0() {
        let g = registry();
        let ctx = LoadContext {
            geometry: &g,
            booleans: &NoBooleans,
            quality: Quality::LOD0,
            strict_kinds: true,
        };
        let a = super::parse::parse_with(&text("crate"), true).expect("parse");
        let built = build(&a, &ctx).expect("build");
        assert_eq!(built.format, FORMAT);
        assert_eq!(built.meshes.len(), count_union(&a.nodes));
        assert!(built.triangles <= built.budget);
        assert_eq!(
            built.triangles,
            estimate_triangles(&a, &ctx).expect("estimate")
        );
        assert_eq!(built.materials[0].0, DEFAULT_MATERIAL);
        assert_eq!(built.materials[0].1, Material::default());
        assert!(!built.colliders.is_empty());
        for m in &built.meshes {
            assert!(m.mesh.validate().is_ok(), "mesh {}", m.node_id);
        }
        assert_eq!(build(&a, &ctx).expect("build again"), built);
    }

    #[test]
    fn every_sample_builds_and_build_time_cases_refuse() {
        let g = registry();
        let ctx = LoadContext {
            geometry: &g,
            booleans: &NoBooleans,
            quality: Quality::LOD0,
            strict_kinds: true,
        };
        for (name, t) in sample_assets() {
            let built = PrimsLoader.load(t.as_bytes(), &ctx);
            assert!(built.is_ok(), "{name}: {built:?}");
        }
        let build_time = [
            "cap.budget",
            "collision.torus_needs_override",
            "boolean.unsupported",
        ];
        for (label, t, code) in invalid_cases() {
            if build_time.contains(&code) {
                let e = PrimsLoader.load(t.as_bytes(), &ctx).expect_err(label);
                assert_eq!(e.code, code, "{label}: {e}");
                assert!(!e.message.is_empty(), "{label}");
                if code == "cap.budget" {
                    assert_eq!(e.path, "/triangleBudget");
                    assert!(e.message.contains("> budget 10"), "{e}");
                }
                if code == "boolean.unsupported" {
                    assert!(!e.path.is_empty(), "{e}");
                }
            }
        }
    }

    #[test]
    fn unregistered_kind_and_loader_basics() {
        let g = GeometryRegistry::new();
        let ctx = LoadContext {
            geometry: &g,
            booleans: &NoBooleans,
            quality: Quality::LOD0,
            strict_kinds: false,
        };
        let t = text("crate");
        let e = PrimsLoader
            .load(t.as_bytes(), &ctx)
            .expect_err("no providers");
        assert_eq!(e.code, "kind.unknown");
        assert_eq!(e.path, "/nodes/0/kind");
        assert!(PrimsLoader.sniff(t.as_bytes()));
        assert!(!PrimsLoader.sniff(b"{\"format\": \"other\"}"));
        assert_eq!(PrimsLoader.format(), FORMAT);
        let e = PrimsLoader
            .load(b"{\n \"a\xff\": 1}", &ctx)
            .expect_err("bad utf-8");
        assert_eq!(e.code, "json.syntax");
        assert_eq!(e.path, "@5 2:4");
    }

    /// Copies the host and records `(node_id, has_children)` of every tool it receives.
    #[derive(Default)]
    struct Recording {
        tools: RefCell<Vec<(String, bool)>>,
    }

    impl crate::BooleanResolver for Recording {
        fn resolve(
            &self,
            host: &BooleanHost<'_>,
            tools: &[BooleanTool<'_>],
            out: &mut MeshData,
        ) -> Result<(), LoadError> {
            for t in tools {
                let rec = (t.node_id.to_string(), t.has_children);
                self.tools.borrow_mut().push(rec);
            }
            out.clear();
            out.append_transformed(host.mesh, &Mat4::IDENTITY);
            Ok(())
        }
    }

    #[test]
    fn tool_with_children_reaches_resolver_with_has_children() {
        let t = format!(
            concat!(
                r#"{{"format":"{}","version":0,"id":"cut","name":"Cut","#,
                r#""nodes":[{{"id":"host","kind":"box","params":{{"halfExtents":[0.5,0.5,0.5]}},"#,
                r#""children":[{{"id":"hole","kind":"box","params":{{"halfExtents":[0.2,0.2,0.6]}},"#,
                r#""op":"subtract","children":[{{"id":"inner","kind":"sphere","#,
                r#""params":{{"radius":0.1}}}}]}}]}}],"materials":{{}},"anchors":{{}}}}"#
            ),
            FORMAT
        );
        let g = registry();
        let rec = Recording::default();
        let ctx = LoadContext {
            geometry: &g,
            booleans: &rec,
            quality: Quality::LOD0,
            strict_kinds: true,
        };
        let built = PrimsLoader
            .load(t.as_bytes(), &ctx)
            .expect("recording load");
        assert_eq!(*rec.tools.borrow(), vec![("hole".to_string(), true)]);
        // The tool's union child adds no mesh and no collider.
        assert_eq!(built.meshes.len(), 1);
        assert_eq!(built.colliders.len(), 1);

        let ctx = LoadContext {
            booleans: &NoBooleans,
            ..ctx
        };
        let e = PrimsLoader
            .load(t.as_bytes(), &ctx)
            .expect_err("NoBooleans");
        assert_eq!(e.code, "boolean.unsupported");
        assert_eq!(e.path, "/nodes/0/children/0");
    }
}
