//! M3 primitives contact sheet: 8 kinds (columns) x LOW, LOD0, HIGH (rows), reference set A.
#[cfg(feature = "alloc-count")]
#[global_allocator]
static A: raa_alloc_count::CountingAlloc = raa_alloc_count::CountingAlloc;

#[cfg(any(target_arch = "wasm32", test))]
#[cfg_attr(
    not(target_arch = "wasm32"),
    allow(dead_code, reason = "host builds only run the CPU unit tests")
)]
mod app {
    use raa_core::{FrameGraph, GpuContext, RaaError};
    use raa_geom::{GeomError, MeshData, Quality};
    use raa_material::Material;
    use raa_math::{Quat, Vec3};
    use raa_prims::{PrimKind, PrimParams};
    use raa_render::{Renderer, RendererOptions};
    use raa_scene::{Camera, NodeId, RenderList, Scene, Transform};
    use raa_web::{FrameCounters, FrameInput, WebApp};

    pub(crate) const SPACING: f32 = 2.5;
    pub(crate) const QUALITIES: [Quality; 3] = [Quality::LOW, Quality::LOD0, Quality::HIGH];
    pub(crate) const CELLS: usize = 24;
    /// Rotation speed about Y, radians per millisecond.
    const SPIN: f64 = 0.000_35;

    pub(crate) struct Cell {
        pub(crate) kind: PrimKind,
        pub(crate) quality: Quality,
        pub(crate) center: Vec3,
    }

    /// The 24 cells, kind-major (ALL order), then LOW, LOD0, HIGH. Columns run left to
    /// right, LOW is the top row.
    pub(crate) fn grid() -> [Cell; CELLS] {
        std::array::from_fn(|i| {
            let (col, row) = (i / 3, i % 3);
            Cell {
                kind: PrimKind::ALL[col],
                quality: QUALITIES[row],
                center: Vec3::new(
                    (col as f32 - 3.5) * SPACING,
                    (1.0 - row as f32) * SPACING,
                    0.0,
                ),
            }
        })
    }

    pub(crate) fn camera(aspect: f32) -> Camera {
        Camera {
            eye: Vec3::new(0.0, 0.0, 18.0),
            target: Vec3::ZERO,
            fov_y_rad: 40.0_f32.to_radians(),
            aspect,
            ..Camera::default()
        }
    }

    pub(crate) fn materials() -> [Material; 8] {
        const C: [[f32; 3]; 8] = [
            [0.80, 0.18, 0.12],
            [0.15, 0.65, 0.22],
            [0.12, 0.30, 0.85],
            [0.85, 0.70, 0.15],
            [0.90, 0.35, 0.10],
            [0.60, 0.20, 0.75],
            [0.10, 0.70, 0.70],
            [0.80, 0.80, 0.82],
        ];
        C.map(|c| Material::pbr(c, 0.45, 0.0))
    }

    fn err(code: &str, e: impl std::fmt::Display) -> RaaError {
        RaaError::Resource(format!("{code}: {e}"))
    }

    struct Running {
        renderer: Renderer,
        scene: Scene,
        list: RenderList,
        nodes: Vec<NodeId>,
        triangles: u64,
    }

    #[derive(Default)]
    pub(crate) struct ContactSheet(Option<Running>);

    impl WebApp for ContactSheet {
        fn init(
            &mut self,
            gpu: &GpuContext,
            graph: &mut FrameGraph,
            size: (u32, u32),
            surface_format: wgpu::TextureFormat,
            surface_is_srgb: bool,
        ) -> Result<(), RaaError> {
            let mut renderer = Renderer::new(
                gpu,
                surface_format,
                surface_is_srgb,
                RendererOptions::default(),
            )?;
            let mut ids = Vec::with_capacity(8);
            for m in materials() {
                ids.push(
                    renderer
                        .materials_mut()
                        .add(m)
                        .map_err(|e| err(e.code, &e.message))?,
                );
            }
            let mut scene = Scene::with_capacity(CELLS, CELLS);
            let mut nodes = Vec::with_capacity(CELLS);
            let mut mesh = MeshData::new();
            let mut triangles = 0u64;
            for (i, cell) in grid().iter().enumerate() {
                let params = PrimParams::reference_set(cell.kind, 'A');
                raa_prims::generate(&params, cell.quality, &mut mesh)
                    .map_err(|e: GeomError| err(e.code(), e))?;
                let id = renderer.upload_mesh(gpu, &mesh)?;
                triangles += renderer.mesh_triangles(id).map_or(0, u64::from);
                let t = Transform::from_trs(cell.center, Quat::IDENTITY, Vec3::ONE);
                let node = scene.add_node(None, t).map_err(|e| err(e.code(), e))?;
                scene
                    .add_instance(node, id, ids[i / 3])
                    .map_err(|e| err(e.code(), e))?;
                nodes.push(node);
            }
            renderer.install(graph);
            scene.camera = camera(size.0 as f32 / size.1.max(1) as f32);
            self.0 = Some(Running {
                renderer,
                scene,
                list: RenderList::with_capacity(CELLS),
                nodes,
                triangles,
            });
            Ok(())
        }

        fn update(
            &mut self,
            gpu: &GpuContext,
            input: &FrameInput,
        ) -> Result<FrameCounters, RaaError> {
            let app = (self.0.as_mut())
                .ok_or_else(|| err("demo.not_initialised", "update before init"))?;
            let rot = Quat::from_rotation_y((input.now_ms * SPIN) as f32);
            for (cell, &node) in grid().iter().zip(&app.nodes) {
                let t = Transform::from_trs(cell.center, rot, Vec3::ONE);
                app.scene.set_local(node, t).map_err(|e| err(e.code(), e))?;
            }
            let counts = app.renderer.prepare(gpu, &mut app.scene, &mut app.list)?;
            Ok(FrameCounters {
                instances: CELLS as u32,
                draw_calls: counts.draw_calls,
                triangles: app.triangles,
            })
        }

        fn resize(&mut self, _gpu: &GpuContext, size: (u32, u32)) {
            if let Some(app) = self.0.as_mut() {
                app.scene.camera.set_viewport(size.0, size.1);
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn grid_has_24_cells_in_all_by_quality_order_with_2_5_m_spacing() {
            let g = grid();
            for (i, c) in g.iter().enumerate() {
                assert_eq!(c.kind, PrimKind::ALL[i / 3]);
                assert_eq!(c.quality, QUALITIES[i % 3]);
            }
            assert_eq!(g[3].center.x - g[0].center.x, SPACING);
            assert_eq!(g[0].center.y - g[1].center.y, SPACING);
            assert_eq!(g[1].center.y - g[2].center.y, SPACING);
            assert_eq!(g[0].center.x, -g[21].center.x);
        }

        #[test]
        fn bounding_spheres_project_inside_ndc() {
            let vp = camera(16.0 / 9.0).view_proj();
            for c in grid() {
                let b = PrimParams::reference_set(c.kind, 'A').bounds();
                let r = b.diagonal() * 0.5;
                for (dx, dy) in [(-r, -r), (r, -r), (-r, r), (r, r)] {
                    let p = vp.transform_point(Vec3::new(c.center.x + dx, c.center.y + dy, r));
                    assert!(
                        p.x.abs() <= 0.95 && p.y.abs() <= 0.95,
                        "{:?}: {p:?}",
                        c.kind
                    );
                }
            }
        }

        #[test]
        fn materials_validate_and_differ() {
            let m = materials();
            for a in &m {
                a.validate().expect("valid");
            }
            for (i, a) in m.iter().enumerate() {
                assert!(m[i + 1..].iter().all(|b| a.base_color != b.base_color));
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod web;

#[cfg(target_arch = "wasm32")]
pub use web::{Demo, start};
