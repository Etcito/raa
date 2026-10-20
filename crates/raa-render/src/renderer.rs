//! The forward renderer: owns the shared [`ForwardState`], the CPU frame preparation and
//! the material library, uploads meshes and installs the [`ForwardPass`] in a frame graph.

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::{FrameGraph, GpuContext, MeshId, RaaError, ResourceId, TargetDesc, TargetSize};
use raa_geom::{GeomError, MeshData};
use raa_material::{Material, MaterialLibrary};
use raa_scene::{RenderList, Scene};

use crate::forward::{ForwardPass, ForwardState, GpuMesh};
use crate::prep::FramePrep;
use crate::{FrameCounts, RendererOptions};

/// Bytes per S1 MaterialGpu.
const MATERIAL_GPU_SIZE: usize = 48;

/// Forward renderer drawing a [`Scene`] through a [`ForwardPass`].
///
/// The material library is seeded with one default [`Material`] (linear grey 0.5,
/// roughness 0.6, PbrLite), so `MaterialId(0)` is always valid.
pub struct Renderer {
    state: Rc<RefCell<ForwardState>>,
    prep: FramePrep,
    materials: MaterialLibrary,
    material_bytes: Vec<u8>,
    opts: RendererOptions,
    surface_is_srgb: bool,
}

impl Renderer {
    /// Creates the pipeline for `surface_format`, the uniform (144 B), instance
    /// (max_instances x 128 B) and material (max_materials x 48 B) buffers, and the
    /// CPU frame buffers. `surface_is_srgb` false makes the shader encode to sRGB.
    pub fn new(
        gpu: &GpuContext,
        surface_format: wgpu::TextureFormat,
        surface_is_srgb: bool,
        opts: RendererOptions,
    ) -> Result<Renderer, RaaError> {
        let state = ForwardState::new(gpu, surface_format, opts)?;
        let mut materials = MaterialLibrary::new();
        // MaterialId(0) is the default material, always present.
        materials
            .add(Material::default())
            .map_err(|e| RaaError::Resource(format!("render.default_material: {e}")))?;
        Ok(Renderer {
            state: Rc::new(RefCell::new(state)),
            prep: FramePrep::new(opts.max_instances),
            materials,
            material_bytes: Vec::with_capacity(opts.max_materials as usize * MATERIAL_GPU_SIZE),
            opts,
            surface_is_srgb,
        })
    }

    /// Creates the `depth` target (Depth32Float, surface size) in `graph`, appends the
    /// `forward` pass sharing this renderer's state and returns the depth id.
    pub fn install(&self, graph: &mut FrameGraph) -> ResourceId {
        let depth = graph.create_target(TargetDesc {
            label: "depth",
            format: wgpu::TextureFormat::Depth32Float,
            size: TargetSize::Surface,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        });
        graph.add_pass(Box::new(ForwardPass::new(Rc::clone(&self.state), depth)));
        depth
    }

    /// Validates `mesh` and uploads its S1 vertex and u32 index bytes. Ids are dense and
    /// removed slots are reused (lowest first). An invalid mesh gives
    /// [`RaaError::Resource`] starting with `render.invalid_mesh`.
    pub fn upload_mesh(&mut self, gpu: &GpuContext, mesh: &MeshData) -> Result<MeshId, RaaError> {
        mesh.validate().map_err(invalid_mesh)?;
        let index_count = u32::try_from(mesh.indices.len()).map_err(|_| {
            RaaError::Resource(format!(
                "render.invalid_mesh: {} indices exceed u32",
                mesh.indices.len()
            ))
        })?;
        let mut bytes = Vec::new();
        mesh.write_vertex_bytes(&mut bytes);
        let vertex = buffer_with(gpu, "raa.mesh.vertices", wgpu::BufferUsages::VERTEX, &bytes);
        mesh.write_index_bytes(&mut bytes);
        let index = buffer_with(gpu, "raa.mesh.indices", wgpu::BufferUsages::INDEX, &bytes);
        let gpu_mesh = GpuMesh {
            vertex,
            index,
            index_count,
        };

        let mut state = self.state.borrow_mut();
        let slot = free_slot(&state.meshes);
        let id = u32::try_from(slot)
            .map_err(|_| RaaError::Resource("render.too_many_meshes: u32 ids".to_string()))?;
        match state.meshes.get_mut(slot) {
            Some(entry) => *entry = Some(gpu_mesh),
            None => state.meshes.push(Some(gpu_mesh)),
        }
        Ok(MeshId(id))
    }

    /// Drops the GPU buffers of `id`; its slot is reused by the next upload. Unknown ids
    /// are ignored.
    pub fn remove_mesh(&mut self, id: MeshId) {
        if let Some(entry) = self.state.borrow_mut().meshes.get_mut(id.0 as usize) {
            *entry = None;
        }
    }

    /// Triangles of an uploaded mesh (index count / 3), `None` when unknown.
    pub fn mesh_triangles(&self, id: MeshId) -> Option<u32> {
        let state = self.state.borrow();
        state
            .meshes
            .get(id.0 as usize)
            .and_then(Option::as_ref)
            .map(|m| m.index_count / 3)
    }

    /// The material library (`MaterialId(0)` is the default material).
    pub fn materials(&self) -> &MaterialLibrary {
        &self.materials
    }

    /// The material library; changes are uploaded at the next [`prepare`](Self::prepare).
    pub fn materials_mut(&mut self) -> &mut MaterialLibrary {
        &mut self.materials
    }

    /// Updates world transforms, builds `list`, writes the instance, uniform and (when
    /// dirty) material bytes to the GPU and sets the draws of the forward pass.
    ///
    /// Refusals (all [`RaaError::Resource`], nothing written to the GPU):
    /// `render.too_many_instances`, `render.unknown_mesh`, `render.too_many_materials`.
    /// Allocates nothing in RAA code after warm-up.
    pub fn prepare(
        &mut self,
        gpu: &GpuContext,
        scene: &mut Scene,
        list: &mut RenderList,
    ) -> Result<FrameCounts, RaaError> {
        scene.update_world();
        scene.build_render_list(list);
        let mut counts = self.prep.prepare_cpu(scene, list, self.surface_is_srgb)?;

        let mut state = self.state.borrow_mut();
        let mut triangles = 0u64;
        for draw in self.prep.draws() {
            let mesh = state
                .meshes
                .get(draw.mesh.0 as usize)
                .and_then(Option::as_ref)
                .ok_or_else(|| unknown_mesh(draw.mesh))?;
            triangles += u64::from(mesh.index_count / 3) * u64::from(draw.instance_count);
        }
        counts.triangles = triangles;

        let material_count = self.materials.len();
        if material_count > self.opts.max_materials as usize {
            return Err(RaaError::Resource(format!(
                "render.too_many_materials: {material_count} materials exceed max_materials {}",
                self.opts.max_materials
            )));
        }

        let queue = gpu.queue();
        let instances = self.prep.instance_bytes();
        if !instances.is_empty() {
            queue.write_buffer(&state.instance_buffer, 0, instances);
        }
        queue.write_buffer(&state.uniform_buffer, 0, self.prep.frame_uniform_bytes());
        if self.materials.is_dirty() {
            self.materials.write_gpu_all(&mut self.material_bytes);
            if !self.material_bytes.is_empty() {
                queue.write_buffer(&state.material_buffer, 0, &self.material_bytes);
            }
        }

        state.draws.clear();
        state.draws.extend_from_slice(self.prep.draws());
        Ok(counts)
    }
}

/// Creates a buffer of `bytes.len()` with `usage | COPY_DST` and writes `bytes` into it.
fn buffer_with(
    gpu: &GpuContext,
    label: &'static str,
    usage: wgpu::BufferUsages,
    bytes: &[u8],
) -> wgpu::Buffer {
    let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !bytes.is_empty() {
        gpu.queue().write_buffer(&buffer, 0, bytes);
    }
    buffer
}

/// First free slot of a mesh table (a removed entry, else the end).
fn free_slot<T>(table: &[Option<T>]) -> usize {
    table
        .iter()
        .position(Option::is_none)
        .unwrap_or(table.len())
}

/// Maps a mesh validation failure to `render.invalid_mesh` plus the reason.
fn invalid_mesh(err: GeomError) -> RaaError {
    match err {
        GeomError::InvalidMesh(reason) => {
            RaaError::Resource(format!("render.invalid_mesh: {reason}"))
        }
        other => RaaError::Resource(format!("render.invalid_mesh: {other}")),
    }
}

/// `render.unknown_mesh` for a draw of a mesh that is not uploaded.
fn unknown_mesh(id: MeshId) -> RaaError {
    RaaError::Resource(format!(
        "render.unknown_mesh: mesh {} is not uploaded",
        id.0
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_slot_reuses_the_first_removed_entry() {
        assert_eq!(free_slot::<u8>(&[]), 0);
        assert_eq!(free_slot(&[Some(1), Some(2)]), 2);
        assert_eq!(free_slot(&[Some(1), None, None]), 1);
    }

    #[test]
    fn invalid_mesh_error_carries_the_code_and_reason() {
        let mut mesh = MeshData::new();
        mesh.positions.push([0.0; 3]);
        let err = invalid_mesh(mesh.validate().expect_err("mismatch must be refused"));
        assert_eq!(err.code(), "core.resource");
        let text = err.to_string();
        assert!(text.contains("render.invalid_mesh"), "{text}");
        assert!(!text.contains("geom.invalid_mesh: geom"), "{text}");
    }

    #[test]
    fn unknown_mesh_error_carries_the_code() {
        let err = unknown_mesh(MeshId(3));
        assert_eq!(err.code(), "core.resource");
        assert!(err.to_string().contains("render.unknown_mesh"));
    }
}
