//! A single planar receiver with a fresh, depth-tested capture of the live scene.

use raa_core::{GpuContext, MeshId, RaaError};
use raa_math::{Mat4, Vec4};
use raa_pipeline::HDR_FORMAT;
use raa_scene::{Camera, Scene};

use crate::frame::{FRAME_GPU_SIZE, FrameParams, write_camera_frame_gpu};
use crate::handle::{LitHandle, LitShared, STREAM0_STRIDE};
use crate::material::DrawClass;
use crate::renderer::LitOptions;
use crate::select::{IdPrep, SelectInput};
use crate::skin_views::SkinDraw;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
// Matches RAA_PLANAR_CAPTURE_FEATURE in raa-shader's WGSL prelude.
pub(crate) const CAPTURE_FEATURE: u32 = 40;
const PLANE_BIAS: f32 = 0.0005;
const COMPOSITE_BYTES: u64 = 160;
const POSITION: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x3];

/// One opaque receiver and an oriented world plane (normal toward the reflecting side).
#[derive(Clone, Copy, Debug)]
pub struct PlanarReflection {
    pub(crate) instance: u32,
    plane: Vec4,
}

impl PlanarReflection {
    /// Accepts `[nx, ny, nz, d]` for `dot(normal, world_position) + d = 0`.
    pub fn new(instance: u32, plane: [f32; 4]) -> Result<Self, RaaError> {
        let [x, y, z, _] = plane;
        let length = x.hypot(y).hypot(z);
        let plane = plane.map(|value| value / length);
        if !plane.iter().all(|value| value.is_finite()) || !length.is_finite() {
            return Err(RaaError::FrameGraph {
                code: "api.invalid_argument",
                message: "the reflection plane needs a finite offset and a finite, nonzero normal"
                    .into(),
            });
        }
        let [x, y, z, d] = plane;
        Ok(Self {
            instance,
            plane: Vec4::new(x, y, z, d),
        })
    }

    fn reflected_camera(self, camera: Camera) -> Option<Camera> {
        let normal = self.plane.xyz();
        let distance = |p| normal.dot(p) + self.plane.w;
        if distance(camera.eye) <= PLANE_BIAS {
            return None;
        }
        let reflect = |p| p - normal * (2.0 * distance(p));
        Some(Camera {
            eye: reflect(camera.eye),
            target: reflect(camera.target),
            up: camera.up - normal * (2.0 * normal.dot(camera.up)),
            ..camera
        })
    }

    // Replace the [0, 1] projection's near plane with the mirror plane. Shifting
    // slightly into the reflected scene also clips coplanar backing geometry.
    fn clipped_projection(self, reflected: Camera) -> Option<Mat4> {
        let inv_view = reflected.view().inverse()?;
        let plane = Vec4::from_vec3(self.plane.xyz(), self.plane.w - PLANE_BIAS);
        let eye_plane = inv_view.transpose().mul_vec4(plane);
        let mut proj = reflected.projection();
        let corner = Vec4::new(eye_plane.x.signum(), eye_plane.y.signum(), 1.0, 1.0);
        let scale = eye_plane.dot(proj.inverse()?.mul_vec4(corner));
        if scale <= 1e-6 || !scale.is_finite() {
            return None;
        }
        for (column, value) in proj.cols.iter_mut().zip(eye_plane.to_array()) {
            column[2] = value / scale;
        }
        Some(proj)
    }
}

struct Receiver {
    reflection: PlanarReflection,
    world: Mat4,
    camera: Camera,
    projection: Mat4,
}

pub(crate) struct PlanarView {
    pub(crate) select: IdPrep,
    pub(crate) ids: wgpu::Buffer,
    frame: wgpu::Buffer,
    group: Option<(wgpu::BindGroup, wgpu::BindGroup)>,
    receiver: Option<Receiver>,
    meshes: Vec<MeshId>,
    view_proj: [u8; 64],
}

impl PlanarView {
    pub(crate) fn new(gpu: &GpuContext, opts: LitOptions) -> Self {
        Self {
            select: IdPrep::new(opts.max_instances, opts.max_materials),
            ids: buffer(
                gpu,
                "raa.lit.planar.ids",
                4 * u64::from(opts.max_instances.max(1)),
                wgpu::BufferUsages::VERTEX,
            ),
            frame: buffer(
                gpu,
                "raa.lit.planar.frame",
                FRAME_GPU_SIZE as u64,
                wgpu::BufferUsages::UNIFORM,
            ),
            group: None,
            receiver: None,
            meshes: Vec::new(),
            view_proj: [0; 64],
        }
    }

    pub(crate) fn select_scene(
        &mut self,
        main: &SelectInput<'_>,
        capture: &SelectInput<'_>,
        skinned: &dyn Fn(u32) -> Option<SkinDraw>,
        reflection: PlanarReflection,
        camera: Camera,
    ) -> Result<(), RaaError> {
        self.receiver = None;
        self.meshes.clear();
        let Some(camera) = reflection.reflected_camera(camera) else {
            return Ok(());
        };
        let Some(projection) = reflection.clipped_projection(camera) else {
            return Ok(());
        };
        let mut world = None;
        for (i, item) in main.list.items.iter().enumerate() {
            if main.tags[i] == reflection.instance
                && (main.selected)(i)
                && main
                    .materials
                    .get(item.material.0 as usize)
                    .is_some_and(|material| material.class == DrawClass::Opaque)
            {
                self.meshes
                    .push((main.level_mesh)(main.slots[i], item.mesh));
                world = Some(item.world);
            }
        }
        let Some(world) = world else {
            return Ok(());
        };
        let capture = SelectInput {
            view: camera.view(),
            ..*capture
        };
        self.select.prepare_camera(&capture, skinned)?;
        self.receiver = Some(Receiver {
            reflection,
            world,
            camera,
            projection,
        });
        Ok(())
    }

    pub(crate) fn upload(
        &mut self,
        gpu: &GpuContext,
        shared: &LitShared,
        scene: &Scene,
        main: FrameParams,
        main_frame: &[u8; FRAME_GPU_SIZE],
    ) -> Result<(), RaaError> {
        let Some(receiver) = &self.receiver else {
            return Ok(());
        };
        let mut params = FrameParams {
            viewport: capture_size(main.viewport),
            msaa: 1,
            // Reflecting the camera basis reverses its horizontal axis.
            camera_jitter_ndc: [-main.camera_jitter_ndc[0], main.camera_jitter_ndc[1]],
            active_lights: main.active_lights,
            feature_bits: main.feature_bits,
            feature_bits_hi: main.feature_bits_hi,
            frame_index: main.frame_index,
            time_s: main.time_s,
            ..FrameParams::default()
        };
        params.set_feature(CAPTURE_FEATURE, true);
        // The direct capture evaluates the full local light. GI's direct-light
        // split expects separate GI lighting of the same view.
        params.set_feature(29, false);
        let mut bytes = [0; FRAME_GPU_SIZE];
        write_camera_frame_gpu(
            &mut bytes,
            scene,
            &params,
            receiver.camera,
            receiver.projection,
        )?;
        if self
            .group
            .as_ref()
            .is_none_or(|(main, _)| main != &shared.frame_bind_group)
        {
            let group = shared.camera_group(gpu, &self.frame);
            self.group = Some((shared.frame_bind_group.clone(), group));
        }
        self.view_proj.copy_from_slice(&main_frame[..64]);
        let queue = gpu.queue();
        queue.write_buffer(&self.frame, 0, &bytes);
        if !self.select.id_bytes().is_empty() {
            queue.write_buffer(&self.ids, 0, self.select.id_bytes());
        }
        Ok(())
    }

    pub(crate) fn ready(&self) -> bool {
        self.receiver.is_some() && self.group.is_some()
    }

    pub(crate) fn frame_group(&self) -> &wgpu::BindGroup {
        &self.group.as_ref().expect("prepared planar capture").1
    }
}

pub(crate) struct PlanarTargets {
    pub(crate) size: (u32, u32),
    pub(crate) color: wgpu::TextureView,
    pub(crate) depth: wgpu::TextureView,
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    group: wgpu::BindGroup,
}

impl PlanarTargets {
    pub(crate) fn new(gpu: &GpuContext, size: (u32, u32), msaa: u32) -> Self {
        let device = gpu.device();
        let (width, height) = capture_size(size);
        let attachment = |label, format, extra_usage| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | extra_usage,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let color = attachment(
            "raa.lit.planar.color",
            HDR_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let depth = attachment(
            "raa.lit.planar.depth",
            DEPTH_FORMAT,
            wgpu::TextureUsages::empty(),
        );
        let uniform = buffer(
            gpu,
            "raa.lit.planar.receiver",
            COMPOSITE_BYTES,
            wgpu::BufferUsages::UNIFORM,
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa.lit.planar.composite"),
            source: wgpu::ShaderSource::Wgsl(include_str!("wgsl/planar_reflection.wgsl").into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("raa.lit.planar.composite"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: STREAM0_STRIDE,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &POSITION,
                })],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                // Match the already drawn receiver across the base and composite shaders.
                bias: wgpu::DepthBiasState {
                    constant: -1,
                    ..Default::default()
                },
            }),
            multisample: wgpu::MultisampleState {
                count: msaa,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa.lit.planar.composite"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&color),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        Self {
            size,
            color,
            depth,
            pipeline,
            uniform,
            group,
        }
    }

    pub(crate) fn composite(
        &self,
        gpu: &GpuContext,
        pass: &mut wgpu::RenderPass<'_>,
        lit: &LitHandle,
        view: &PlanarView,
    ) {
        let Some(receiver) = &view.receiver else {
            return;
        };
        let mut bytes = [0; COMPOSITE_BYTES as usize];
        bytes[..64].copy_from_slice(&view.view_proj);
        receiver.world.write_le_bytes(&mut bytes[64..128]);
        let plane = receiver.reflection.plane.to_array();
        let inverse_size = [1.0 / self.size.0 as f32, 1.0 / self.size.1 as f32, 0.0, 0.0];
        for (out, value) in bytes[128..]
            .chunks_exact_mut(4)
            .zip(plane.into_iter().chain(inverse_size))
        {
            out.copy_from_slice(&value.to_le_bytes());
        }
        gpu.queue().write_buffer(&self.uniform, 0, &bytes);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        for &mesh in &view.meshes {
            if let Some(mesh) = lit.mesh_storage(mesh) {
                pass.set_vertex_buffer(0, mesh.stream0.slice(..));
                pass.set_index_buffer(mesh.index.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(
                    mesh.first_index..mesh.first_index + mesh.index_count,
                    0,
                    0..1,
                );
            }
        }
    }
}

fn buffer(gpu: &GpuContext, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn capture_size(size: (u32, u32)) -> (u32, u32) {
    (size.0.div_ceil(2).max(1), size.1.div_ceil(2).max(1))
}
