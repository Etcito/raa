//! Optional editor overlays. No textures, buffers, uploads or commands while inactive.
use crate::{DrawClass, LitHandle, LitMeshId};
use raa_core::RaaError;
use raa_pipeline::{FeaturePass, RecordCtx};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WireframeMode {
    #[default]
    Off,
    Overlay,
    Only,
}

#[derive(Default)]
pub(crate) struct EditorState {
    pub global: WireframeMode,
    pub wire: HashMap<u32, WireframeMode>,
    pub outlines: HashMap<u32, [f32; 3]>,
}
impl EditorState {
    fn active(&self) -> bool {
        self.global != WireframeMode::Off
            || self.wire.values().any(|m| *m != WireframeMode::Off)
            || !self.outlines.is_empty()
    }
    fn mode(&self, tag: u32) -> WireframeMode {
        self.wire.get(&tag).copied().unwrap_or(self.global)
    }
}
fn tag(bytes: &[u8], i: u32) -> u32 {
    bytes
        .get(i as usize * 128 + 116..i as usize * 128 + 120)
        .and_then(|b| b.try_into().ok())
        .map(u32::from_le_bytes)
        .unwrap_or(0)
}
impl LitHandle {
    /// Global default; an instance override wins.
    pub fn set_wireframe(&self, mode: WireframeMode, instance: Option<u32>) {
        let mut s = self.shared.borrow_mut();
        if let Some(tag) = instance {
            s.editor.wire.insert(tag, mode);
        } else {
            s.editor.global = mode;
        }
    }
    /// Removes the instance wireframe override, restoring the global default.
    pub fn clear_wireframe(&self, instance: u32) {
        self.shared.borrow_mut().editor.wire.remove(&instance);
    }
    /// Outline colour is display sRGB. None disables this instance's outline.
    pub fn set_outline(&self, instance: u32, color: Option<[f32; 3]>) {
        let mut s = self.shared.borrow_mut();
        if let Some(color) = color {
            s.editor.outlines.insert(instance, color);
        } else {
            s.editor.outlines.remove(&instance);
        }
    }
    /// Draws surface instances, respecting editor edges-only overrides.
    /// Contributors drawing mesh surfaces should use this instead of draw_indexed.
    pub fn draw_surface_instances(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        indices: std::ops::Range<u32>,
        instances: std::ops::Range<u32>,
    ) {
        let s = self.shared.borrow();
        if s.editor.global != WireframeMode::Only
            && !s.editor.wire.values().any(|m| *m == WireframeMode::Only)
        {
            pass.draw_indexed(indices, 0, instances);
            return;
        }
        for i in instances {
            if s.editor.mode(tag(s.prep.instance_bytes(), i)) != WireframeMode::Only {
                pass.draw_indexed(indices.clone(), 0, i..i + 1);
            }
        }
    }

    pub(crate) fn editor_active(&self) -> bool {
        self.shared.borrow().editor.active()
    }
    /// Replaces optional polygon edges after the caller validates all vertex indices.
    pub fn set_mesh_edges(&self, mesh: LitMeshId, m: &crate::AttributedMesh, edges: &[u32]) {
        self.shared
            .borrow_mut()
            .meshes
            .set_edges(mesh, crate::meshes::edge_bytes(m, edges));
    }
}

pub(crate) struct EditorPass {
    lit: LitHandle,
    resources: Option<Resources>,
    bytes: Vec<u8>,
}
struct Resources {
    size: (u32, u32),
    format: wgpu::TextureFormat,
    settings: wgpu::Buffer,
    capacity: u64,
    mask: wgpu::TextureView,
    group: wgpu::BindGroup,
    mask_group: wgpu::BindGroup,
    wire: wgpu::RenderPipeline,
    wire_posed: wgpu::RenderPipeline,
    silhouette: wgpu::RenderPipeline,
    silhouette_double: wgpu::RenderPipeline,
    only: wgpu::RenderPipeline,
    only_double: wgpu::RenderPipeline,
    outline: wgpu::RenderPipeline,
}
impl EditorPass {
    pub(crate) fn new(lit: LitHandle) -> Self {
        Self {
            lit,
            resources: None,
            bytes: Vec::new(),
        }
    }
}
const INSTANCE: [wgpu::VertexAttribute; 4] =
    wgpu::vertex_attr_array![2=>Float32x4,3=>Float32x4,4=>Float32x4,5=>Float32x4];
const SETTINGS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![6=>Float32x4,7=>Float32x4];
const SEGMENT: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0=>Float32x3,1=>Float32x3];
/// A posed edge record holds two S1 vertices: the second starts at byte 24.
const POSED_SEGMENT: [wgpu::VertexAttribute; 2] = [
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 24, shader_location: 1 },
];
const POSITION: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0=>Float32x3];
fn instance(stride: u64) -> Option<wgpu::VertexBufferLayout<'static>> {
    Some(wgpu::VertexBufferLayout {
        array_stride: stride,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &INSTANCE,
    })
}
fn settings_layout(stride: u64) -> Option<wgpu::VertexBufferLayout<'static>> {
    Some(wgpu::VertexBufferLayout {
        array_stride: stride,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &SETTINGS,
    })
}
impl Resources {
    fn new(ctx: &RecordCtx<'_>, lit: &LitHandle, capacity: u64) -> Self {
        let d = ctx.gpu.device();
        let settings = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa.editor.settings"),
            size: capacity,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mask = d
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("raa.editor.mask"),
                size: wgpu::Extent3d {
                    width: ctx.size.0.max(1),
                    height: ctx.size.1.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa.editor.textures"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let group = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa.editor.textures"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(ctx.targets.depth()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&mask),
                },
            ],
        });
        let mask_group = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa.editor.mask-depth"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(ctx.targets.depth()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(ctx.targets.normal()),
                },
            ],
        });
        let frame_layout = lit.frame_layout();
        let pipeline_layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raa.editor"),
            bind_group_layouts: &[Some(&frame_layout), Some(&layout)],
            immediate_size: 0,
        });
        let shader = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa.editor"),
            source: wgpu::ShaderSource::Wgsl(include_str!("wgsl/editor.wgsl").into()),
        });
        let build = |vertex,
                     fragment,
                     buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
                     format,
                     blend,
                     cull| {
            d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(fragment),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vertex),
                    compilation_options: Default::default(),
                    buffers,
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode: cull,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fragment),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let wire_buffers = |stride, attributes| {
            [
                Some(wgpu::VertexBufferLayout {
                    array_stride: stride,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes,
                }),
                instance(0),
                settings_layout(0),
            ]
        };
        let wire = build(
            "vs_wire",
            "fs_wire",
            &wire_buffers(24, &SEGMENT),
            ctx.backbuffer_format,
            Some(wgpu::BlendState::ALPHA_BLENDING),
            None,
        );
        let wire_posed = build(
            "vs_wire",
            "fs_wire",
            &wire_buffers(crate::SKINNED_EDGE_STRIDE, &POSED_SEGMENT),
            ctx.backbuffer_format,
            Some(wgpu::BlendState::ALPHA_BLENDING),
            None,
        );
        let silhouette = build(
            "vs_mask",
            "fs_mask",
            &[
                Some(wgpu::VertexBufferLayout {
                    array_stride: 24,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &POSITION,
                }),
                instance(128),
                settings_layout(32),
            ],
            wgpu::TextureFormat::Rgba8Unorm,
            None,
            Some(wgpu::Face::Back),
        );
        let silhouette_double = build(
            "vs_mask",
            "fs_mask",
            &[
                Some(wgpu::VertexBufferLayout {
                    array_stride: 24,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &POSITION,
                }),
                instance(128),
                settings_layout(32),
            ],
            wgpu::TextureFormat::Rgba8Unorm,
            None,
            None,
        );
        let only_buffers = [
            Some(wgpu::VertexBufferLayout {
                array_stride: 24,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &POSITION,
            }),
            instance(128),
            settings_layout(32),
        ];
        let only = build(
            "vs_only",
            "fs_only",
            &only_buffers,
            ctx.backbuffer_format,
            None,
            Some(wgpu::Face::Back),
        );
        let only_double = build(
            "vs_only",
            "fs_only",
            &only_buffers,
            ctx.backbuffer_format,
            None,
            None,
        );
        let outline = build(
            "vs_full",
            "fs_outline",
            &[],
            ctx.backbuffer_format,
            Some(wgpu::BlendState::ALPHA_BLENDING),
            None,
        );
        Self {
            size: ctx.size,
            format: ctx.backbuffer_format,
            settings,
            capacity,
            mask,
            group,
            mask_group,
            wire,
            wire_posed,
            silhouette,
            silhouette_double,
            only,
            only_double,
            outline,
        }
    }
}
fn attachment(
    view: &wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
) -> Option<wgpu::RenderPassColorAttachment<'_>> {
    Some(wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load,
            store: wgpu::StoreOp::Store,
        },
    })
}
impl FeaturePass for EditorPass {
    fn name(&self) -> &str {
        "editor"
    }
    fn order(&self) -> u32 {
        210
    }
    fn output_overlay(&self) -> bool {
        true
    }
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if !self.lit.editor_active() {
            return Ok(());
        }
        let mut s = self.lit.shared.borrow_mut();
        let wire = s.editor.global != WireframeMode::Off
            || s.editor.wire.values().any(|m| *m != WireframeMode::Off);
        let outlined = !s.editor.outlines.is_empty();
        if wire {
            s.meshes.prepare_edges(ctx.gpu);
        }
        self.bytes.clear();
        for record in s.prep.instance_bytes().chunks_exact(128) {
            let tag = u32::from_le_bytes(
                record[116..120]
                    .try_into()
                    .map_err(|_| RaaError::Resource("invalid instance record".into()))?,
            );
            let mode = match s.editor.mode(tag) {
                WireframeMode::Off => 0.0_f32,
                WireframeMode::Overlay => 1.0,
                WireframeMode::Only => 2.0,
            };
            let color = s.editor.outlines.get(&tag);
            let c = color.copied().unwrap_or([0.0; 3]);
            let background = ctx.lighting.env().background;
            for f in [
                background[0],
                background[1],
                background[2],
                mode,
                c[0],
                c[1],
                c[2],
                if color.is_some() { 1.0 } else { 0.0 },
            ] {
                self.bytes.extend_from_slice(&f.to_le_bytes());
            }
        }
        drop(s);
        let capacity = (self.bytes.len() as u64).max(32).next_power_of_two();
        if self.resources.as_ref().is_none_or(|r| {
            r.size != ctx.size || r.format != ctx.backbuffer_format || r.capacity < capacity
        }) {
            self.resources = Some(Resources::new(ctx, &self.lit, capacity));
        }
        let Some(r) = &self.resources else {
            return Ok(());
        };
        if !self.bytes.is_empty() {
            ctx.gpu.queue().write_buffer(&r.settings, 0, &self.bytes);
        }
        let frame = self.lit.frame_bind_group();
        let s = self.lit.shared.borrow();
        if outlined {
            // Mask must not bind itself as a sampled resource while being an attachment.
            let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("raa.editor.mask"),
                color_attachments: &[attachment(
                    &r.mask,
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                )],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&r.silhouette);
            pass.set_bind_group(0, &frame, &[]);
            pass.set_bind_group(1, &r.mask_group, &[]);
            pass.set_vertex_buffer(1, s.instance_buffer.slice(..));
            pass.set_vertex_buffer(2, r.settings.slice(..));
            for class in [DrawClass::Opaque, DrawClass::Fur, DrawClass::Transparent] {
                for draw in s.prep.draws(class) {
                    let double_sided = s
                        .materials
                        .get(draw.material)
                        .is_some_and(|m| m.flags & crate::FLAG_DOUBLE_SIDED != 0);
                    pass.set_pipeline(if double_sided {
                        &r.silhouette_double
                    } else {
                        &r.silhouette
                    });
                    let (mesh, first, count) = s.meshes.resolve(draw.mesh)?;
                    pass.set_index_buffer(mesh.index.slice(..), wgpu::IndexFormat::Uint32);
                    for i in draw.first_instance..draw.first_instance + draw.instance_count {
                        let tag = tag(s.prep.instance_bytes(), i);
                        if s.editor.outlines.contains_key(&tag) {
                            // Selected instances write their own posed mesh view.
                            let geometry = crate::skin_editor::OverlayGeometry::select(
                                &s.skin_source,
                                tag,
                                &mesh.stream0,
                                mesh.edges.as_ref(),
                            );
                            pass.set_vertex_buffer(0, geometry.stream0.slice(..));
                            pass.draw_indexed(first..first + count, 0, i..i + 1);
                        }
                    }
                }
            }
        }
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("raa.editor.overlay"),
            color_attachments: &[attachment(ctx.backbuffer, wgpu::LoadOp::Load)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(0, &frame, &[]);
        pass.set_bind_group(1, &r.group, &[]);
        if wire {
            pass.set_vertex_buffer(1, s.instance_buffer.slice(..));
            pass.set_vertex_buffer(2, r.settings.slice(..));
            for class in [DrawClass::Opaque, DrawClass::Fur, DrawClass::Transparent] {
                for draw in s.prep.draws(class) {
                    let double_sided = s
                        .materials
                        .get(draw.material)
                        .is_some_and(|m| m.flags & crate::FLAG_DOUBLE_SIDED != 0);
                    pass.set_pipeline(if double_sided {
                        &r.only_double
                    } else {
                        &r.only
                    });
                    let (mesh, first, count) = s.meshes.resolve(draw.mesh)?;
                    pass.set_index_buffer(mesh.index.slice(..), wgpu::IndexFormat::Uint32);
                    for i in draw.first_instance..draw.first_instance + draw.instance_count {
                        let tag = tag(s.prep.instance_bytes(), i);
                        if s.editor.mode(tag) == WireframeMode::Only {
                            // The flat fill must cover the same silhouette as the posed edges.
                            let geometry = crate::skin_editor::OverlayGeometry::select(
                                &s.skin_source,
                                tag,
                                &mesh.stream0,
                                mesh.edges.as_ref(),
                            );
                            pass.set_vertex_buffer(0, geometry.stream0.slice(..));
                            pass.draw_indexed(first..first + count, 0, i..i + 1);
                        }
                    }
                }
            }
            for class in [DrawClass::Opaque, DrawClass::Fur, DrawClass::Transparent] {
                for draw in s.prep.draws(class) {
                    let Some(record) = s.meshes.submesh(draw.mesh) else {
                        continue;
                    };
                    if s.meshes.submesh_ids(record.lit).and_then(|ids| ids.first())
                        != Some(&draw.mesh)
                    {
                        continue;
                    }
                    let (mesh, _, _) = s.meshes.resolve(draw.mesh)?;
                    for i in draw.first_instance..draw.first_instance + draw.instance_count {
                        let tag = tag(s.prep.instance_bytes(), i);
                        if s.editor.mode(tag) == WireframeMode::Off {
                            continue;
                        }
                        // Each instance expands its own rest or posed edge segments.
                        let geometry = crate::skin_editor::OverlayGeometry::select(
                            &s.skin_source,
                            tag,
                            &mesh.stream0,
                            mesh.edges.as_ref(),
                        );
                        let Some(edges) = &geometry.edges else {
                            continue;
                        };
                        pass.set_pipeline(if geometry.edge_stride
                            == crate::SKINNED_EDGE_STRIDE
                        {
                            &r.wire_posed
                        } else {
                            &r.wire
                        });
                        pass.set_vertex_buffer(0, edges.slice(..));
                        // Repeat the world/settings record for all segments of this instance.
                        pass.set_vertex_buffer(
                            1,
                            s.instance_buffer
                                .slice(i as u64 * 128..i as u64 * 128 + 128),
                        );
                        pass.set_vertex_buffer(
                            2,
                            r.settings.slice(i as u64 * 32..i as u64 * 32 + 32),
                        );
                        // Instance streams 1/2 have a zero stride in the wire pipeline.
                        pass.draw(0..6, 0..geometry.edge_count());
                    }
                }
            }
        }
        if outlined {
            pass.set_pipeline(&r.outline);
            pass.draw(0..3, 0..1);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shader_validates() {
        raa_shader::Composer::validate(include_str!("wgsl/editor.wgsl")).unwrap();
    }
    #[test]
    fn instance_override_and_multiple_outlines() {
        let mut state = EditorState::default();
        assert!(!state.active());
        state.global = WireframeMode::Overlay;
        state.wire.insert(7, WireframeMode::Off);
        assert_eq!(state.mode(7), WireframeMode::Off);
        assert_eq!(state.mode(8), WireframeMode::Overlay);
        state.outlines.insert(7, [1.0, 0.0, 0.0]);
        state.outlines.insert(8, [0.0, 1.0, 0.0]);
        assert_eq!(state.outlines.len(), 2);
    }
}
