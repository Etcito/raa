//! Order-35 `light.tiles` GPU resources and recording (S12): one r32uint D2Array per
//! viewport with `ceil(width/16)` by `ceil(height/16)` texels and `TILE_WORDS` layers,
//! written through a compute-private group 2 and published for forward recording.
//!
//! Handoff: `pub(crate) fn LightTiles::new(gpu: &GpuContext, inputs:
//! Rc<RefCell<TileInputs>>) -> Result<Self, RaaError>` and its `FeaturePass` impl (name
//! `light.tiles`, order 35) in `crates/raa-lit/src/tiles.rs` are attached by
//! `install_local_lights(gpu, pipeline, lit)` in the same file, which adds the pass to
//! the renderer's `Pipeline`. The renderer share keeps the same
//! `Rc<RefCell<TileInputs>>` and, per viewport before prepare, writes one `FrameGpu`
//! snapshot (`light_info` = active count, tile columns, rows and `TILE_WORDS`) plus
//! `LocalLights::write_gpu` records of the registry keyed by stable `LightHandle`.
//! Per-viewport state is keyed by the prepass depth and reused until a resize.
//! Handoff: coverage consumes `pub(crate) fn select_tile(candidates: &[(u64, f32)],
//! out: &mut [u64; 127]) -> (usize, u32)` in `crates/raa-lit/src/tile_selection.rs`,
//! attached by the test module in `lib.rs`; this compute workgroup uses its dense
//! handle-ordered GPU indices and its CPU fixture stays text-only.

#![allow(dead_code)] // Staged before the order-35 installation share lands; see Handoff.

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::{GpuContext, RaaError};
use raa_pipeline::{FeaturePass, LightingResources, PrepareCtx, RecordCtx, Targets};
use raa_shader::{
    Composer, FRAME_GPU_SIZE, LIGHT_GPU_SIZE, MAX_LOCAL_LIGHTS, TILE_SIZE, TILE_WORDS,
};

/// Tile coverage and conservative candidate source, appended after PRELUDE_WGSL.
pub const LIGHT_TILES_WGSL: &str = include_str!("wgsl/light_tiles.wgsl");

/// Compose the registered compute source with the authoritative frame/light ABI/constants.
pub fn light_tiles_source() -> String {
    format!("{}\n{LIGHT_TILES_WGSL}", raa_shader::PRELUDE_WGSL)
}

const LIGHT_BYTES: usize = MAX_LOCAL_LIGHTS * LIGHT_GPU_SIZE;
const MAX_VIEWS: usize = 4;

/// Per-viewport inputs the renderer publishes before `LightTiles::prepare`.
pub(crate) struct TileInputs {
    pub frame: [u8; FRAME_GPU_SIZE],
    pub lights: [u8; LIGHT_BYTES],
}

/// Tile grid of a framebuffer: `ceil(width/16)` by `ceil(height/16)`.
pub(crate) fn tile_grid(size: (u32, u32)) -> (u32, u32) {
    let axis = |v: u32| v.max(1).div_ceil(TILE_SIZE);
    (axis(size.0), axis(size.1))
}

/// The S12 tile-word texture: `grid` texels, one layer per word.
fn texture_desc(grid: (u32, u32)) -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: Some("raa.lit.light_tiles"),
        size: wgpu::Extent3d {
            width: grid.0,
            height: grid.1,
            depth_or_array_layers: TILE_WORDS,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    }
}

fn buffer(
    gpu: &GpuContext,
    label: &'static str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn whole(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn array_view(texture: &wgpu::Texture, label: &'static str) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some(label),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}

/// One viewport's tile texture, overflow counter and compute bind group.
struct ViewTiles {
    /// The prepass depth identifying this viewport's targets.
    depth: wgpu::Texture,
    grid: (u32, u32),
    tile: wgpu::Texture,
    fragment: wgpu::TextureView,
    counter: wgpu::Buffer,
    compute_group: wgpu::BindGroup,
    stamp: u64,
}

/// The `light.tiles` compute pass and its reused per-viewport GPU state.
pub(crate) struct LightTiles {
    inputs: Rc<RefCell<TileInputs>>,
    pipeline: wgpu::ComputePipeline,
    frame: wgpu::Buffer,
    lights: wgpu::Buffer,
    frame_group: wgpu::BindGroup,
    views: Vec<ViewTiles>,
    current: usize,
    clock: u64,
}

impl LightTiles {
    /// CPU-validates the composed WGSL, then creates the compute pipeline synchronously.
    /// Refuses `shader.parse`/`shader.validate` before touching any GPU resource.
    pub(crate) fn new(gpu: &GpuContext, inputs: Rc<RefCell<TileInputs>>) -> Result<Self, RaaError> {
        Composer::validate(&light_tiles_source()).map_err(|e| RaaError::FrameGraph {
            code: e.code,
            message: e.message,
        })?;
        let device = gpu.device();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa.lit.light_tiles"),
            source: wgpu::ShaderSource::Wgsl(light_tiles_source().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("raa.lit.light_tiles"),
            layout: None,
            module: &module,
            entry_point: Some("light_tiles"),
            compilation_options: Default::default(),
            cache: None,
        });
        let frame = buffer(
            gpu,
            "raa.lit.light_tiles.frame",
            FRAME_GPU_SIZE as u64,
            wgpu::BufferUsages::UNIFORM,
        );
        let lights = buffer(
            gpu,
            "raa.lit.light_tiles.lights",
            LIGHT_BYTES as u64,
            wgpu::BufferUsages::UNIFORM,
        );
        let frame_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa.lit.light_tiles.frame"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[whole(0, &frame)],
        });
        Ok(Self {
            inputs,
            pipeline,
            frame,
            lights,
            frame_group,
            views: Vec::new(),
            current: 0,
            clock: 0,
        })
    }

    /// The target's viewport state, allocated only when the depth texture (a new
    /// viewport or a resize) differs from every retained one.
    fn view_index(&mut self, gpu: &GpuContext, targets: &Targets) -> usize {
        self.clock += 1;
        let depth = targets.depth_texture();
        if let Some(index) = self.views.iter().position(|v| v.depth == *depth) {
            self.views[index].stamp = self.clock;
            return index;
        }
        let grid = tile_grid(targets.size());
        let tile = gpu.device().create_texture(&texture_desc(grid));
        let storage = array_view(&tile, "raa.lit.light_tiles.storage");
        let fragment = array_view(&tile, "raa.lit.light_tiles.fragment");
        let counter = buffer(
            gpu,
            "raa.lit.light_tiles.counter",
            4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let resources = [
            (0, wgpu::BindingResource::TextureView(targets.depth())),
            (1, self.lights.as_entire_binding()),
            (2, wgpu::BindingResource::TextureView(&storage)),
            (3, counter.as_entire_binding()),
        ];
        let entries: Vec<_> = resources
            .into_iter()
            .map(|(binding, resource)| wgpu::BindGroupEntry { binding, resource })
            .collect();
        let compute_group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa.lit.light_tiles.group2"),
            layout: &self.pipeline.get_bind_group_layout(2),
            entries: &entries,
        });
        if self.views.len() == MAX_VIEWS {
            self.views.clear();
        }
        self.views.push(ViewTiles {
            depth: depth.clone(),
            grid,
            tile,
            fragment,
            counter,
            compute_group,
            stamp: self.clock,
        });
        self.views.len() - 1
    }

    /// Uploads this viewport's frame/lights and publishes its tile view for forward.
    fn prepare_frame(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let index = self.view_index(ctx.gpu, ctx.targets);
        let inputs = self.inputs.borrow();
        ctx.gpu.queue().write_buffer(&self.frame, 0, &inputs.frame);
        ctx.gpu
            .queue()
            .write_buffer(&self.lights, 0, &inputs.lights);
        ctx.lighting
            .set_local_lights(&inputs.lights, self.views[index].fragment.clone());
        self.current = index;
        Ok(())
    }

    /// Clears the reused overflow counter and dispatches one workgroup per tile.
    fn record_frame(&self, ctx: &mut RecordCtx<'_>) {
        let view = &self.views[self.current];
        ctx.encoder.clear_buffer(&view.counter, 0, None);
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raa.lit.light_tiles"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.frame_group, &[]);
        pass.set_bind_group(2, &view.compute_group, &[]);
        pass.dispatch_workgroups(view.grid.0, view.grid.1, 1);
    }
}

impl FeaturePass for LightTiles {
    fn name(&self) -> &str {
        "light.tiles"
    }

    fn order(&self) -> u32 {
        35
    }

    fn set_enabled(&mut self, enabled: bool, lighting: &mut LightingResources) {
        if !enabled {
            lighting.clear_local_lights();
        }
    }

    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        self.prepare_frame(ctx)
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        self.record_frame(ctx);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_resources_match_s12() {
        assert_eq!((TILE_SIZE, TILE_WORDS, MAX_LOCAL_LIGHTS), (16, 128, 256));
        assert_eq!(LIGHT_BYTES, 16384);
        assert_eq!(tile_grid((1920, 1080)), (120, 68));
        let desc = texture_desc((3, 2));
        assert_eq!(desc.format, wgpu::TextureFormat::R32Uint);
        let size = desc.size;
        assert_eq!(
            (size.width, size.height, size.depth_or_array_layers),
            (3, 2, TILE_WORDS)
        );
    }
}
