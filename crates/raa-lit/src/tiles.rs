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
//! Per-viewport state is keyed by the depth texture the pass resolved and reused until a
//! resize or a different bound depth.
//!
//! The tiling pass resolves its depth through `Targets::depth_input` at
//! `crates/raa-pipeline/src/targets.rs:323 Targets::depth_input` with pass `light.tiles`.
//! That method's Handoff doc comment hands the bound G-buffer depth to card 'raa-lit: light
//! tiles read depth from the bound input' from 'raa-pipeline: G-buffer input views on
//! Targets and group-1 slot views on LightingResources'; this card consumes that entry
//! point and writes no Handoff line of its own.
//! Handoff: coverage consumes `pub(crate) fn select_tile(candidates: &[(u64, f32)],
//! out: &mut [u64; 127]) -> (usize, u32)` in `crates/raa-lit/src/tile_selection.rs`,
//! attached by the test module in `lib.rs`; this compute workgroup uses its dense
//! handle-ordered GPU indices and its CPU fixture stays text-only.

#![allow(dead_code)] // Retained per-view GPU handles outlive their views; see Handoff.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use crate::light_stats::ReadbackCoordinator;
use crate::local_lights::LocalLights;
use crate::renderer::{LitRenderer, pipeline_busy};

use raa_core::{GpuContext, RaaError};
use raa_pipeline::{
    FeaturePass, ImageDesc, ImageDim, LightingResources, Pipeline, PrepareCtx, RecordCtx, Targets,
};
use raa_shader::{
    Composer, FRAME_GPU_SIZE, LIGHT_GPU_SIZE, MAX_LOCAL_LIGHTS, TILE_SIZE, TILE_WORDS,
};

/// S12 `light.tiles` pass name and default order.
pub(crate) const LIGHT_TILES_NAME: &str = "light.tiles";
const LIGHT_TILES_ORDER: u32 = 35;
/// M2-GPU-CONSTANTS logical feature id 24 (tiling); resolved through `feature_address`.
pub(crate) const TILING_FEATURE_ID: u32 = 24;

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
    /// Engine invocation and viewport identity, independent of target-cache position.
    pub index: u64,
    pub view: usize,
    pub readback: Arc<Mutex<ReadbackCoordinator>>,
    /// The light records changed since their last upload (warm frames upload nothing).
    pub lights_dirty: bool,
}

/// Tile grid of a framebuffer: `ceil(width/16)` by `ceil(height/16)`.
pub(crate) fn tile_grid(size: (u32, u32)) -> (u32, u32) {
    let axis = |v: u32| v.max(1).div_ceil(TILE_SIZE);
    (axis(size.0), axis(size.1))
}

/// S10 `light_info` of one viewport: active count (zero while the tiling feature is
/// disabled), tile columns, tile rows and the tile stride in words.
pub(crate) fn light_info(active: u32, viewport: (u32, u32), enabled: bool) -> [u32; 4] {
    let (columns, rows) = tile_grid(viewport);
    [if enabled { active } else { 0 }, columns, rows, TILE_WORDS]
}

/// Copies the renderer-owned registry into the shared tile inputs only when it changed;
/// authored lights stay in the registry whether or not the tiling pass runs.
pub(crate) fn publish_local_lights(inputs: &mut TileInputs, lights: &mut LocalLights) {
    if lights.is_dirty() {
        lights.write_gpu(&mut inputs.lights);
        lights.clear_dirty();
        inputs.lights_dirty = true;
    }
}

/// Registers the order-35 `light.tiles` pass against `lit`'s renderer-owned `LocalLights`.
///
/// Handoff: `pub fn install_local_lights(gpu: &GpuContext, pipeline:
/// &Rc<RefCell<Pipeline>>, lit: &mut LitRenderer) -> Result<(), RaaError>` in
/// `crates/raa-lit/src/tiles.rs` is called by the m28 light installation after
/// `LitRenderer::new`; it attaches the pass to the renderer's `Rc<RefCell<TileInputs>>`,
/// which carries the invocation index/view, the `FrameGpu` snapshot and the dense
/// `LocalLights::write_gpu` records keyed by stable `LightHandle`. Its setters borrow the
/// same registry through `LitRenderer::local_lights`/`local_lights_mut`. A duplicate name,
/// a busy pipeline or a CPU shader-validation error refuses without adding anything.
pub fn install_local_lights(
    gpu: &GpuContext,
    pipeline: &Rc<RefCell<Pipeline>>,
    lit: &mut LitRenderer,
) -> Result<(), RaaError> {
    if pipeline
        .try_borrow()
        .map_err(|_| pipeline_busy())?
        .entry_enabled(LIGHT_TILES_NAME)
        .is_some()
    {
        return Err(RaaError::FrameGraph {
            code: "pipeline.duplicate_pass",
            message: format!("an entry named `{LIGHT_TILES_NAME}` is already in the pipeline"),
        });
    }
    let tiles = LightTiles::new(gpu, lit.tile_inputs())?;
    pipeline
        .try_borrow_mut()
        .map_err(|_| pipeline_busy())?
        .add(Box::new(tiles))
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
    /// The depth texture bound to this viewport's tiling.
    depth: wgpu::Texture,
    grid: (u32, u32),
    tile: wgpu::Texture,
    fragment: wgpu::TextureView,
    counter: wgpu::Buffer,
    compute_group: wgpu::BindGroup,
    stamp: u64,
    /// Every tile word of this view is known to list no light, so a frame without an
    /// eligible light needs no dispatch.
    zeroed: Cell<bool>,
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
    /// Records of the last upload the shader can list (non-black radiance).
    eligible: u32,
}

/// Number of light records whose radiance is not black: the lights the tiling shader
/// can list (`any(color_inv_range_sq.rgb != 0)`, record bytes 16..28).
pub(crate) fn eligible_lights(lights: &[u8; LIGHT_BYTES]) -> u32 {
    lights
        .chunks_exact(LIGHT_GPU_SIZE)
        .filter(|record| {
            record[16..28]
                .chunks_exact(4)
                .any(|c| f32::from_le_bytes(c.try_into().expect("word")) != 0.0)
        })
        .count() as u32
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
        inputs
            .borrow()
            .readback
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .attach(gpu);
        Ok(Self {
            inputs,
            pipeline,
            frame,
            lights,
            frame_group,
            views: Vec::new(),
            current: 0,
            clock: 0,
            eligible: 0,
        })
    }

    /// The bound depth's viewport state, allocated only when its texture (a new
    /// viewport, a resize or a separately bound depth input) differs from every retained
    /// one.
    fn view_index(
        &mut self,
        gpu: &GpuContext,
        targets: &Targets,
        depth: &wgpu::TextureView,
    ) -> usize {
        self.clock += 1;
        if let Some(index) = self.views.iter().position(|v| v.depth == *depth.texture()) {
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
            (0, wgpu::BindingResource::TextureView(depth)),
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
            depth: depth.texture().clone(),
            grid,
            tile,
            fragment,
            counter,
            compute_group,
            stamp: self.clock,
            zeroed: Cell::new(false),
        });
        self.views.len() - 1
    }

    /// Uploads this viewport's frame (always) and lights (only when changed), then
    /// publishes its tile view for forward.
    fn prepare_frame(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let depth = ctx.targets.depth_input(ctx.resources, LIGHT_TILES_NAME);
        let index = self.view_index(ctx.gpu, ctx.targets, &depth);
        let mut inputs = self.inputs.borrow_mut();
        ctx.gpu.queue().write_buffer(&self.frame, 0, &inputs.frame);
        if inputs.lights_dirty {
            ctx.gpu
                .queue()
                .write_buffer(&self.lights, 0, &inputs.lights);
            self.eligible = eligible_lights(&inputs.lights);
            inputs.lights_dirty = false;
        }
        ctx.lighting
            .set_local_lights(&inputs.lights, self.views[index].fragment.clone());
        self.current = index;
        // S55 `lightTiles`: the viewport's tile array at its grid size and layer count.
        let view = &self.views[index];
        let desc = ImageDesc::new(
            ImageDim::D2Array,
            wgpu::TextureFormat::R32Uint,
            view.grid.0,
            view.grid.1,
            TILE_WORDS,
            1,
        );
        ctx.resources
            .publish(LIGHT_TILES_NAME, "lightTiles", &view.fragment, desc);
        Ok(())
    }

    /// Clears the reused overflow counter and dispatches one workgroup per tile. With no
    /// eligible light the tile words are all zero counts: they are written once (by the
    /// dispatch of the first such frame) and the sweep is skipped afterwards.
    fn record_frame(&self, ctx: &mut RecordCtx<'_>) {
        let view = &self.views[self.current];
        ctx.encoder.clear_buffer(&view.counter, 0, None);
        let skip = self.eligible == 0 && view.zeroed.get();
        view.zeroed.set(self.eligible == 0);
        if !skip {
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
        let inputs = self.inputs.borrow();
        inputs
            .readback
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .copy(inputs.index, inputs.view, ctx.encoder, &view.counter);
    }
}

impl FeaturePass for LightTiles {
    fn name(&self) -> &str {
        LIGHT_TILES_NAME
    }

    fn order(&self) -> u32 {
        LIGHT_TILES_ORDER
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

    fn inputs() -> TileInputs {
        TileInputs {
            frame: [0; FRAME_GPU_SIZE],
            lights: [0; LIGHT_BYTES],
            index: 0,
            view: 0,
            readback: Arc::new(Mutex::new(ReadbackCoordinator::new())),
            lights_dirty: false,
        }
    }

    #[test]
    fn tile_resources_match_s12() {
        assert_eq!((TILE_SIZE, TILE_WORDS, MAX_LOCAL_LIGHTS), (16, 128, 256));
        assert_eq!(LIGHT_BYTES, 16384);
        assert_eq!(tile_grid((1920, 1080)), (120, 68));
        assert_eq!(tile_grid((1, 1)), (1, 1));
        assert_eq!(tile_grid((17, 33)), (2, 3));
        let desc = texture_desc((3, 2));
        assert_eq!(desc.format, wgpu::TextureFormat::R32Uint);
        let size = desc.size;
        assert_eq!(
            (size.width, size.height, size.depth_or_array_layers),
            (3, 2, TILE_WORDS)
        );
    }

    #[test]
    fn only_lights_with_radiance_are_eligible() {
        let mut lights = LocalLights::new();
        let mut off = crate::local_light::LocalLight::default();
        off.intensity = 0.0;
        lights.add(off).expect("off light");
        let mut bytes = [0u8; LIGHT_BYTES];
        lights.write_gpu(&mut bytes);
        assert_eq!(eligible_lights(&bytes), 0, "an off light is never eligible");
        let mut on = crate::local_light::LocalLight::default();
        on.intensity = 2.0;
        lights.add(on).expect("lit light");
        lights.write_gpu(&mut bytes);
        assert_eq!(eligible_lights(&bytes), 1);
    }

    #[test]
    fn light_info_zeroes_counts_while_disabled() {
        assert_eq!(light_info(7, (1920, 1080), true), [7, 120, 68, TILE_WORDS]);
        assert_eq!(light_info(7, (1920, 1080), false), [0, 120, 68, TILE_WORDS]);
        assert_eq!(light_info(0, (16, 16), true), [0, 1, 1, TILE_WORDS]);
    }

    #[test]
    fn unchanged_registry_uploads_nothing() {
        let mut inputs = inputs();
        let mut lights = LocalLights::new();
        publish_local_lights(&mut inputs, &mut lights);
        assert!(
            !inputs.lights_dirty,
            "an empty registry has nothing to upload"
        );
        let handle = lights
            .add(crate::local_light::LocalLight::default())
            .expect("light");
        publish_local_lights(&mut inputs, &mut lights);
        assert!(inputs.lights_dirty, "the first light is uploaded");
        assert!(!lights.is_dirty(), "the upload marks the registry clean");
        inputs.lights_dirty = false; // the pass uploaded the records
        let mut sentinel = inputs.lights;
        sentinel[0] ^= 0xff;
        inputs.lights = sentinel;
        publish_local_lights(&mut inputs, &mut lights);
        assert_eq!(
            inputs.lights, sentinel,
            "an unchanged registry is not rewritten"
        );
        assert!(!inputs.lights_dirty);
        lights
            .remove(handle)
            .expect("removing the light dirties the registry");
        publish_local_lights(&mut inputs, &mut lights);
        assert!(inputs.lights_dirty && !lights.is_dirty());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn per_view_state_is_reused_and_resized() {
        let test = "per_view_state_is_reused_and_resized";
        let Some(gpu) = raa_core::testing::software_gpu() else {
            raa_core::testing::skip(test, "software_gpu returned None");
            return;
        };
        let mut tiles = LightTiles::new(&gpu, Rc::new(RefCell::new(inputs()))).expect("tiles");
        let first = Targets::new(&gpu, (64, 48));
        let first_depth = first.depth().clone();
        let a = tiles.view_index(&gpu, &first, &first_depth);
        assert_eq!(tiles.views[a].grid, (4, 3));
        assert_eq!(
            tiles.view_index(&gpu, &first, &first_depth),
            a,
            "the same depth is reused"
        );
        let resized = Targets::new(&gpu, (32, 16));
        let resized_depth = resized.depth().clone();
        let b = tiles.view_index(&gpu, &resized, &resized_depth);
        assert_ne!(b, a, "a new depth allocates new per-viewport state");
        assert_eq!(tiles.views[b].grid, (2, 1));
        for _ in 0..MAX_VIEWS {
            let fresh = Targets::new(&gpu, (8, 8));
            tiles.view_index(&gpu, &fresh, fresh.depth());
            assert!(tiles.views.len() <= MAX_VIEWS, "stale views are disposed");
        }
        assert_eq!(tiles.views.last().map(|v| v.grid), Some((1, 1)));
    }

    /// The bind group is keyed on the texture behind the resolved depth view: resolving
    /// twice with the target depth reuses one `ViewTiles`, while a separately bound
    /// Depth32Float view of the same size adds a second entry keyed on its texture. The
    /// targets and the bound texture start cleared; the tiling pass reads no depth values
    /// during prepare, so no mesh pass runs first.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn bound_depth_selects_view_tiles() {
        use raa_pipeline::{PublishedImage, ResourceTable};

        let test = "bound_depth_selects_view_tiles";
        let Some(gpu) = raa_core::testing::software_gpu() else {
            raa_core::testing::skip(test, "software_gpu returned None");
            return;
        };
        let mut tiles = LightTiles::new(&gpu, Rc::new(RefCell::new(inputs()))).expect("tiles");
        let targets = Targets::new(&gpu, (64, 48));
        let table = ResourceTable::new();
        let a = tiles.view_index(
            &gpu,
            &targets,
            &targets.depth_input(&table, LIGHT_TILES_NAME),
        );
        assert_eq!(
            tiles.view_index(
                &gpu,
                &targets,
                &targets.depth_input(&table, LIGHT_TILES_NAME)
            ),
            a,
            "the unbound view resolves to the target depth"
        );
        assert_eq!(tiles.views.len(), 1);
        let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("test.bound_depth"),
            size: wgpu::Extent3d {
                width: 64,
                height: 48,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let bound = texture.create_view(&wgpu::TextureViewDescriptor::default());
        table.bind_input(
            LIGHT_TILES_NAME,
            "depth",
            PublishedImage {
                view: bound.clone(),
                desc: ImageDesc::new(
                    ImageDim::D2,
                    wgpu::TextureFormat::Depth32Float,
                    64,
                    48,
                    1,
                    1,
                ),
            },
        );
        let b = tiles.view_index(
            &gpu,
            &targets,
            &targets.depth_input(&table, LIGHT_TILES_NAME),
        );
        assert_ne!(b, a, "a separately bound depth adds per-viewport state");
        assert!(tiles.views[b].depth == *bound.texture());
        assert_eq!(
            tiles.view_index(
                &gpu,
                &targets,
                &targets.depth_input(&table, LIGHT_TILES_NAME)
            ),
            b,
            "the bound depth is reused"
        );
        assert_eq!(tiles.views.len(), 2);
    }

    /// `prepare` publishes the tile array under (`light.tiles`, `lightTiles`) at its grid
    /// size with `TILE_WORDS` layers. The shared inputs are the zeroed `inputs()` of this
    /// module (frame and light records, a fresh readback); no mesh pass runs first, and
    /// prepare reads no depth values.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn prepare_publishes_the_tile_array() {
        use raa_pipeline::ResourceTable;

        let test = "prepare_publishes_the_tile_array";
        let Some(gpu) = raa_core::testing::software_gpu() else {
            raa_core::testing::skip(test, "software_gpu returned None");
            return;
        };
        let mut tiles = LightTiles::new(&gpu, Rc::new(RefCell::new(inputs()))).expect("tiles");
        let targets = Targets::new(&gpu, (8, 8));
        let table = ResourceTable::new();
        let mut lighting = LightingResources::new(&gpu);
        let mut ctx = PrepareCtx {
            gpu: &gpu,
            size: (8, 8),
            frame: 0,
            time_s: 0.0,
            dt_s: 0.0,
            targets: &targets,
            lighting: &mut lighting,
            resources: &table,
        };
        tiles.prepare(&mut ctx).expect("prepare");
        let entries = table.entries();
        let keys: Vec<_> = entries
            .iter()
            .map(|(pass, port, _)| (pass.as_str(), port.as_str()))
            .collect();
        assert_eq!(keys, [("light.tiles", "lightTiles")]);
        let desc = entries[0].2;
        assert_eq!(desc.format, wgpu::TextureFormat::R32Uint);
        assert_eq!(
            (desc.dim, desc.width, desc.height, desc.layers, desc.mips),
            (ImageDim::D2Array, 1, 1, TILE_WORDS, 1)
        );
    }
}
