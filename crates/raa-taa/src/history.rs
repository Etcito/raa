//! The reusable ping-pong history (contract `taa-history`): a CPU parity state plus two
//! textures of the surface format that temporal passes alternate between.

use raa_core::GpuContext;
use raa_core::wgpu;

/// Usages of both history textures (sampled, copied and writable by compute).
const HISTORY_USAGE: wgpu::TextureUsages = wgpu::TextureUsages::TEXTURE_BINDING
    .union(wgpu::TextureUsages::STORAGE_BINDING)
    .union(wgpu::TextureUsages::COPY_SRC)
    .union(wgpu::TextureUsages::COPY_DST);

/// Clamps each axis to at least 1, the size every texture and state store.
fn clamped(size: (u32, u32)) -> (u32, u32) {
    (size.0.max(1), size.1.max(1))
}

/// CPU-side parity and accumulated-frame validity of a [`HistoryPair`]: the write index is
/// the texture the next recorded frame writes, the read index the one the last commit wrote.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct HistoryState {
    size: (u32, u32),
    write_index: usize,
    accumulated: u32,
}

impl HistoryState {
    /// A new invalid state of the clamped size: write index 0, 0 accumulated frames.
    pub fn new(size: (u32, u32)) -> HistoryState {
        HistoryState {
            size: clamped(size),
            write_index: 0,
            accumulated: 0,
        }
    }

    /// The clamped size the pair's textures have.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Stores a different clamped size and invalidates; false when the size is unchanged.
    pub fn resize(&mut self, size: (u32, u32)) -> bool {
        let size = clamped(size);
        if size == self.size {
            return false;
        }
        self.size = size;
        self.accumulated = 0;
        true
    }

    /// The texture the next recorded frame writes (0 or 1).
    pub fn write_index(&self) -> usize {
        self.write_index
    }

    /// The other index: the texture the last committed frame wrote.
    pub fn read_index(&self) -> usize {
        self.write_index ^ 1
    }

    /// Flips the write index and adds one accumulated frame (saturating at `u32::MAX`).
    pub fn commit(&mut self) {
        self.write_index ^= 1;
        self.accumulated = self.accumulated.saturating_add(1);
    }

    /// Drops the accumulated frames to 0; the write index is unchanged.
    pub fn invalidate(&mut self) {
        self.accumulated = 0;
    }

    /// How many frames were committed since the last invalidation or resize.
    pub fn accumulated_frames(&self) -> u32 {
        self.accumulated
    }

    /// True when at least one frame was committed since the last invalidation or resize.
    pub fn is_valid(&self) -> bool {
        self.accumulated > 0
    }

    /// Usable only when this history is valid and the caller's frame history is too.
    pub fn usable(&self, frame_history_valid: bool) -> bool {
        self.is_valid() && frame_history_valid
    }
}

/// One history texture with its default full view (the `Target` pattern of
/// raa-pipeline's `targets.rs`).
struct Target {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Target {
    fn new(gpu: &GpuContext, label: &str, size: (u32, u32), format: wgpu::TextureFormat) -> Target {
        let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: HISTORY_USAGE,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Target { texture, view }
    }
}

/// Two textures created from `label` plus ".0" and ".1" (two-element ping-pong).
fn create_pair(
    gpu: &GpuContext,
    label: &str,
    format: wgpu::TextureFormat,
    size: (u32, u32),
) -> [Target; 2] {
    [
        Target::new(gpu, &format!("{label}.0"), size, format),
        Target::new(gpu, &format!("{label}.1"), size, format),
    ]
}

/// A GPU ping-pong history at render resolution: two textures plus the [`HistoryState`]
/// that picks which one is read and which one is written.
///
/// Handoff: `raa-taa: TAA bind group layout, pipeline and per-size bind groups` builds bind
/// groups from `HistoryPair::view`, which today returns the default view of the texture at
/// `index & 1` in this pair.
/// Handoff: `raa-taa: taa feature pass, install and handle` calls `HistoryPair::new` and
/// `HistoryPair::ensure` to create and recreate this pair, and reaches `HistoryState::commit`
/// and `HistoryState::invalidate` through `HistoryPair::state_mut` to flip and reset parity.
pub struct HistoryPair {
    textures: [Target; 2],
    state: HistoryState,
    label: &'static str,
    format: wgpu::TextureFormat,
}

impl HistoryPair {
    /// Creates both clamped-size textures with the history usages and a new state.
    pub fn new(
        gpu: &GpuContext,
        label: &'static str,
        format: wgpu::TextureFormat,
        size: (u32, u32),
    ) -> HistoryPair {
        let size = clamped(size);
        HistoryPair {
            textures: create_pair(gpu, label, format, size),
            state: HistoryState::new(size),
            label,
            format,
        }
    }

    /// Recreates both textures only when the state's resize reports a new size.
    pub fn ensure(&mut self, gpu: &GpuContext, size: (u32, u32)) -> bool {
        if !self.state.resize(size) {
            return false;
        }
        self.textures = create_pair(gpu, self.label, self.format, self.state.size());
        true
    }

    /// The parity and validity of this pair.
    pub fn state(&self) -> &HistoryState {
        &self.state
    }

    /// Mutable parity used to commit or invalidate a recorded frame.
    pub fn state_mut(&mut self) -> &mut HistoryState {
        &mut self.state
    }

    /// Texture at `index & 1`.
    pub fn texture(&self, index: usize) -> &wgpu::Texture {
        &self.textures[index & 1].texture
    }

    /// Default view of the texture at `index & 1`.
    pub fn view(&self, index: usize) -> &wgpu::TextureView {
        &self.textures[index & 1].view
    }

    /// View of the texture the last committed frame wrote.
    pub fn read_view(&self) -> &wgpu::TextureView {
        self.view(self.state.read_index())
    }

    /// View of the texture the next recorded frame writes.
    pub fn write_view(&self) -> &wgpu::TextureView {
        self.view(self.state.write_index())
    }

    /// The format both textures were created with.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// Two textures of `width x height` at the format's block copy size (8 for Rgba16Float).
    pub fn allocated_bytes(&self) -> u64 {
        let (width, height) = self.state.size();
        let block = u64::from(self.format.block_copy_size(None).unwrap_or(0));
        2 * u64::from(width) * u64::from(height) * block
    }
}
