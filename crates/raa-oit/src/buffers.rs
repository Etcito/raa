//! GPU buffers of the linked-list OIT (OIT value): heads, node pool, node counter and the
//! uniforms, the group-2 layout of `PASS_TRANSPARENT_WGSL`, the bind groups of the
//! transparent draw, of the head clear (`CLEAR_WGSL`) and of the resolve (`RESOLVE_WGSL`),
//! and the overflow readback.
//!
//! Everything is allocated by `create` / `rebuild`; a frame only records (`copy_counter`)
//! and maps (`poll`). Readback: `copy_counter` copies the counter into the next free staging
//! buffer of a ring of 3 (all busy: the copy is skipped), the caller submits, then the next
//! `poll` maps the buffers written since and returns the newest counter read. The map
//! callback stores the value with its sequence number and unmaps. On wasm the browser drives
//! the callbacks; on native `poll` polls the device without blocking. The shared slot is an
//! `Arc<Mutex<..>>` because wgpu requires a `Send` map callback on native.
//!
//! Growth: a counter read above the pool capacity is an overflow frame; the factor in use
//! doubles (bounded by the adapter limit) and `needs_rebuild` tells the pass to recreate the
//! buffers at its next prepare.

#![allow(
    dead_code,
    reason = "used by the transparent.oit FeaturePass (later task)"
)]

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use raa_core::GpuContext;

use crate::node::{NODE_SIZE, pool_nodes};
use crate::shaders::{CLEAR_PARAMS_SIZE, CLEAR_WORKGROUP_SIZE, RESOLVE_PARAMS_SIZE};

/// Number of staging buffers of the counter readback ring.
pub(crate) const STAGING_RING: usize = 3;
/// Size of the group-2 OitParams uniform (width, height, capacity, 0).
pub(crate) const OIT_PARAMS_SIZE: u64 = 16;
/// Size of the node counter (one atomic u32).
const COUNTER_SIZE: u64 = 4;

/// Bytes the node pool may use: min(max_storage_buffer_binding_size, max_buffer_size).
pub(crate) fn pool_limit_bytes(limits: &wgpu::Limits) -> u64 {
    limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size)
}

/// Smallest factor whose pool reaches the limit for `pixels` (larger factors give the same
/// pool), at least 1.
pub(crate) fn max_factor(pixels: u64, limit_bytes: u64) -> u32 {
    let cap = limit_bytes / NODE_SIZE as u64;
    let f = cap.div_ceil(pixels.max(1)).max(1);
    u32::try_from(f).unwrap_or(u32::MAX)
}

/// Clear dispatch (x, y) for `count` heads: ceil(count / 64) workgroups, split over y when
/// x would exceed `max_per_dimension` (the shader's flat index is y * groups.x * 64 + x).
pub(crate) fn clear_groups(count: u32, max_per_dimension: u32) -> (u32, u32) {
    let total = count.div_ceil(CLEAR_WORKGROUP_SIZE).max(1);
    let x = total.min(max_per_dimension.max(1));
    (x, total.div_ceil(x))
}

/// Pool growth bookkeeping (pure; carried over by `rebuild`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Growth {
    /// Pool factor in use (nodes per pixel before the limit clamp).
    pub(crate) factor: u32,
    /// Counter reads that exceeded the capacity.
    pub(crate) overflow_frames: u32,
    /// The factor grew since the buffers were created.
    pub(crate) rebuild: bool,
}

impl Growth {
    pub(crate) fn new(factor: u32) -> Growth {
        Growth {
            factor: factor.max(1),
            overflow_frames: 0,
            rebuild: false,
        }
    }

    /// One counter read against the current pool. Above `capacity`: overflow_frames += 1 and,
    /// unless a rebuild is already pending, the factor doubles, clamped to
    /// `max_factor(pixels, limit_bytes)`; a factor already at the bound stays and asks for
    /// no rebuild.
    pub(crate) fn observe(&mut self, counter: u32, capacity: u32, pixels: u64, limit_bytes: u64) {
        if counter <= capacity {
            return;
        }
        self.overflow_frames = self.overflow_frames.saturating_add(1);
        if self.rebuild {
            return;
        }
        let next = self
            .factor
            .saturating_mul(2)
            .min(max_factor(pixels, limit_bytes));
        if next > self.factor {
            self.factor = next;
            self.rebuild = true;
        }
    }
}

/// State of one staging buffer of the ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    Free,
    /// A copy with this sequence number was recorded; mapped by the next `poll`.
    Written(u64),
    /// map_async is pending for this sequence number.
    Mapping(u64),
}

/// Device-free ring and sequence bookkeeping of the counter readback.
#[derive(Debug)]
pub(crate) struct Readback {
    slots: [Slot; STAGING_RING],
    /// Next slot tried by `reserve` (round robin).
    cursor: usize,
    /// Sequence number of the next reservation (starts at 1).
    next_seq: u64,
    /// Sequence number of the newest stored counter (0 = none yet).
    applied_seq: u64,
    newest: Option<u32>,
}

impl Readback {
    pub(crate) fn new() -> Readback {
        Readback {
            slots: [Slot::Free; STAGING_RING],
            cursor: 0,
            next_seq: 1,
            applied_seq: 0,
            newest: None,
        }
    }

    /// Reserves the next free slot for a copy and gives it the next sequence number; None
    /// when every slot is busy (the copy is skipped).
    pub(crate) fn reserve(&mut self) -> Option<usize> {
        for k in 0..STAGING_RING {
            let i = (self.cursor + k) % STAGING_RING;
            if self.slots[i] == Slot::Free {
                self.slots[i] = Slot::Written(self.next_seq);
                self.next_seq += 1;
                self.cursor = (i + 1) % STAGING_RING;
                return Some(i);
            }
        }
        None
    }

    /// Marks every slot written since the last call as mapping; entry i is the sequence
    /// number to map slot i with.
    pub(crate) fn start_mapping(&mut self) -> [Option<u64>; STAGING_RING] {
        let mut out = [None; STAGING_RING];
        for (slot, o) in self.slots.iter_mut().zip(out.iter_mut()) {
            if let Slot::Written(seq) = *slot {
                *slot = Slot::Mapping(seq);
                *o = Some(seq);
            }
        }
        out
    }

    /// Map callback of `slot` for `seq`: frees the slot and stores `value` (None = map
    /// failed) unless a newer counter was already stored.
    pub(crate) fn complete(&mut self, slot: usize, seq: u64, value: Option<u32>) {
        if let Some(s) = self.slots.get_mut(slot)
            && *s == Slot::Mapping(seq)
        {
            *s = Slot::Free;
        }
        if let Some(v) = value
            && seq > self.applied_seq
        {
            self.applied_seq = seq;
            self.newest = Some(v);
        }
    }

    /// The newest completed counter, once.
    pub(crate) fn take(&mut self) -> Option<u32> {
        self.newest.take()
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Decodes one little-endian u32.
fn decode_counter(bytes: &[u8]) -> Option<u32> {
    let b: [u8; 4] = bytes.get(..4)?.try_into().ok()?;
    Some(u32::from_le_bytes(b))
}

/// (a, b, c, 0) as 16 little-endian bytes.
fn uvec4_bytes(a: u32, b: u32, c: u32) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (chunk, v) in out.chunks_exact_mut(4).zip([a, b, c, 0]) {
        chunk.copy_from_slice(&v.to_le_bytes());
    }
    out
}

fn buffer_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::BufferBindingType,
    size: u64,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(size),
        },
        count: None,
    }
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn storage(read_only: bool) -> wgpu::BufferBindingType {
    wgpu::BufferBindingType::Storage { read_only }
}

fn layout(
    gpu: &GpuContext,
    label: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> wgpu::BindGroupLayout {
    gpu.device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries,
        })
}

/// Bind group layouts, created once and shared by every generation of buffers (the pass
/// builds its pipelines on them).
#[derive(Clone, Debug)]
pub(crate) struct OitLayouts {
    /// Group 2 of `PASS_TRANSPARENT_WGSL` (FRAGMENT): 0 heads, 1 nodes, 2 counter (storage
    /// rw), 3 OitParams uniform.
    pub(crate) transparent: wgpu::BindGroupLayout,
    /// Group 0 of `CLEAR_WGSL` (COMPUTE): 0 heads, 1 counter (storage rw), 2 params uniform.
    pub(crate) clear: wgpu::BindGroupLayout,
    /// Group 0 of `RESOLVE_WGSL` (FRAGMENT): 0 heads, 1 nodes (storage read), 2 hdr.src
    /// (textureLoad), 3 params uniform.
    pub(crate) resolve: wgpu::BindGroupLayout,
}

impl OitLayouts {
    pub(crate) fn new(gpu: &GpuContext) -> OitLayouts {
        let fragment = wgpu::ShaderStages::FRAGMENT;
        let compute = wgpu::ShaderStages::COMPUTE;
        let uniform = wgpu::BufferBindingType::Uniform;
        let transparent = layout(
            gpu,
            "raa-oit.transparent",
            &[
                buffer_entry(0, fragment, storage(false), 4),
                buffer_entry(1, fragment, storage(false), NODE_SIZE as u64),
                buffer_entry(2, fragment, storage(false), COUNTER_SIZE),
                buffer_entry(3, fragment, uniform, OIT_PARAMS_SIZE),
            ],
        );
        let clear = layout(
            gpu,
            "raa-oit.clear",
            &[
                buffer_entry(0, compute, storage(false), 4),
                buffer_entry(1, compute, storage(false), COUNTER_SIZE),
                buffer_entry(2, compute, uniform, CLEAR_PARAMS_SIZE),
            ],
        );
        let resolve = layout(
            gpu,
            "raa-oit.resolve",
            &[
                buffer_entry(0, fragment, storage(true), 4),
                buffer_entry(1, fragment, storage(true), NODE_SIZE as u64),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: fragment,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                buffer_entry(3, fragment, uniform, RESOLVE_PARAMS_SIZE),
            ],
        );
        OitLayouts {
            transparent,
            clear,
            resolve,
        }
    }
}

/// Heads, node pool, counter, uniforms, bind groups and the counter readback ring of one
/// surface size and pool factor.
pub(crate) struct OitBuffers {
    layouts: OitLayouts,
    /// Kept to poll the map callbacks on native.
    device: wgpu::Device,
    size: (u32, u32),
    /// Nodes in the pool (`pool_nodes`).
    capacity: u32,
    /// min(max_storage_buffer_binding_size, max_buffer_size).
    limit_bytes: u64,
    growth: Growth,
    max_fragments: u32,
    /// w x h u32, STORAGE.
    heads: wgpu::Buffer,
    /// capacity x 16 B, STORAGE.
    nodes: wgpu::Buffer,
    /// One atomic u32, STORAGE | COPY_SRC.
    counter: wgpu::Buffer,
    /// OitParams (width, height, capacity, 0).
    params: wgpu::Buffer,
    /// Clear params (count = w x h, 0, 0, 0).
    clear_params: wgpu::Buffer,
    /// Resolve params (width, height, max_fragments, 0).
    resolve_params: wgpu::Buffer,
    transparent_group: wgpu::BindGroup,
    clear_group: wgpu::BindGroup,
    clear_groups: (u32, u32),
    /// MAP_READ | COPY_DST, 4 B each.
    staging: [wgpu::Buffer; STAGING_RING],
    readback: Arc<Mutex<Readback>>,
}

impl OitBuffers {
    /// Creates the layouts and the buffers for a `width` x `height` surface (each at least 1)
    /// with `factor` nodes per pixel (at least 1), bounded by the adapter limit.
    pub(crate) fn create(gpu: &GpuContext, width: u32, height: u32, factor: u32) -> OitBuffers {
        let max_fragments = crate::resolve_cpu::DEFAULT_MAX_FRAGMENTS as u32;
        Self::build(
            gpu,
            OitLayouts::new(gpu),
            (width, height),
            Growth::new(factor),
            max_fragments,
        )
    }

    /// New buffers for `width` x `height` with the same layouts, the factor in use, the
    /// overflow count and max_fragments (on resize, or when `needs_rebuild`).
    pub(crate) fn rebuild(&self, gpu: &GpuContext, width: u32, height: u32) -> OitBuffers {
        let growth = Growth {
            rebuild: false,
            ..self.growth
        };
        Self::build(
            gpu,
            self.layouts.clone(),
            (width, height),
            growth,
            self.max_fragments,
        )
    }

    fn build(
        gpu: &GpuContext,
        layouts: OitLayouts,
        size: (u32, u32),
        growth: Growth,
        max_fragments: u32,
    ) -> OitBuffers {
        let (w, h) = (size.0.max(1), size.1.max(1));
        let device = gpu.device();
        let limits = gpu.limits();
        let limit_bytes = pool_limit_bytes(&limits);
        let capacity = pool_nodes(w, h, growth.factor, limit_bytes);
        let pixels = w.saturating_mul(h);
        let storage = wgpu::BufferUsages::STORAGE;
        let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let heads = buffer("raa-oit.heads", u64::from(pixels) * 4, storage);
        let nodes = buffer(
            "raa-oit.nodes",
            u64::from(capacity) * NODE_SIZE as u64,
            storage,
        );
        let counter = buffer(
            "raa-oit.counter",
            COUNTER_SIZE,
            storage | wgpu::BufferUsages::COPY_SRC,
        );
        let params = buffer("raa-oit.params", OIT_PARAMS_SIZE, uniform);
        let clear_params = buffer("raa-oit.clear_params", CLEAR_PARAMS_SIZE, uniform);
        let resolve_params = buffer("raa-oit.resolve_params", RESOLVE_PARAMS_SIZE, uniform);
        let queue = gpu.queue();
        queue.write_buffer(&params, 0, &uvec4_bytes(w, h, capacity));
        queue.write_buffer(&clear_params, 0, &uvec4_bytes(pixels, 0, 0));
        queue.write_buffer(&resolve_params, 0, &uvec4_bytes(w, h, max_fragments));
        let staging = std::array::from_fn(|_| {
            buffer(
                "raa-oit.counter_staging",
                COUNTER_SIZE,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            )
        });
        let transparent_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-oit.transparent"),
            layout: &layouts.transparent,
            entries: &[
                entry(0, &heads),
                entry(1, &nodes),
                entry(2, &counter),
                entry(3, &params),
            ],
        });
        let clear_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-oit.clear"),
            layout: &layouts.clear,
            entries: &[
                entry(0, &heads),
                entry(1, &counter),
                entry(2, &clear_params),
            ],
        });
        let clear_groups = clear_groups(pixels, limits.max_compute_workgroups_per_dimension);
        OitBuffers {
            device: device.clone(),
            layouts,
            size: (w, h),
            capacity,
            limit_bytes,
            growth,
            max_fragments,
            heads,
            nodes,
            counter,
            params,
            clear_params,
            resolve_params,
            transparent_group,
            clear_group,
            clear_groups,
            staging,
            readback: Arc::new(Mutex::new(Readback::new())),
        }
    }

    pub(crate) fn layouts(&self) -> &OitLayouts {
        &self.layouts
    }

    /// Group 2 of the transparent draws.
    pub(crate) fn transparent_bind_group(&self) -> &wgpu::BindGroup {
        &self.transparent_group
    }

    /// Group 0 of the head clear compute.
    pub(crate) fn clear_bind_group(&self) -> &wgpu::BindGroup {
        &self.clear_group
    }

    /// Workgroups (x, y) of the head clear dispatch.
    pub(crate) fn clear_dispatch(&self) -> (u32, u32) {
        self.clear_groups
    }

    /// Group 0 of the resolve reading `hdr_src` (the caller caches one per hdr view).
    pub(crate) fn resolve_bind_group(
        &self,
        gpu: &GpuContext,
        hdr_src: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-oit.resolve"),
            layout: &self.layouts.resolve,
            entries: &[
                entry(0, &self.heads),
                entry(1, &self.nodes),
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(hdr_src),
                },
                entry(3, &self.resolve_params),
            ],
        })
    }

    /// Writes the resolve params when `max_fragments` changed.
    pub(crate) fn set_max_fragments(&mut self, gpu: &GpuContext, max_fragments: u32) {
        if max_fragments == self.max_fragments {
            return;
        }
        self.max_fragments = max_fragments;
        let (w, h) = self.size;
        gpu.queue()
            .write_buffer(&self.resolve_params, 0, &uvec4_bytes(w, h, max_fragments));
    }

    /// After the transparent draws: copies the counter into the next free staging buffer;
    /// skipped when every staging buffer is busy.
    pub(crate) fn copy_counter(&self, encoder: &mut wgpu::CommandEncoder) {
        let reserved = lock(&self.readback).reserve();
        if let Some(staging) = reserved.and_then(|i| self.staging.get(i)) {
            encoder.copy_buffer_to_buffer(&self.counter, 0, staging, 0, COUNTER_SIZE);
        }
    }

    /// Call after the encoder of `copy_counter` was submitted (the next prepare): maps the
    /// staging buffers written since the last call and returns the newest counter mapped
    /// since, without blocking (None: nothing new). A counter above the capacity counts an
    /// overflow frame and may grow the factor (`needs_rebuild`).
    pub(crate) fn poll(&mut self) -> Option<u32> {
        let mapping = lock(&self.readback).start_mapping();
        for (slot, (seq, buffer)) in mapping.into_iter().zip(&self.staging).enumerate() {
            let Some(seq) = seq else {
                continue;
            };
            let readback = Arc::clone(&self.readback);
            let mapped = buffer.clone();
            buffer.map_async(wgpu::MapMode::Read, .., move |result| {
                let value = result.ok().and_then(|()| {
                    mapped
                        .get_mapped_range(..)
                        .ok()
                        .and_then(|view| decode_counter(&view))
                });
                // Unmapping an unmapped buffer (failed map) is a no-op.
                mapped.unmap();
                lock(&readback).complete(slot, seq, value);
            });
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Non-blocking: runs the callbacks of maps that completed; an error only delays
            // the readback to a later poll.
            let _ = self.device.poll(wgpu::PollType::Poll);
        }
        let counter = lock(&self.readback).take()?;
        let pixels = u64::from(self.size.0) * u64::from(self.size.1);
        self.growth
            .observe(counter, self.capacity, pixels, self.limit_bytes);
        Some(counter)
    }

    /// The factor grew after an overflow: the pass recreates the buffers (`rebuild`) at its
    /// next prepare.
    pub(crate) fn needs_rebuild(&self) -> bool {
        self.growth.rebuild
    }

    /// Pool factor in use.
    pub(crate) fn factor(&self) -> u32 {
        self.growth.factor
    }

    /// Nodes in the pool.
    pub(crate) fn pool_nodes(&self) -> u32 {
        self.capacity
    }

    /// Counter reads that exceeded the capacity, over every generation of buffers.
    pub(crate) fn overflow_frames(&self) -> u32 {
        self.growth.overflow_frames
    }

    /// Surface size the buffers were created for (each at least 1).
    pub(crate) fn size(&self) -> (u32, u32) {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1280 x 720.
    const PIXELS: u64 = 921_600;
    /// The WebGPU default storage binding limit (128 MiB).
    const LIMIT: u64 = 128 << 20;

    #[test]
    fn overflow_doubles_the_factor_and_asks_for_a_rebuild() {
        // 1 GiB: the factor is far from the bound (72 at 720p).
        let limit = 1 << 30;
        let mut g = Growth::new(8);
        let capacity = pool_nodes(1280, 720, 8, limit);
        g.observe(capacity + 1, capacity, PIXELS, limit);
        assert_eq!(g.factor, 16);
        assert_eq!(g.overflow_frames, 1);
        assert!(g.rebuild);
    }

    #[test]
    fn overflow_while_a_rebuild_is_pending_counts_but_does_not_double_again() {
        let mut g = Growth::new(4);
        g.observe(10, 5, PIXELS, LIMIT);
        g.observe(10, 5, PIXELS, LIMIT);
        assert_eq!(g.factor, 8);
        assert_eq!(g.overflow_frames, 2);
    }

    #[test]
    fn growth_is_clamped_at_the_limit() {
        // 128 MiB / 16 B = 8_388_608 nodes = 9.1 nodes per pixel at 720p: the bound is 10.
        assert_eq!(max_factor(PIXELS, LIMIT), 10);
        let mut g = Growth::new(8);
        let capacity = pool_nodes(1280, 720, 8, LIMIT);
        g.observe(u32::MAX, capacity, PIXELS, LIMIT);
        assert_eq!(g.factor, 10);
        assert!(g.rebuild);
        assert_eq!(pool_nodes(1280, 720, g.factor, LIMIT), 8_388_608);

        // At the bound: the overflow is counted, the factor stays, no rebuild.
        let mut g = Growth {
            rebuild: false,
            ..g
        };
        let capacity = pool_nodes(1280, 720, g.factor, LIMIT);
        g.observe(u32::MAX, capacity, PIXELS, LIMIT);
        assert_eq!(g.factor, 10);
        assert_eq!(g.overflow_frames, 2);
        assert!(!g.rebuild);
    }

    #[test]
    fn no_growth_without_overflow() {
        let mut g = Growth::new(8);
        let capacity = pool_nodes(1280, 720, 8, LIMIT);
        g.observe(0, capacity, PIXELS, LIMIT);
        g.observe(capacity, capacity, PIXELS, LIMIT);
        assert_eq!(g, Growth::new(8));
    }

    #[test]
    fn doubling_saturates_instead_of_wrapping() {
        let mut g = Growth::new(u32::MAX / 2 + 1);
        g.observe(2, 1, 1, u64::MAX);
        assert_eq!(g.factor, u32::MAX);
        assert_eq!(Growth::new(0).factor, 1);
    }

    #[test]
    fn pool_limit_is_the_smaller_of_both_limits() {
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: 1 << 20,
            max_buffer_size: 1 << 30,
            ..wgpu::Limits::default()
        };
        assert_eq!(pool_limit_bytes(&limits), 1 << 20);
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: 1 << 30,
            max_buffer_size: 1 << 21,
            ..wgpu::Limits::default()
        };
        assert_eq!(pool_limit_bytes(&limits), 1 << 21);
    }

    #[test]
    fn clear_groups_cover_every_head_and_split_over_y() {
        assert_eq!(clear_groups(1280 * 720, 65_535), (14_400, 1));
        assert_eq!(clear_groups(1, 65_535), (1, 1));
        let (x, y) = clear_groups(3840 * 2160, 65_535);
        assert!(x <= 65_535);
        assert!(u64::from(x) * u64::from(y) * 64 >= 3840 * 2160);
    }

    #[test]
    fn readback_reserves_round_robin_and_skips_when_full() {
        let mut r = Readback::new();
        assert_eq!(r.reserve(), Some(0));
        assert_eq!(r.reserve(), Some(1));
        assert_eq!(r.reserve(), Some(2));
        assert_eq!(r.reserve(), None, "every staging buffer busy");
        let seqs = r.start_mapping();
        assert_eq!(seqs, [Some(1), Some(2), Some(3)]);
        r.complete(1, 2, Some(7));
        assert_eq!(r.reserve(), Some(1));
    }

    #[test]
    fn readback_keeps_only_the_newest_counter_and_gives_it_once() {
        let mut r = Readback::new();
        r.reserve();
        r.reserve();
        r.start_mapping();
        r.complete(1, 2, Some(20));
        r.complete(0, 1, Some(10));
        assert_eq!(r.take(), Some(20));
        assert_eq!(r.take(), None);
    }

    #[test]
    fn uniform_bytes_are_little_endian_with_a_zero_tail() {
        let b = uvec4_bytes(1280, 720, 7_372_800);
        assert_eq!(decode_counter(&b[0..4]), Some(1280));
        assert_eq!(decode_counter(&b[4..8]), Some(720));
        assert_eq!(decode_counter(&b[8..12]), Some(7_372_800));
        assert_eq!(&b[12..], &[0, 0, 0, 0]);
        assert_eq!(decode_counter(&[1, 2]), None);
    }
}
