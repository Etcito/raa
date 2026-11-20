//! Billboard bake targets (M6): the eight-direction tile layout and the private GPU targets
//! a billboard bake renders into before it publishes three images (diffuse, normal,
//! opacity) to the material atlas.
//!
//! Layout ([`billboard_layout`], the sole packing authority, copied unchanged into bake
//! metadata): resolutions 64, 128 and 256 use eight columns and one row, 512 uses four
//! columns and two rows. The tile pitch is `resolution + 2` (a one-texel gutter on every
//! side); tile `k` sits at column `k % columns`, row `k / columns`, and its content rect is
//! `[(k % columns) * pitch + 1, (k / columns) * pitch + 1, resolution, resolution]`
//! (x, y, width, height in texels). Consumers read the tiles and extent from the layout
//! and never infer a column count.
//!
//! GPU path ([`BillboardTargets`]): three resolution-square rgba8unorm colour attachments
//! and one Depth32Float attachment are reused for each direction; `encode_tile` copies each
//! colour attachment into its direction's content rect of a combined image of
//! `layout.extent` and duplicates the edge texels into the gutters, all with GPU copies.
//! `publish` copies the three combined images into the atlas with
//! `TextureAtlas::add_gpu_image`. No texel is generated, read back or uploaded by the CPU.

use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use raa_core::{GpuContext, RaaError};

use crate::textures::{ATLAS_FORMAT, ShelfPacker, TextureAtlas};

/// Number of baked directions (and tiles).
const DIRECTIONS: usize = 8;
/// Every encoded direction bit set.
const ALL_DIRECTIONS: u8 = 0xFF;
/// Format of the depth attachment.
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// The tile layout of one billboard bake at `resolution`: the content rect of each
/// direction (x, y, width, height in texels, direction rank order) and the combined image
/// extent (width, height), gutters included.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BillboardLayout {
    pub resolution: u32,
    pub tiles: [[u32; 4]; 8],
    pub extent: [u32; 2],
}

fn billboard_error(code: &'static str, message: String) -> RaaError {
    RaaError::FrameGraph { code, message }
}

/// The layout of a `resolution` bake (see the module documentation). Refuses
/// `billboard.invalid` for a resolution other than 64, 128, 256 or 512.
pub fn billboard_layout(resolution: u32) -> Result<BillboardLayout, RaaError> {
    if !matches!(resolution, 64 | 128 | 256 | 512) {
        return Err(billboard_error(
            "billboard.invalid",
            format!("billboard resolution {resolution} must be 64, 128, 256 or 512"),
        ));
    }
    let columns: u32 = if resolution == 512 { 4 } else { 8 };
    let pitch = resolution + 2;
    let mut tiles = [[0; 4]; DIRECTIONS];
    for (k, tile) in (0u32..).zip(tiles.iter_mut()) {
        *tile = [
            (k % columns) * pitch + 1,
            (k / columns) * pitch + 1,
            resolution,
            resolution,
        ];
    }
    Ok(BillboardLayout {
        resolution,
        tiles,
        extent: [columns * pitch, (DIRECTIONS as u32 / columns) * pitch],
    })
}

/// The content rect of `direction` (0..7) in a `resolution` bake: `billboard_layout(
/// resolution)?.tiles[direction]`. Refuses `billboard.invalid` for an invalid resolution
/// or a direction above 7.
pub fn billboard_tile_rect(resolution: u32, direction: u8) -> Result<[u32; 4], RaaError> {
    let layout = billboard_layout(resolution)?;
    layout
        .tiles
        .get(usize::from(direction))
        .copied()
        .ok_or_else(|| {
            billboard_error(
                "billboard.invalid",
                format!("billboard direction {direction} must be within 0..=7"),
            )
        })
}

/// Reserves the three images (diffuse, normal, opacity) of one `extent` publication on
/// `packer`, all or nothing: when one reservation fails, the ones made before it are
/// freed and `billboard.bake_failed` is returned, so `packer` is unchanged.
pub(crate) fn reserve_images(
    packer: &mut ShelfPacker,
    extent: [u32; 2],
) -> Result<[u32; 3], RaaError> {
    let mut ids = [0; 3];
    for i in 0..ids.len() {
        match packer.alloc(extent[0], extent[1]) {
            Ok((id, _)) => ids[i] = id,
            Err(e) => {
                for &id in &ids[..i] {
                    packer.free(id);
                }
                return Err(billboard_error(
                    "billboard.bake_failed",
                    format!("billboard atlas images do not fit: {e}"),
                ));
            }
        }
    }
    Ok(ids)
}

/// The reusable per-direction attachments of [`BillboardTargets::attachments`].
#[allow(
    dead_code,
    reason = "read by the billboard bake renderer in the next share"
)]
pub(crate) struct BillboardAttachments<'a> {
    pub diffuse: &'a wgpu::TextureView,
    pub normal: &'a wgpu::TextureView,
    pub opacity: &'a wgpu::TextureView,
    pub depth: &'a wgpu::TextureView,
}

/// The atlas ids of a successful [`BillboardTargets::publish`] and the layout it was made
/// with; the caller owns the ids and removes them with `TextureAtlas::remove`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct BillboardAtlasImages {
    pub diffuse: u32,
    pub normal: u32,
    pub opacity: u32,
    pub tiles: [[u32; 4]; 8],
    pub extent: [u32; 2],
}

/// The temporary GPU textures of one bake.
struct TargetResources {
    /// Diffuse, normal, opacity attachments (resolution square).
    color: [wgpu::Texture; 3],
    color_views: [wgpu::TextureView; 3],
    depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    /// Diffuse, normal, opacity combined images (`layout.extent`).
    combined: [wgpu::Texture; 3],
}

impl TargetResources {
    fn destroy(self) {
        for t in self.color.iter().chain(self.combined.iter()) {
            t.destroy();
        }
        self.depth.destroy();
    }
}

/// One GPU copy of [`gutter_copies`]: source origin in the attachment, destination origin
/// in the combined image, size (all in texels).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct TileCopy {
    pub src: [u32; 2],
    pub dst: [u32; 2],
    pub size: [u32; 2],
}

/// The nine copies that place a `tile[2]` x `tile[3]` attachment at content rect `tile`:
/// the content, then its edge columns and rows into the left, right, top and bottom
/// gutters, then its four corner texels into the gutter corners.
pub(crate) fn gutter_copies(tile: [u32; 4]) -> [TileCopy; 9] {
    let [x, y, w, h] = tile;
    let (r, b) = (w - 1, h - 1);
    let c = |src: [u32; 2], dst: [u32; 2], size: [u32; 2]| TileCopy { src, dst, size };
    [
        c([0, 0], [x, y], [w, h]),
        c([0, 0], [x - 1, y], [1, h]),
        c([r, 0], [x + w, y], [1, h]),
        c([0, 0], [x, y - 1], [w, 1]),
        c([0, b], [x, y + h], [w, 1]),
        c([0, 0], [x - 1, y - 1], [1, 1]),
        c([r, 0], [x + w, y - 1], [1, 1]),
        c([0, b], [x - 1, y + h], [1, 1]),
        c([r, b], [x + w, y + h], [1, 1]),
    ]
}

#[derive(Default)]
struct WorkState {
    done: bool,
    waker: Option<Waker>,
}

/// Resolves once the work submitted on the queue before its creation has completed.
struct WorkDone(Arc<Mutex<WorkState>>);

impl Future for WorkDone {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if state.done {
            Poll::Ready(())
        } else {
            state.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

/// A future for the completion of the work submitted on `gpu` so far. On WebGPU the
/// browser resolves it from its event loop; natively (no event loop) the device is polled
/// until the work is done before the future is returned.
fn submitted_work_done(gpu: &GpuContext) -> WorkDone {
    let state = Arc::new(Mutex::new(WorkState::default()));
    let signal = Arc::clone(&state);
    gpu.queue().on_submitted_work_done(move || {
        let mut s = signal.lock().unwrap_or_else(|e| e.into_inner());
        s.done = true;
        if let Some(waker) = s.waker.take() {
            waker.wake();
        }
    });
    #[cfg(not(target_arch = "wasm32"))]
    {
        // A lost device completes nothing more; the callback still fires on teardown and
        // the caller checks `ensure_live` after the wait.
        let _ = gpu.device().poll(wgpu::PollType::wait_indefinitely());
    }
    WorkDone(state)
}

/// The private offscreen targets of one billboard bake at one resolution (see the module
/// documentation): per-direction attachments, three combined images and the set of
/// directions encoded so far. It owns no atlas id: a successful
/// [`BillboardTargets::publish`] hands its three ids to the caller.
pub struct BillboardTargets {
    layout: BillboardLayout,
    /// None once released.
    resources: Option<TargetResources>,
    /// Bit `k` set once direction `k` is encoded.
    encoded: Cell<u8>,
    published: bool,
}

fn color_target(
    gpu: &GpuContext,
    label: &str,
    size: [u32; 2],
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: ATLAS_FORMAT,
        usage,
        view_formats: &[],
    })
}

#[allow(
    dead_code,
    reason = "called by the billboard bake renderer in the next share"
)]
impl BillboardTargets {
    /// Creates the temporary targets of a `resolution` bake: three resolution-square
    /// rgba8unorm attachments (render attachment and copy source; diffuse, normal,
    /// opacity), one resolution-square Depth32Float attachment and three combined images of
    /// `layout.extent` (copy destination and copy source). Reserves no atlas id. Refuses
    /// `billboard.invalid` (resolution, see [`billboard_layout`]) and `core.gpu` (lost
    /// device).
    ///
    /// Handoff: 'Render eight directions into the bake targets' takes over the GPU bake
    /// through `BillboardTargets::new`, `BillboardTargets::attachments`,
    /// `BillboardTargets::encode_tile`, `BillboardTargets::publish` and
    /// `BillboardTargets::release` (contract m6-bake-targets-api). Today they create the
    /// targets, lend the reusable views, copy one rendered direction with its gutters into
    /// the combined images, copy the three images into the atlas with
    /// `TextureAtlas::add_gpu_image` (all-or-nothing, `billboard.bake_failed`) and destroy
    /// the temporary textures; nothing in the crate calls them yet, and the caller binds
    /// group 0 again when `TextureAtlas::is_allocated` turned true during publish.
    /// LIFECYCLE: new creates temporary resources; encode_tile follows each direction's
    /// render; publish completes before release; the returned atlas ids live until the bake
    /// owner removes them.
    pub(crate) fn new(gpu: &GpuContext, resolution: u32) -> Result<Self, RaaError> {
        let layout = billboard_layout(resolution)?;
        gpu.ensure_live()?;
        let square = [resolution, resolution];
        let attachment = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC;
        let image = wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC;
        let color = ["diffuse", "normal", "opacity"]
            .map(|n| color_target(gpu, &format!("raa.lit.billboard.{n}"), square, attachment));
        let combined = ["diffuse", "normal", "opacity"].map(|n| {
            color_target(
                gpu,
                &format!("raa.lit.billboard.{n}_image"),
                layout.extent,
                image,
            )
        });
        let depth = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("raa.lit.billboard.depth"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let color_views = [0, 1, 2].map(|i: usize| color[i].create_view(&Default::default()));
        let depth_view = depth.create_view(&Default::default());
        Ok(BillboardTargets {
            layout,
            resources: Some(TargetResources {
                color,
                color_views,
                depth,
                depth_view,
                combined,
            }),
            encoded: Cell::new(0),
            published: false,
        })
    }

    /// The layout of this bake (copied unchanged into the published images).
    pub(crate) fn layout(&self) -> &BillboardLayout {
        &self.layout
    }

    /// The reusable per-direction views (diffuse, normal, opacity, depth). The renderer
    /// clears and renders them for one direction, then calls
    /// [`BillboardTargets::encode_tile`]. Panics after [`BillboardTargets::release`].
    pub(crate) fn attachments(&self) -> BillboardAttachments<'_> {
        let r = self
            .resources
            .as_ref()
            .expect("BillboardTargets::attachments after release");
        BillboardAttachments {
            diffuse: &r.color_views[0],
            normal: &r.color_views[1],
            opacity: &r.color_views[2],
            depth: &r.depth_view,
        }
    }

    /// Records on `encoder` the GPU copies of the three colour attachments into
    /// `direction`'s content rect of the combined images, with the edge texels duplicated
    /// into the one-texel gutters ([`gutter_copies`]). Other directions' tiles are not
    /// touched. Refuses `billboard.invalid` (direction above 7, direction already encoded,
    /// targets released); nothing is recorded then.
    pub(crate) fn encode_tile(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        direction: u8,
    ) -> Result<(), RaaError> {
        let invalid = |m: String| billboard_error("billboard.invalid", m);
        let Some(r) = self.resources.as_ref() else {
            return Err(invalid("billboard targets are released".into()));
        };
        let Some(&tile) = self.layout.tiles.get(usize::from(direction)) else {
            return Err(invalid(format!(
                "billboard direction {direction} must be within 0..=7"
            )));
        };
        let bit = 1u8 << direction;
        if self.encoded.get() & bit != 0 {
            return Err(invalid(format!(
                "billboard direction {direction} is already encoded"
            )));
        }
        for (src, dst) in r.color.iter().zip(r.combined.iter()) {
            for copy in gutter_copies(tile) {
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: src,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: copy.src[0],
                            y: copy.src[1],
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: dst,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: copy.dst[0],
                            y: copy.dst[1],
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: copy.size[0],
                        height: copy.size[1],
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
        self.encoded.set(self.encoded.get() | bit);
        Ok(())
    }

    /// Publishes the three combined images to `atlas` with `TextureAtlas::add_gpu_image`
    /// (diffuse, normal, opacity), awaits the submitted work and the GPU validation scope,
    /// and returns their ids with this layout. Refuses `billboard.invalid` when a direction
    /// is not encoded, after a successful publish or after release (nothing changes).
    /// Refuses `billboard.bake_failed` when the three images do not all fit (checked on a
    /// copy of the packing first, so nothing is reserved), when an add or the GPU fails, or
    /// when the device is lost: every id this call allocated is removed with
    /// `TextureAtlas::remove` and no partial result escapes. On success the caller owns the
    /// three ids.
    pub(crate) async fn publish(
        &mut self,
        gpu: &GpuContext,
        atlas: &mut TextureAtlas,
    ) -> Result<BillboardAtlasImages, RaaError> {
        let invalid = |m: String| billboard_error("billboard.invalid", m);
        let failed = |m: String| billboard_error("billboard.bake_failed", m);
        if self.published {
            return Err(invalid("billboard targets are already published".into()));
        }
        let Some(r) = self.resources.as_ref() else {
            return Err(invalid("billboard targets are released".into()));
        };
        let encoded = self.encoded.get();
        if encoded != ALL_DIRECTIONS {
            let missing: Vec<u8> = (0..DIRECTIONS as u8)
                .filter(|k| encoded & (1 << k) == 0)
                .collect();
            return Err(invalid(format!(
                "billboard directions {missing:?} are not encoded"
            )));
        }
        gpu.ensure_live().map_err(|e| failed(e.to_string()))?;
        let extent = self.layout.extent;
        reserve_images(&mut atlas.packer().clone(), extent)?;
        let scope = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let mut ids = Vec::with_capacity(3);
        let mut error = None;
        for image in &r.combined {
            match atlas.add_gpu_image(gpu, image, extent) {
                Ok(id) => ids.push(id),
                Err(e) => {
                    error = Some(e.to_string());
                    break;
                }
            }
        }
        submitted_work_done(gpu).await;
        if let Some(e) = scope.pop().await {
            error.get_or_insert(format!("GPU validation: {e}"));
        }
        if let Err(e) = gpu.ensure_live() {
            error.get_or_insert(e.to_string());
        }
        if let Some(e) = error {
            for &id in &ids {
                atlas.remove(gpu, id);
            }
            return Err(failed(format!("billboard publication failed: {e}")));
        }
        self.published = true;
        Ok(BillboardAtlasImages {
            diffuse: ids[0],
            normal: ids[1],
            opacity: ids[2],
            tiles: self.layout.tiles,
            extent,
        })
    }

    /// Destroys the temporary textures; a second call does nothing. Published atlas ids
    /// are never removed. WebGPU and wgpu defer the destruction of a texture used by
    /// submitted work until that work completes, so the copies already submitted finish
    /// first. After release `encode_tile` and `publish` refuse `billboard.invalid`.
    pub(crate) fn release(&mut self) {
        if let Some(r) = self.resources.take() {
            r.destroy();
        }
    }
}

impl Drop for BillboardTargets {
    fn drop(&mut self) {
        if let Some(r) = self.resources.take() {
            r.destroy();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::textures::{ATLAS_SIZE, MAX_TEXTURES};
    use raa_core::testing::{block_on, skip, software_gpu};

    fn code(e: RaaError) -> &'static str {
        e.code()
    }

    #[test]
    fn gutter_copies_cover_the_padded_tile_exactly_once() {
        let tile = [5, 9, 4, 3];
        let mut hits = vec![0u32; 6 * 5];
        for c in gutter_copies(tile) {
            assert!(c.src[0] + c.size[0] <= 4 && c.src[1] + c.size[1] <= 3);
            for dy in 0..c.size[1] {
                for dx in 0..c.size[0] {
                    let (x, y) = (c.dst[0] + dx - 4, c.dst[1] + dy - 8);
                    hits[(y * 6 + x) as usize] += 1;
                }
            }
        }
        assert!(hits.iter().all(|&h| h == 1), "{hits:?}");
        let copies = gutter_copies(tile);
        // Edges duplicate the nearest content texels.
        assert_eq!(copies[2].src, [3, 0]);
        assert_eq!(copies[2].dst, [9, 9]);
        assert_eq!(copies[4].src, [0, 2]);
        assert_eq!(copies[8].dst, [9, 12]);
    }

    #[test]
    fn reserve_images_is_all_or_nothing() {
        let extent = billboard_layout(64).unwrap().extent;
        // Ids 1 and 2 only: the third reservation fails and the first two are freed.
        let mut packer = ShelfPacker::new(ATLAS_SIZE, 3);
        let err = reserve_images(&mut packer, extent).unwrap_err();
        assert_eq!(code(err), "billboard.bake_failed");
        assert!(packer.is_empty());
        let mut packer = ShelfPacker::new(ATLAS_SIZE, MAX_TEXTURES);
        let ids = reserve_images(&mut packer, extent).unwrap();
        assert_eq!(ids, [1, 2, 3]);
        assert_eq!(packer.rect(3).map(|r| [r.width, r.height]), Some(extent));
    }

    fn clear_attachments(
        gpu: &GpuContext,
        targets: &BillboardTargets,
        encoder: &mut wgpu::CommandEncoder,
        k: u8,
    ) {
        let a = targets.attachments();
        let value = f64::from(k) / 7.0;
        let color = |view| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: value,
                        g: 0.5,
                        b: 1.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })
        };
        let _ = gpu;
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("raa.lit.billboard.test"),
            color_attachments: &[color(a.diffuse), color(a.normal), color(a.opacity)],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: a.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        drop(pass);
    }

    #[test]
    fn targets_encode_publish_and_release_on_the_gpu() {
        let test = "targets_encode_publish_and_release_on_the_gpu";
        let Some(gpu) = software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let mut atlas = TextureAtlas::new(&gpu);
        let mut targets = BillboardTargets::new(&gpu, 64).expect("targets");
        assert_eq!(*targets.layout(), billboard_layout(64).unwrap());
        assert!(atlas.packer().is_empty(), "new reserves no atlas id");
        let mut encoder = gpu.device().create_command_encoder(&Default::default());
        for k in 0..7u8 {
            clear_attachments(&gpu, &targets, &mut encoder, k);
            targets.encode_tile(&mut encoder, k).expect("encode");
        }
        assert_eq!(
            code(targets.encode_tile(&mut encoder, 3).unwrap_err()),
            "billboard.invalid"
        );
        assert_eq!(
            code(targets.encode_tile(&mut encoder, 8).unwrap_err()),
            "billboard.invalid"
        );
        gpu.queue().submit([encoder.finish()]);
        let missing = block_on(targets.publish(&gpu, &mut atlas)).unwrap_err();
        assert_eq!(code(missing), "billboard.invalid");
        assert!(atlas.packer().is_empty());

        let mut encoder = gpu.device().create_command_encoder(&Default::default());
        clear_attachments(&gpu, &targets, &mut encoder, 7);
        targets.encode_tile(&mut encoder, 7).expect("encode 7");
        gpu.queue().submit([encoder.finish()]);
        let images = block_on(targets.publish(&gpu, &mut atlas)).expect("publish");
        assert_eq!(images.tiles, targets.layout().tiles);
        assert_eq!(images.extent, targets.layout().extent);
        assert!(atlas.is_allocated());
        for id in [images.diffuse, images.normal, images.opacity] {
            let r = atlas.packer().rect(id).expect("live id");
            assert_eq!([r.width, r.height], images.extent);
        }
        let again = block_on(targets.publish(&gpu, &mut atlas)).unwrap_err();
        assert_eq!(code(again), "billboard.invalid");
        targets.release();
        targets.release();
        assert_eq!(atlas.packer().len(), 3, "release keeps the published ids");
    }

    #[test]
    fn publish_that_does_not_fit_reserves_nothing() {
        let test = "publish_that_does_not_fit_reserves_nothing";
        let Some(gpu) = software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let mut atlas = TextureAtlas::new(&gpu);
        // Fill all but 64 rows: a 64-resolution image needs 80 aligned rows.
        let filler = color_target(
            &gpu,
            "raa.lit.billboard.test_filler",
            [ATLAS_SIZE, ATLAS_SIZE - 64],
            wgpu::TextureUsages::COPY_SRC,
        );
        let filler_id = atlas
            .add_gpu_image(&gpu, &filler, [ATLAS_SIZE, ATLAS_SIZE - 64])
            .expect("filler");
        let mut targets = BillboardTargets::new(&gpu, 64).expect("targets");
        let mut encoder = gpu.device().create_command_encoder(&Default::default());
        for k in 0..8u8 {
            clear_attachments(&gpu, &targets, &mut encoder, k);
            targets.encode_tile(&mut encoder, k).expect("encode");
        }
        gpu.queue().submit([encoder.finish()]);
        let err = block_on(targets.publish(&gpu, &mut atlas)).unwrap_err();
        assert_eq!(code(err), "billboard.bake_failed");
        assert_eq!(atlas.packer().len(), 1);
        assert!(atlas.packer().rect(filler_id).is_some());
        drop(targets);
    }
}
