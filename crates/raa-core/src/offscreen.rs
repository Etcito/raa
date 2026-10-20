//! Offscreen render targets with RGBA8 readback (native), and the `non_empty` pixel check.

#[cfg(not(target_arch = "wasm32"))]
use crate::{GpuContext, RaaError};

/// True when any pixel differs from `clear_rgba8` by more than 2 in any channel.
/// A trailing partial pixel (fewer than 4 bytes) is ignored.
pub fn non_empty(pixels_rgba8: &[u8], clear_rgba8: [u8; 4]) -> bool {
    pixels_rgba8
        .chunks_exact(4)
        .any(|px| px.iter().zip(clear_rgba8).any(|(&a, b)| a.abs_diff(b) > 2))
}

/// A 2D texture that can be rendered into and read back on the CPU.
#[cfg(not(target_arch = "wasm32"))]
pub struct OffscreenTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
}

#[cfg(not(target_arch = "wasm32"))]
impl OffscreenTarget {
    /// Creates a `width` x `height` target (zero sizes clamp to 1) with usage
    /// RENDER_ATTACHMENT | COPY_SRC.
    pub fn new(
        gpu: &GpuContext,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> OffscreenTarget {
        let (width, height) = (width.max(1), height.max(1));
        let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("raa-offscreen"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        OffscreenTarget {
            texture,
            view,
            width,
            height,
            format,
        }
    }

    /// The view to render into.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// The underlying texture.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// `(width, height)` in pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// The texture format.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// Copies the texture into a MAP_READ buffer (rows padded to 256 bytes), waits for the
    /// mapping, and returns tightly packed RGBA8 rows, top row first. Only Rgba8Unorm and
    /// Rgba8UnormSrgb are supported (`core.unsupported` otherwise); a failed mapping or
    /// device poll gives `core.gpu`.
    pub fn read_rgba8(&self, gpu: &GpuContext) -> Result<Vec<u8>, RaaError> {
        match self.format {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {}
            other => {
                return Err(RaaError::Unsupported(format!(
                    "read_rgba8 supports Rgba8Unorm and Rgba8UnormSrgb, not {other:?}"
                )));
            }
        }
        let row_bytes = self.width as usize * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let padded_row = row_bytes.div_ceil(align) * align;
        let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-offscreen-readback"),
            size: (padded_row * self.height as usize) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("raa-offscreen-readback"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row as u32),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue().submit([encoder.finish()]);

        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            // The receiver outlives the poll below; a send error cannot be acted upon here.
            let _ = tx.send(result);
        });
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| RaaError::Gpu(format!("device poll failed: {e}")))?;
        rx.recv()
            .map_err(|_| RaaError::Gpu("buffer map callback was never called".to_string()))?
            .map_err(|e| RaaError::Gpu(format!("buffer map failed: {e}")))?;

        let mut pixels = Vec::with_capacity(row_bytes * self.height as usize);
        {
            let mapped = slice
                .get_mapped_range()
                .map_err(|e| RaaError::Gpu(format!("mapped range unavailable: {e}")))?;
            for row in mapped.chunks_exact(padded_row) {
                pixels.extend_from_slice(&row[..row_bytes]);
            }
        }
        buffer.unmap();
        Ok(pixels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLEAR: [u8; 4] = [13, 15, 20, 255];

    #[test]
    fn equal_pixels_are_empty() {
        let pixels: Vec<u8> = CLEAR.repeat(16);
        assert!(!non_empty(&pixels, CLEAR));
    }

    #[test]
    fn a_diff_of_two_is_still_empty() {
        let mut pixels: Vec<u8> = CLEAR.repeat(16);
        pixels[5] += 2;
        pixels[8] -= 2;
        assert!(!non_empty(&pixels, CLEAR));
    }

    #[test]
    fn a_diff_of_three_is_non_empty() {
        let mut pixels: Vec<u8> = CLEAR.repeat(16);
        pixels[62] += 3;
        assert!(non_empty(&pixels, CLEAR));
        let mut below: Vec<u8> = CLEAR.repeat(16);
        below[0] -= 3;
        assert!(non_empty(&below, CLEAR));
    }

    #[test]
    fn no_pixels_are_empty() {
        assert!(!non_empty(&[], CLEAR));
    }
}
