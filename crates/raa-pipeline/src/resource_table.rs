//! Resource table of the render graph (S55): images and buffers each pass publishes under
//! (pass entry, port key), and the inputs a graph binds to a pass. Allocation-free after
//! warm-up: `clear` resets live counts only and keeps every slot and its String capacity.

use std::cell::RefCell;

/// Dimension of a published image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageDim {
    D2,
    D2Array,
    D3,
    Cube,
}

/// How a shader samples a published image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleKind {
    Float,
    UnfilterableFloat,
    Depth,
    Uint,
    Sint,
}

/// Description of a published image; `layers` is the array layer count for D2Array, 6 for
/// Cube, the depth for D3 and 1 for D2.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageDesc {
    pub dim: ImageDim,
    pub format: wgpu::TextureFormat,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub mips: u32,
    pub sample: SampleKind,
}

impl ImageDesc {
    /// A description whose `sample` is `sample_kind_for(format)`.
    pub fn new(
        dim: ImageDim,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        layers: u32,
        mips: u32,
    ) -> ImageDesc {
        let sample = sample_kind_for(format);
        ImageDesc {
            dim,
            format,
            width,
            height,
            layers,
            mips,
            sample,
        }
    }
}

/// The sample kind of a texture format: depth formats Depth, integer formats Uint or Sint,
/// 32-bit float formats UnfilterableFloat, everything else Float.
pub fn sample_kind_for(format: wgpu::TextureFormat) -> SampleKind {
    use wgpu::TextureFormat as F;
    match format {
        F::R32Float | F::Rg32Float | F::Rgba32Float => return SampleKind::UnfilterableFloat,
        _ => {}
    }
    if format.is_depth_stencil_format() {
        return SampleKind::Depth;
    }
    match format.sample_type(None, None) {
        Some(wgpu::TextureSampleType::Uint) => SampleKind::Uint,
        Some(wgpu::TextureSampleType::Sint) => SampleKind::Sint,
        Some(wgpu::TextureSampleType::Depth) => SampleKind::Depth,
        _ => SampleKind::Float,
    }
}

/// A published image: a view (reference-counted handle clone) and its description.
#[derive(Clone, Debug)]
pub struct PublishedImage {
    pub view: wgpu::TextureView,
    pub desc: ImageDesc,
}

enum Payload {
    Image(PublishedImage),
    Buffer(wgpu::Buffer, u64),
}

struct Slot<T> {
    pass: String,
    port: String,
    value: T,
}

struct Slots<T> {
    slots: Vec<Slot<T>>,
    live: usize,
}

impl<T> Slots<T> {
    const fn new() -> Self {
        Slots {
            slots: Vec::new(),
            live: 0,
        }
    }

    fn live(&self) -> &[Slot<T>] {
        &self.slots[..self.live]
    }

    fn find(&self, pass: &str, port: &str) -> Option<&T> {
        let slot = self
            .live()
            .iter()
            .find(|s| s.pass == pass && s.port == port)?;
        Some(&slot.value)
    }

    /// Replaces the value of a live key in place, else reuses the next free slot.
    fn put(&mut self, pass: &str, port: &str, value: T) {
        let live = self.live;
        if let Some(slot) = self.slots[..live]
            .iter_mut()
            .find(|s| s.pass == pass && s.port == port)
        {
            slot.value = value;
            return;
        }
        if let Some(slot) = self.slots.get_mut(live) {
            slot.pass.clear();
            slot.pass.push_str(pass);
            slot.port.clear();
            slot.port.push_str(port);
            slot.value = value;
        } else {
            self.slots.push(Slot {
                pass: pass.to_string(),
                port: port.to_string(),
                value,
            });
        }
        self.live += 1;
    }
}

struct Inner {
    published: Slots<Payload>,
    inputs: Slots<PublishedImage>,
}

/// Registry of the images and buffers published this frame, keyed by (pass, port), and of
/// the inputs bound to passes (a separate key space). Interior mutability: every method
/// takes `&self`.
pub struct ResourceTable {
    inner: RefCell<Inner>,
}

impl Default for ResourceTable {
    fn default() -> Self {
        ResourceTable::new()
    }
}

impl ResourceTable {
    /// An empty table.
    pub fn new() -> ResourceTable {
        let inner = Inner {
            published: Slots::new(),
            inputs: Slots::new(),
        };
        ResourceTable {
            inner: RefCell::new(inner),
        }
    }

    /// Publishes an image; a second publish under one key replaces the first in place.
    pub fn publish(&self, pass: &str, port: &str, view: &wgpu::TextureView, desc: ImageDesc) {
        let image = PublishedImage {
            view: view.clone(),
            desc,
        };
        self.inner
            .borrow_mut()
            .published
            .put(pass, port, Payload::Image(image));
    }

    /// Publishes a buffer of `size` bytes under the same key space as images.
    pub fn publish_buffer(&self, pass: &str, port: &str, buffer: &wgpu::Buffer, size: u64) {
        let payload = Payload::Buffer(buffer.clone(), size);
        self.inner.borrow_mut().published.put(pass, port, payload);
    }

    /// The image published under (pass, port); None for a buffer or an unknown key.
    pub fn get(&self, pass: &str, port: &str) -> Option<PublishedImage> {
        match self.inner.borrow().published.find(pass, port)? {
            Payload::Image(image) => Some(image.clone()),
            Payload::Buffer(..) => None,
        }
    }

    /// The buffer and its size in bytes published under (pass, port); None for an image.
    pub fn buffer(&self, pass: &str, port: &str) -> Option<(wgpu::Buffer, u64)> {
        match self.inner.borrow().published.find(pass, port)? {
            Payload::Buffer(buffer, size) => Some((buffer.clone(), *size)),
            Payload::Image(_) => None,
        }
    }

    /// Binds an input image to (pass, port); replaces an earlier binding in place.
    pub fn bind_input(&self, pass: &str, port: &str, image: PublishedImage) {
        self.inner.borrow_mut().inputs.put(pass, port, image);
    }

    /// The input bound to (pass, port).
    pub fn input(&self, pass: &str, port: &str) -> Option<PublishedImage> {
        self.inner.borrow().inputs.find(pass, port).cloned()
    }

    /// The published images (pass, port, desc) in insertion order.
    pub fn entries(&self) -> Vec<(String, String, ImageDesc)> {
        let inner = self.inner.borrow();
        let slots = inner.published.live().iter();
        slots
            .filter_map(|s| match &s.value {
                Payload::Image(image) => Some((s.pass.clone(), s.port.to_string(), image.desc)),
                Payload::Buffer(..) => None,
            })
            .collect()
    }

    /// The published buffers (pass, port, size in bytes) in insertion order.
    pub fn buffer_entries(&self) -> Vec<(String, String, u64)> {
        let inner = self.inner.borrow();
        let slots = inner.published.live().iter();
        slots
            .filter_map(|s| match &s.value {
                Payload::Buffer(_, size) => Some((s.pass.clone(), s.port.to_string(), *size)),
                Payload::Image(_) => None,
            })
            .collect()
    }

    /// Empties images, buffers and inputs; keeps the slots for reuse.
    pub fn clear(&self) {
        let mut inner = self.inner.borrow_mut();
        inner.published.live = 0;
        inner.inputs.live = 0;
    }
}
