//! RAA patch: destroy device-created textures and buffers when wgpu drops their last handle.
//!
//! Upstream wgpu 30 leaves a dropped `GPUTexture` / `GPUBuffer` to the JS garbage collector.
//! The collector does not see GPU memory, so replaced render targets (every canvas resize,
//! every graph rebuild) stay allocated until a collection happens to run, and can exhaust
//! VRAM before it does.
//!
//! This module applies wgpu-core's lifetime rule instead. A resource created by
//! `device.createTexture` / `device.createBuffer` lives while
//! - a handle to it lives: the `Texture` / `Buffer` (and their clones), a `TextureView` of it
//!   (which holds its texture), or a `BindGroup` that references it ([`Owned`] is shared);
//! - a command encoder that may have recorded it is neither submitted nor dropped.
//!
//! When the last [`Owned`] goes away the resource is retired: destroyed at once when no
//! command encoder is open, otherwise once every encoder that was open at that moment has
//! been submitted (or dropped). `GPUTexture.destroy()` / `GPUBuffer.destroy()` let work that
//! was already submitted complete, so destroying right after the submit is safe.
//!
//! Not owned (never destroyed here): canvas textures from `getCurrentTexture` and textures
//! wrapped from foreign handles.

use alloc::{
    collections::{BTreeSet, VecDeque},
    rc::Rc,
};
use core::cell::{Cell, RefCell};

use super::webgpu_sys;

/// A texture or buffer this device created.
#[derive(Debug, Clone)]
pub(crate) enum Resource {
    Texture(webgpu_sys::GpuTexture),
    Buffer(webgpu_sys::GpuBuffer),
}

impl Resource {
    fn destroy(&self) {
        match self {
            Resource::Texture(texture) => texture.destroy(),
            Resource::Buffer(buffer) => buffer.destroy(),
        }
    }
}

/// Per-device bookkeeping: open command encoders and resources waiting for them.
#[derive(Debug, Default)]
pub(crate) struct Retirement {
    /// Identifier of the next command encoder.
    next_encoder: Cell<u64>,
    /// Encoders created and not yet submitted or dropped.
    open_encoders: RefCell<BTreeSet<u64>>,
    /// Retired resources with the identifier of the next encoder at retirement: only
    /// encoders with a smaller identifier can have recorded them. Sorted by that bound.
    pending: RefCell<VecDeque<(u64, Resource)>>,
}

impl Retirement {
    /// Shared ownership of `resource`, destroyed when the last clone is dropped.
    pub(crate) fn own(self: &Rc<Self>, resource: Resource) -> Rc<Owned> {
        Rc::new(Owned {
            resource,
            retirement: Rc::clone(self),
        })
    }

    /// Opens a command encoder; the token closes it when dropped (after its submit).
    pub(crate) fn open_encoder(self: &Rc<Self>) -> Rc<EncoderToken> {
        let id = self.next_encoder.get();
        self.next_encoder.set(id + 1);
        self.open_encoders.borrow_mut().insert(id);
        Rc::new(EncoderToken {
            id,
            retirement: Rc::clone(self),
        })
    }

    fn retire(&self, resource: Resource) {
        if self.open_encoders.borrow().is_empty() {
            resource.destroy();
        } else {
            let bound = self.next_encoder.get();
            self.pending.borrow_mut().push_back((bound, resource));
        }
    }

    fn close_encoder(&self, id: u64) {
        let oldest = {
            let mut open = self.open_encoders.borrow_mut();
            open.remove(&id);
            open.first().copied()
        };
        loop {
            let next = {
                let mut pending = self.pending.borrow_mut();
                match pending.front() {
                    Some((bound, _)) if oldest.is_none_or(|oldest| *bound <= oldest) => {
                        pending.pop_front()
                    }
                    _ => None,
                }
            };
            match next {
                Some((_, resource)) => resource.destroy(),
                None => break,
            }
        }
    }
}

/// The owning reference of a device-created texture or buffer (see the module docs).
#[derive(Debug)]
pub(crate) struct Owned {
    resource: Resource,
    retirement: Rc<Retirement>,
}

impl Drop for Owned {
    fn drop(&mut self) {
        self.retirement.retire(self.resource.clone());
    }
}

/// Keeps a command encoder (then its command buffer) open until it is submitted or dropped.
#[derive(Debug)]
pub(crate) struct EncoderToken {
    id: u64,
    retirement: Rc<Retirement>,
}

impl Drop for EncoderToken {
    fn drop(&mut self) {
        self.retirement.close_encoder(self.id);
    }
}
