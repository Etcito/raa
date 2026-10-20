//! Generational handles and the pool that hands them out.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

/// A typed, generational reference into a [`Pool`].
///
/// A handle stays valid until its entry is removed; after that the slot's generation
/// moves on, so the stale handle returns `None` everywhere. The traits are implemented
/// by hand so that `Handle<T>` is `Copy`, `Eq`, `Hash` and `Debug` whatever `T` is.
pub struct Handle<T> {
    index: u32,
    generation: u32,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Handle<T> {
    fn new(index: u32, generation: u32) -> Self {
        Self {
            index,
            generation,
            _marker: PhantomData,
        }
    }

    /// Slot index inside the pool.
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Generation of the slot when this handle was issued.
    pub fn generation(&self) -> u32 {
        self.generation
    }
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Handle<T> {}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}

impl<T> Eq for Handle<T> {}

impl<T> Hash for Handle<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}

impl<T> fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Handle")
            .field("index", &self.index)
            .field("generation", &self.generation)
            .finish()
    }
}

struct Entry<T> {
    generation: u32,
    value: Option<T>,
}

/// Slot storage addressed by [`Handle`]s. Free slots are reused last-in, first-out.
pub struct Pool<T> {
    slots: Vec<Entry<T>>,
    free: Vec<u32>,
    len: usize,
}

impl<T> Default for Pool<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Pool<T> {
    /// An empty pool.
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            len: 0,
        }
    }

    /// An empty pool with room for `n` entries before reallocating.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            slots: Vec::with_capacity(n),
            free: Vec::new(),
            len: 0,
        }
    }

    /// Stores `v` and returns its handle. The most recently freed slot is reused first.
    ///
    /// # Panics
    /// When the pool would need more than `u32::MAX` slots.
    pub fn insert(&mut self, v: T) -> Handle<T> {
        self.len += 1;
        if let Some(index) = self.free.pop()
            && let Some(entry) = self.slots.get_mut(index as usize)
        {
            entry.value = Some(v);
            return Handle::new(index, entry.generation);
        }
        // Invariant: handle indices are u32; more than u32::MAX live slots cannot be addressed.
        let index = u32::try_from(self.slots.len()).expect("Pool exceeds u32::MAX slots");
        self.slots.push(Entry {
            generation: 0,
            value: Some(v),
        });
        Handle::new(index, 0)
    }

    /// The value behind `h`, or `None` when the handle is stale or unknown.
    pub fn get(&self, h: Handle<T>) -> Option<&T> {
        match self.slots.get(h.index as usize) {
            Some(entry) if entry.generation == h.generation => entry.value.as_ref(),
            _ => None,
        }
    }

    /// Mutable access to the value behind `h`, or `None` when the handle is stale or unknown.
    pub fn get_mut(&mut self, h: Handle<T>) -> Option<&mut T> {
        match self.slots.get_mut(h.index as usize) {
            Some(entry) if entry.generation == h.generation => entry.value.as_mut(),
            _ => None,
        }
    }

    /// Removes and returns the value behind `h`. The slot's generation is bumped, so `h`
    /// and its copies become stale.
    pub fn remove(&mut self, h: Handle<T>) -> Option<T> {
        let entry = self.slots.get_mut(h.index as usize)?;
        if entry.generation != h.generation {
            return None;
        }
        let value = entry.value.take()?;
        entry.generation = entry.generation.wrapping_add(1);
        self.free.push(h.index);
        self.len -= 1;
        Some(value)
    }

    /// Number of live entries.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when the pool holds no live entry.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Live entries with their handles, in index order.
    pub fn iter(&self) -> impl Iterator<Item = (Handle<T>, &T)> {
        self.slots.iter().zip(0u32..).filter_map(|(entry, index)| {
            entry
                .value
                .as_ref()
                .map(|v| (Handle::new(index, entry.generation), v))
        })
    }
}
