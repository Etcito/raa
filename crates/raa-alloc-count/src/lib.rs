//! A counting global allocator for allocation benchmarks and demos.
//!
//! This crate is the single global-state exception in RAA. It is linked as a
//! `#[global_allocator]` only by `raa-bench` and by demos with the
//! `alloc-count` feature; library crates never install it.
//!
//! Usage in a binary or feature-gated demo:
//!
//! ```ignore
//! #[global_allocator]
//! static A: raa_alloc_count::CountingAlloc = raa_alloc_count::CountingAlloc;
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

/// A system allocator that counts allocation requests and requested bytes.
pub struct CountingAlloc;

// SAFETY: operations are forwarded to `System` (test: counting_alloc_forwards_to_system)
unsafe impl GlobalAlloc for CountingAlloc {
    // SAFETY: forwards the caller layout unchanged (test: counting_alloc_forwards_to_system)
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        // SAFETY: forwards the caller layout unchanged (test: counting_alloc_forwards_to_system)
        unsafe { System.alloc(layout) }
    }

    // SAFETY: forwards the caller layout unchanged (test: counting_alloc_forwards_to_system)
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        // SAFETY: forwards the caller layout unchanged (test: counting_alloc_forwards_to_system)
        unsafe { System.alloc_zeroed(layout) }
    }

    // SAFETY: forwards pointer and layout unchanged (test: counting_alloc_forwards_to_system)
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        // SAFETY: forwards realloc arguments unchanged (test: counting_alloc_forwards_to_system)
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    // SAFETY: forwards pointer and layout unchanged (test: counting_alloc_forwards_to_system)
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwards deallocation arguments unchanged (test: counting_alloc_forwards_to_system)
        unsafe { System.dealloc(ptr, layout) }
    }
}

/// Returns the number of allocation, zeroed allocation, and reallocation calls.
pub fn allocations() -> u64 {
    ALLOCATIONS.load(Ordering::Relaxed)
}

/// Returns the total requested bytes across allocation and reallocation calls.
pub fn bytes() -> u64 {
    BYTES.load(Ordering::Relaxed)
}

/// Resets both counters to zero.
pub fn reset() {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::{CountingAlloc, allocations, bytes};

    #[global_allocator]
    static A: CountingAlloc = CountingAlloc;

    #[test]
    fn counting_alloc_forwards_to_system() {
        let before_allocations = allocations();
        let before_bytes = bytes();
        let mut values = Vec::with_capacity(1000);
        values.extend(0_u64..1000);

        for (index, value) in values.iter().enumerate() {
            assert_eq!(*value, index as u64);
        }
        assert!(allocations() > before_allocations);
        assert!(bytes() >= before_bytes + 8000);

        let original_capacity = values.capacity();
        values.reserve(original_capacity + 1);
        assert!(values.capacity() > original_capacity);
        assert_eq!(values[999], 999);
    }
}
