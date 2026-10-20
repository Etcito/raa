# raa-alloc-count

`raa-alloc-count` provides a system-backed global allocator that counts
allocation requests and requested bytes. It is intended for measuring
allocation behavior in benchmarks and demos.

## Public items

- `CountingAlloc` forwards allocation, zeroed allocation, reallocation, and
  deallocation to `std::alloc::System`.
- `allocations()` returns the number of allocation, zeroed allocation, and
  reallocation calls.
- `bytes()` returns requested bytes for those allocation and reallocation
  calls.
- `reset()` sets both counters to zero.

This crate is RAA's single global-state exception. Install it as
`#[global_allocator]` only in `raa-bench` or in demos built with the
`alloc-count` feature. Library crates must never install it.

## Usage

In a binary or feature-gated demo, add:

```rust
#[global_allocator]
static A: raa_alloc_count::CountingAlloc = raa_alloc_count::CountingAlloc;
```

Read the counters with `raa_alloc_count::allocations()` and
`raa_alloc_count::bytes()`. Call `raa_alloc_count::reset()` to start a new
measurement interval.

## Testing

Run `cargo test -p raa-alloc-count`. The unit test installs the allocator in
the test harness, checks allocated vector contents and counter increases, and
grows the vector to exercise reallocation.
