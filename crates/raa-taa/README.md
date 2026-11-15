# raa-taa

Temporal anti-aliasing for the lit renderer and the reusable ping-pong histories that later
temporal passes share. It resolves at render resolution 1:1, reading `hdr.src()`, the S12
`depth` and `motion` targets and the FrameGpu uniform, writing `hdr.dst()` and its history,
then swapping.

## Public items

- `HistoryState` (CPU only): the clamped size, the write index (0 or 1), the read index
  (the other one) and the accumulated-frame count. `commit` flips parity and adds one frame;
  `invalidate` drops the count to 0; `resize` stores a different clamped size and
  invalidates. `is_valid` is true after a commit; `usable(frame_history_valid)` is
  `is_valid()` and the caller's frame history.
- `HistoryPair`: two D2 textures of one format with default views, labelled `<label>.0` and
  `<label>.1`, with usages `TEXTURE_BINDING | STORAGE_BINDING | COPY_SRC | COPY_DST`.
  `new` creates them, `ensure` recreates both only when the size changes, `read_view` and
  `write_view` follow the state parity, `texture(index)` and `view(index)` index with
  `index & 1`, and `allocated_bytes` reports two textures at the format's block copy size.
- `TAA_PASS_NAME` (`"taa"`) and `TAA_ORDER` (`100`): the S12 pass name and order.

`HistoryPair` and `HistoryState` are reused by the M6 shadow-mask history and the M7
reservoir history. Their GPU textures are view-local; only submitted work commits them.

## Usage

`TaaFeature::install` adds the S12 compute pass `taa` (order 100, after `particles.blend` 85
and before `dof` 110) already disabled. It reads the lit renderer's live 768-byte FrameGpu
uniform:

```rust
let taa = TaaFeature::install(gpu, &pipeline, &lit.handle().frame_buffer())?;
```

Enable it from Rust with `pipeline.borrow_mut().set_enabled("taa", true)?` or from JS with
`setPassEnabled('taa', true)`. A second install is refused with `pipeline.duplicate_pass`.

`TaaHandle` clones share the pass's state: `accumulated_frames()` and `history_valid()`
report the frames since the last invalidation, and `invalidate()` drops them to 0 at once.

The history is invalidated on every resize (the pair is recreated when the size changes), on
`set_enabled(false)` and on `TaaHandle::invalidate` (a viewport camera cut). The pass uses it
only when its own `HistoryState` is valid and FrameGpu `temporal.y`/`temporal.z` are set; the
resolve shader is adapted from solstice (MIT, Copyright (c) 2026 SamG-Coder).

## Test

`cargo test -p raa-taa --test history`: the CPU parity, the resize and zero-size clamps, and
the commit/invalidate/usable rules.

## License

MIT (see the workspace `LICENSE`).
