# raa-exposure

Automatic exposure from a luminance histogram: the pass `exposure.auto` (S46, order 125, disabled
by default). It measures the log luminance of the HDR chain on the GPU, adapts an exposure over
frames and multiplies the HDR colour by it, after `bloom` and before `grading`.

The crate holds the constants, the settings, the maths twins, the WGSL shaders, the readback pool
and the feature: `AutoExposureFeature::install` adds the pass, and its `AutoExposureHandle` reads
and changes it.

## Constants (`src/lib.rs`)

`AUTO_EXPOSURE_PASS_NAME` = `"exposure.auto"`, `AUTO_EXPOSURE_ORDER` = 125, `HISTOGRAM_BINS` = 128,
`MIN_LOG_LUMINANCE` = -8.0, `LOG_LUMINANCE_RANGE` = 18.0, `SAMPLE_STRIDE` = 4 (every 4th pixel in x
and y), `HISTOGRAM_WORKGROUP` = 16 (16 x 16), `LUMINANCE_FLOOR` = 1e-5, `MIDDLE_GREY` = 0.18,
`SETTLED_EV` = 0.01, `PARAMS_SIZE` = 48, `RESULT_SIZE` = 16, `READBACK_SLOTS` = 3.

## Settings (`AutoExposureDesc`, `src/desc.rs`)

| Rust field | JS key | default | rule (else `post.invalid` naming the JS key) |
|---|---|---|---|
| `enabled` | `enabled` | false | |
| `compensation_ev` | `compensationEv` | 0.263 (log2 1.2) | finite, -8..=8 |
| `low_percent` | `lowPercent` | 0.2 | finite, 0..=1, below `highPercent` |
| `high_percent` | `highPercent` | 0.7 | finite, 0..=1 |
| `speed_up` | `speedUp` | 2.4 | finite, > 0 |
| `speed_down` | `speedDown` | 1.1 | finite, > 0 |
| `min_ev` | `minEv` | -6 | finite, <= `maxEv` |
| `max_ev` | `maxEv` | 6 | finite |

`AutoExposureDesc::validate` checks the rows in this order and returns the first failure as
`RaaError::FrameGraph { code: "post.invalid", message }`, where the message is
``"`<jsKey>` must be <rule>, got <value>"``. The low/high order failure names `lowPercent`.

## Pass and readback

The pass runs at order 125 and is disabled by default. Its record dispatches a 128-bin histogram
over the HDR source, resolves the band average into an `ExposureState` (16 B) and applies the
linear exposure to the HDR colour; `setToneMapping`'s `exposureEv` multiplies after it.

The readback has `READBACK_SLOTS` (3) MAP_READ buffers of 16 B. A slot copied in frame n is mapped
in the next prepare of the same pass; its callback stores the decoded result and returns the slot.
There is no synchronous wait and no growth, a slot is never mapped twice, and results older than
the latest snap are ignored.

## Handle (`src/feature.rs`)

`AutoExposureFeature::install(gpu, pipeline)` builds the two compute pipelines, the apply render
pipeline, the params, histogram and state buffers and the readback pool, then adds the pass
`exposure.auto` at order 125 and installs it disabled, so a page that never calls the setters
renders as before. A borrowed pipeline (`pipeline.busy`) or a second install
(`pipeline.duplicate_pass`) refuses with nothing installed. The returned `AutoExposureHandle`
shares its state with the pass and its clones: `get` and `set` read and replace the
`AutoExposureDesc` (`set` validates first, so a refusal changes nothing), `set_enabled_flag`
mirrors the pipeline flag without validation, `invalidate` makes the next prepared frame snap, and
`state` reports the tracker's state for that flag.

## Attribution

Adapted from solstice@7e465b5 crates/solstice-render/shaders/post/exposure.wgsl, MIT licence,
Copyright (c) 2026 SamG-Coder
