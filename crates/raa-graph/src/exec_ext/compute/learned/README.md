# FP32 area inference pilot

The pilot runs on the renderer's existing `GpuContext`. Each view can own a default-empty `raa_graph::learned::AreaPilotSlot`. Load a validated manifest and weights with `AreaPilotPackage::load` or select `embedded()`. Await `AreaPilot::prepare` and stage the result. At the beginning of a real frame, call `begin_real_frame`, upload the scene-grid input and call `record_real_frame` on the frame encoder. Submit before uploading another input to the same pilot. An owner can publish the resident weights and output to graph buffer ports with `publish`.

The manifest specifies NHWC input order and normalization multipliers. An input occupies the full scene grid. The active rectangle sets both dispatch and spatial gather bounds. The shader produces twelve FP32 values per pixel in the same layout. The first nine are normalized spatial weights, followed by the hard-masked history gate, confidence and resolved visibility. Confidence is exposed independently. The spatial filter and temporal blend apply the same coefficients to S and U before division.

Run the following commands from the RAA checkout on Windows with Kong's Cargo environment and a hardware adapter.

```sh
python3 scripts/generate-area-pilot.py --check
cargo test -p raa-graph --test compute_pass --test compute_validation
cargo test -p raa-graph --test compute_pass benchmark_fp32 -- --ignored --nocapture
```

The scalar generator freezes the deterministic synthetic weights, a per-tensor manifest and five golden vectors. The package digest covers the canonical sorted compact manifest with its package digest omitted, followed by the weights and golden file bytes.

The benchmark uses 960×540 FP32 inputs, one discarded warm-up per mode and twenty timed submissions per mode. Set `RAA_PILOT_SAMPLES` to change the count. Timestamp queries bracket the dispatches when supported. Wall time includes CPU packing, queue upload, encoding, submission, synchronization and timing readback. Cold preparation includes compilation of all six pipeline entry points, allocation and the first weight and uniform transfers. Buffer totals cover explicit allocations. Queue-internal staging, driver caches and pipeline allocations are opaque. The reported host input and packing buffers overlap during upload.

The four-light control repeats the same synthetic twelve-channel input and reuses the output buffer. It measures the cost of repeating the trunk versus reusing it for four heads under that condition. Distinct per-light S/U inputs require distinct trunk evaluations with this ABI.


## Fixed capture overfit

Run the 960×540 benchmark first. Export a small fixed set of scene-native, fixed-exposure frames from the Windows capture workflow. The trainer accepts a JSON manifest with format `raa.area-pilot-capture/1` and a `frames` array. Each record contains a unique string `id`, a nonempty `referenceSeeds` array of unsigned integers, `size: [width, height]`, and `input` and `reference` descriptors with a relative `file` and its lowercase `sha256`.

The two files are tightly packed little-endian FP32. Input occupies the whole active rectangle in row-major NHWC with all twelve channels in the ABI manifest. Pack the shader's raw scene-linear values before normalization. Reference contains one visibility ratio per pixel from the fixed-pose reference, with zero denominator resolved to one. Crop both tensors to the same active rectangle before invoking the trainer.

With CPU PyTorch available, run `python scripts/train-area-pilot.py capture.json fit` to create a new output directory. Training starts from the byte-exact synthetic initialization and minimizes equally weighted per-frame visibility MSE. The confidence output has no separate training target in this experiment. The report freezes the capture manifest SHA-256, initialization hash, frame IDs, reference seeds, optimizer settings, initial and final loss, and per-frame error. The export retains the synthetic fixture metadata for ABI checks and writes the fitted model's trainer vectors separately.

On Windows set `RAA_PILOT_FIT_DIR` to that output directory and run `cargo test -p raa-graph --test compute_pass fitted_capture_matches_trainer -- --ignored --nocapture`. This reloads the weight package, dispatches the pilot on a hardware adapter and checks every output lane against the trainer output. Its log reports maximum absolute error and the number of lanes outside tolerance for every frame.
