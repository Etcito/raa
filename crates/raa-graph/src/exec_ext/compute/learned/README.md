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

### Fixed six-frame numerical stability experiment

The legacy run at learning rate 0.003 first produced non-finite gradients at
step 400 on `main-16`. Its per-frame loss was finite. One pixel at (31, 44)
had blended S of zero and positive U of `2.25609e-43`. Differentiating the
quotient overflowed FP32 before the gradients reached the convolution layers.
A training-only fix exposed a related forward failure in the old reduction.
The best checkpoint appeared to beat the mixture bound because 76 outputs
rounded to the visibility-one fallback outside their valid mixture intervals.

The trainer and shader now normalize the log masses of supported pairs before
reducing their visibility. A spatial mass includes the spatial log-softmax and
log U. The previous mass includes log U and the raw gate logit after cancelling
the common current-frame factor. True empty support resolves to one. Adam
checks the gradient norm and clips it at one. The measured maximum norm was
0.08452, so clipping never fired. Checkpoint scoring and export use the
shader's ordered FP32 fused multiply-adds from the bias for each convolution.
The ordinary PyTorch convolution reference exceeded the existing tolerance on
19 lanes at the blocker step. The fused reference passed with the same weights
and tolerance.

The fixed capture manifest SHA-256 is
`10bb17ad439e04e8a82942b877018745ed51f4b3cbc791db5610173d2a7f72ad`.
The original capture, replay and fitted packages are retained under `out/`.
From the worktree root, the final run used CPU PyTorch 2.15.0+cpu:

```sh
python -B scripts/train-area-pilot.test.py
python scripts/train-area-pilot.py out/area-pilot-win/manifest.json out/area-pilot-stable-fma-3000 --steps 3000
```

The best FP32 checkpoint was step 2975 of 3000 at learning rate 0.003.
The fitted MSE is 27.6% below the embedded initialization and 15.5% below
the earlier 500-step fit. The oracle lets each pixel independently choose
any visibility between the minimum and maximum S/U of its positive-weight
3×3 taps and accepted previous pair. Its error is the squared distance from
the teacher to that interval. Empty support contributes the error at one.
This bound gives the best possible mixture with those inputs and unrestricted
per-pixel coefficients. It accounts for 92.5% of the fitted MSE on this set.

| Frame | Initial MSE | Fitted MSE | Oracle MSE | Fit minus oracle |
| --- | ---: | ---: | ---: | ---: |
| main-0 | 0.024148 | 0.019153 | 0.016764 | 0.002389 |
| main-14 | 0.021960 | 0.016300 | 0.014616 | 0.001684 |
| main-15 | 0.023145 | 0.017793 | 0.016449 | 0.001344 |
| main-16 | 0.037295 | 0.026952 | 0.025989 | 0.000963 |
| main-17 | 0.040729 | 0.028614 | 0.026757 | 0.001857 |
| main-18 | 0.037351 | 0.024952 | 0.023162 | 0.001790 |
| Mean | 0.030771 | 0.022294 | 0.020623 | 0.001671 |

The final vectors stayed within their oracle intervals to `1e-6`. Windows
wgpu/Vulkan replay on the non-fallback RTX 3090 (driver 595.79) passed all
229,320 fitted output lanes at the existing tolerance. Maximum absolute
difference was `3.58e-7`. The five embedded golden cases also passed with
maximum absolute difference `1.19e-7`. The full report is
`out/area-pilot-stable-fma-3000/training-report.json` and the fitted parity
log is `out/area-pilot-stable-fma-parity.log`. This experiment measured
fixed-set accuracy and numerical parity. The revised shader's cost and
held-out quality remain unmeasured.
