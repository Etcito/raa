# Joint real-frame fit

The candidate uses the existing `raa.real-frame-model/1` FP32 ABI and the same runtime tensor and dispatch sizes as `df4c0ace`. The recurrent trunk and the three heads train in one optimizer. Each display unroll reads the current forward HDR and raw SSR incident data into the recurrent state. The display decoder and its jitter-corrected bilinear comparison both read the same submitted SSR composite. Offline replay holds that composite fixed when comparing checkpoints. In the full render graph, upstream learned ray reconstruction can also change that input.

The primary SR teacher is the same raster graph rendered directly at the display extent with the same camera, simulation, and lights. A separate high-sample mesh path estimate at that pose measures the error against a different estimator. The conditional SSR teacher is evaluated on actual valid screen hits, paired with the recorded roughness and cone level. Coverage, source metadata, and the existing offscreen fallback stay in the provider. The Ticket 10 area visibility data supplies auxiliary training. It has no independent area holdout.

Training starts the display head at the positive bilinear identity and warm-starts the other weights from `df4c0ace`. Six-frame windows and up to four earlier burn-in frames carry feature history. The objective combines log HDR, relative linear energy, fixed-display error, image gradients, and motion-warped reconstruction-error change. Light event pixels receive extra current-frame weight. Reflection change error compares only valid same-source correspondences outside reactive regions. Area supervision weights the penumbra. Only development trajectories enter checkpoint selection.

Use Kong's CPU Python environment with PyTorch installed. The trainer verifies the captured plane hashes and refuses training or development overlap at final scoring using trajectory groups, replay hashes, frame IDs, and source/guide/target fingerprints.

## Reproduce the development selection

Capture manifests for each role remain separate. Give the trainer the three train tapes and the development tape only. The fixed budget compared path auxiliary weights `0` and `0.15`. Each variant trained for 120 CPU steps, checked development every 20 steps, and started from the same seed and checkpoint. Select by the development composite in `joint-fit-report.json`.

```sh
python -B scripts/train-real-frame.py fit-joint out/model   --train <train-a/manifest.json> <train-b/manifest.json> <train-c/manifest.json>   --development <development/manifest.json>   --base examples/global-illumination/lib/gallery-temporal-fit   --area-capture <area-pilot-win/manifest.json>   --steps 120 --unroll 6 --validate-every 20 --learning-rate 0.001   --seed 0 --path-weight <0-or-0.15>
```

Commit the selected package before releasing the final captures. Both complete final trajectories use the saved network's previous features. Display scoring uses fixed unit exposure, Reinhard, and sRGB. Warp error compares the current reconstruction residual with the motion-warped previous residual, so a real change present in the teacher is preserved. The report averages eligible frames within each trajectory, then weights trajectories equally.

```sh
python -B scripts/train-real-frame.py compare-joint out/final-report.json   --candidate examples/global-illumination/lib/gallery-temporal-joint-fit   --base examples/global-illumination/lib/gallery-temporal-fit   --final <final-a/manifest.json> <final-b/manifest.json>
```

For the composed Windows graph comparison, also pass `--resolved-candidate` and `--resolved-existing` with the two matching native replay manifests for each checkpoint. The scorer verifies the exact replay, frame clock, event, extent, jitter, and weight hash. It keeps separate arms for analytic, all learned heads, and learned SSR plus SR with analytic area lighting.

## Frozen checkpoint

The native RTX 3090 capture stage at `948dc6d` supplied 50 training frames over
three clips and 19 development frames over a fourth. The matched-raster fit
won the development selection by 0.938043 vs 0.938207 for the path auxiliary.
The selected step was 120. Both fits improved displayed PSNR over the prior
checkpoint on development. The matched fit reached 26.12 dB, compared with
23.60 dB for the prior checkpoint and 25.91 dB for bilinear.

The candidate weighs 6,560 bytes. Its weights SHA-256 is
`d6b8d801d2d414110527265a5c88bad468235ba8e6bcb7a6427f274fb5cc3e6d`
and its package SHA-256 is
`1d2d02260fe464129a754aa78f9398bb4b166ba3e261eaa434f65f0f7897b3f3`.
The candidate remains experimental and is stored alongside the original checkpoint.

## Independent final score

The package was committed in `701eccc` before opening either final trajectory.
The final set contained 43 Windows RTX 3090 captures over two independently
seeded camera, motion, and light-event replays. Every source and teacher plane
passed its stored SHA-256. These numbers are CPU replays of the frozen model on
the actual GPU-captured inputs. They isolate reconstruction with a fixed input
composite at 160×90 → 320×180.

| Held-out signal | Bilinear / raw | Prior `df4c0ace` | Candidate `d6b8d80` |
|---|---:|---:|---:|
| Matched raster display PSNR (dB) | 25.842 | 23.928 | 26.024 |
| Matched raster display SSIM | 0.90617 | 0.78951 | 0.90756 |
| Matched raster linear HDR RMSE | 0.06423 | 0.06525 | 0.06466 |
| Matched raster warped log change MAE | 0.006240 | 0.006950 | 0.006217 |
| Matched raster light-event log MAE | 0.006791 | 0.010924 | 0.006777 |
| Path-traced display PSNR (dB) | 14.824 | 14.412 | 14.825 |
| Path-traced display SSIM | 0.47314 | 0.41546 | 0.47208 |
| Path-traced linear HDR RMSE | 0.17447 | 0.17438 | 0.17510 |
| Path-traced warped log change MAE | 0.011790 | 0.012006 | 0.011765 |
| Accepted SSR incident linear HDR RMSE | 0.08645 | 0.07779 | 0.06492 |
| Accepted SSR incident log MAE | 0.04911 | 0.04492 | 0.03954 |
| Accepted SSR incident warped log change MAE | 0.007624 | 0.008361 | 0.008547 |

Both final trajectories gained matched-raster display PSNR, SSIM, and
warped-change quality over both baselines. Bilinear kept the lower matched
raster HDR RMSE. The candidate reduced accepted-SSR HDR RMSE by 16.6% over the
prior fit and 24.9% over raw, with a 2.2% increase in SSR temporal error over
the prior fit. Path-traced disocclusion log MAE was 0.09485 vs 0.09231 for the
prior fit. These differences keep the checkpoint opt-in.

Area visibility RMSE on the six auxiliary fitting frames was 0.14033, compared
with 0.14148 for the prior checkpoint and 0.20046 for the raw pairs. A separate
CPU development replay of two gallery area slots yielded 0.07744 for the
candidate, 0.07758 for the prior fit, 0.11304 for one raw ray, and 0.03686 for
nine actual rays. That diagnostic replay used the recorded forward guides
and no incident history input. The corresponding earlier GPU diagnostic had
zero area output support because the typed area receiver was prepared too late.
The runtime fix is distinct from the model fit. Area lacks a final checkpoint
qualification here.

Machine-readable selection, fit and final reports are under
`out/joint-real-frame-d6b8d80`. The native candidate composed replay, paired 720p GPU timing, and physical
input-to-photon gate remain unverified. The currently shipped
gallery continues loading the prior checkpoint on its existing opt-in path.
