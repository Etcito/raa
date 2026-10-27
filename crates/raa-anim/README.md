# raa-anim

Dependency-free JSON reading and canonical JSON writing for animation assets.
`Rig::load` parses S16 bone names, local rest transforms and parent-before-child
hierarchies, with one root and JSON-pointer error paths. Rest numbers retain f64
source precision; quaternions are validated without normalization.
`Rig::to_canonical_json` writes accepted values in S16 field order, with `roles`
and `attach` sorted bytewise, so canonical rigs round-trip byte-identically and
non-canonical input canonicalises idempotently (`1.0` -> `1`, `-0` -> `0`,
`1e21` -> `1e+21`). Bone references are written back as names.

Supported data: unique bone names matching `^[A-Za-z0-9_.-]{1,32}$` with one
null-parent root; S16 roles; `twoBone` (three bones) or `ccd`/`fabrik` (two to
eight bones) chains with optional unit pole and 1-16 iterations; per-bone XYZ
limits with `min <= max`; named attachment transforms.

Ragdoll data is deliberately unsupported: S16 is the Sketchbook `RigPreset`
schema without ragdoll fields, so a `ragdoll` key or any unknown field is
refused with `rig.schema`.

`raa_anim::MAX_BONES` is **256**. Otherwise-valid rigs with exactly 256 bones are
accepted; larger bone arrays fail with `rig.cap_bones`, never truncation. This
same rig limit applies to rigs used by the clip API.

`Clip::load` reads S17 `raa.clip` v2 documents against a loaded rig: at most one
track per bone and channel, each bone from the rig; `translation` (3 values),
`rotation` (quaternion, 4 values) or `scale` (3 values) keys with strictly
increasing integer ticks; `CONSTANT`/`LINEAR`/`BEZIER` interpolation with
optional AClip v2 `hl`/`hr` handles (allowed only on BEZIER keys); named events;
and `none`/`xz`/`xyz` root motion. Key values and handles retain their source
document numbers, and `Clip::to_canonical_json` writes S17 field order.

`raa_anim::CLIP_CAPS` is the inclusive S17 count contract:
`max_tracks = 256`, `max_keys_per_track = 4096`, `max_events = 1024`. Documents
exactly at a limit load; one above fails with `clip.cap` and is never truncated.
There is no additional `lengthTicks` cap, only its S17 integer/range rules.

`Clip::sample_track`, `local_pose_at` and `pose_at` sample integer ticks. Looping
clips address `tick % lengthTicks`; individual tracks clamp outside their key
range. Empty tracks retain the corresponding rest channel. Bezier handle `dt`
offsets are preserved but ignored during sampling. Rotation keys and their
handle offsets are sign-aligned at load, then sampled rotations are normalized.
`pose_at` returns local TRS plus column-major model and skin matrices.

Run `cargo test -p raa-anim --test clip sample_reference -- --nocapture` to
compare native track values, local transforms, model matrices and skin matrices
against every tick in `tests/fixtures/samples.json`. The test widens native
values to document-number precision and prints the measured maximum absolute
difference, requiring at most `1e-6`; it does not compare hashes. The independent
JavaScript reference and regeneration instructions live in
[`tools/anim-ref`](../../tools/anim-ref/README.md). Cases run sequentially, so no
independent worker processes are started.

`Clip::root_delta_at(rig, tick)` returns a stateless `[f32; 3]` translation
increment. Tick 0 returns zero; the previous sample always means `tick - 1`,
regardless of API call order. `rootMotion: xz` selects x/z, `xyz` selects all
translation axes, and `none` returns zero. `local_pose_at` and `pose_at` restore
selected axes to the unique parent-null bone's rest translation; unselected
translation axes, rotation, scale and other bones keep their sampled values.
`sample_track` continues to expose the unextracted channel.

For length L > 0, let P(s) sample the root without wrapping on [0,L]. Loop
motion follows U(t) = P(t mod L) + floor(t/L)*(P(L)-P(0)). The delta is
U(t)-U(t-1), evaluated using adjacent local ticks without accumulating a loop
count in f32. Every seam uses P(L)-P(L-1); pose addressing remains modulo L.
Non-looping motion uses P(min(t,L))-P(min(t-1,L)), giving zero after completion.
Zero duration returns zero defensively; existing JSON validation still requires
positive `lengthTicks`. No fields or validation constraints were added.
