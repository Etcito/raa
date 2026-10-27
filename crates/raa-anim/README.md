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
