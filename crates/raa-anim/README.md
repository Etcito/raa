# raa-anim

Dependency-free JSON reading and canonical JSON writing for animation assets.
`Rig::load` parses S16 bone names, local rest transforms and parent-before-child
hierarchies, with one root and JSON-pointer error paths. Rest numbers retain f64
source precision; quaternions are validated without normalization.

`raa_anim::MAX_BONES` is **256**. Otherwise-valid rigs with exactly 256 bones are
accepted; larger bone arrays fail with `rig.cap_bones`, never truncation. This
same rig limit applies to rigs used by the forthcoming clip API.

Roles, IK chains, limits and attachments are currently reserved top-level fields
and are not yet validated. Canonical rig serialization, clip APIs and pose
matrices are subsequent work. `to_canonical_json` writes generic JSON values.
