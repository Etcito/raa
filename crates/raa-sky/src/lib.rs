//! Procedural sky of the lit renderer (S12 `sky.env`, `sky.background`).
//!
//! [`SkyDesc`] describes a zenith/horizon/ground gradient, a sun disc and a procedural cloud
//! layer ([`CloudDesc`]) in scene-referred linear HDR. [`SkyDesc::preset`] gives the built-in
//! moods (`day`, `snow`, `jungle`, `night`) and [`SkyDesc::validate`] refuses invalid
//! descriptions with `sky.invalid`.

mod desc;

pub use desc::{CloudDesc, EnvUpdate, SkyDesc};
