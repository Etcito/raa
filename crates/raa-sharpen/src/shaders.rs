//! WGSL of the sharpen pass (S47).

/// The fullscreen sharpen module.
pub const SHARPEN_WGSL: &str = include_str!("wgsl/sharpen.wgsl");
/// Vertex entry of the fullscreen triangle.
pub const VS_MAIN: &str = "vs_main";
/// Fragment entry applying the contrast-adaptive kernel.
pub const FS_SHARPEN: &str = "fs_sharpen";
