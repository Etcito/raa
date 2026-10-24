//! WGSL sources of the OIT head clear and resolve, their entry points and uniform sizes.

/// Compute shader clearing the heads to END and the node counter to 0.
/// Group 0: 0 heads (storage rw), 1 counter (storage rw), 2 params uniform (count, 0, 0, 0).
pub const CLEAR_WGSL: &str = include_str!("wgsl/clear.wgsl");
/// Compute entry point of `CLEAR_WGSL`.
pub const CLEAR_ENTRY: &str = "cs_main";
/// Workgroup size of `CLEAR_WGSL` (x).
pub const CLEAR_WORKGROUP_SIZE: u32 = 64;
/// Size of the clear params uniform in bytes.
pub const CLEAR_PARAMS_SIZE: u64 = 16;

/// Fullscreen resolve: sorts the nearest nodes per pixel and composites them over hdr.src.
/// Group 0: 0 heads (storage read), 1 nodes (storage read), 2 hdr.src texture (textureLoad),
/// 3 params uniform (width, height, max_fragments, 0).
pub const RESOLVE_WGSL: &str = include_str!("wgsl/resolve.wgsl");
/// Vertex entry point of `RESOLVE_WGSL` (fullscreen triangle, 3 vertices).
pub const RESOLVE_VS_ENTRY: &str = "vs_main";
/// Fragment entry point of `RESOLVE_WGSL`.
pub const RESOLVE_FS_ENTRY: &str = "fs_main";
/// Size of the resolve params uniform in bytes.
pub const RESOLVE_PARAMS_SIZE: u64 = 16;
