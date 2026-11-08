//! Standalone resolve shader; no renderer prelude or material bindings.
pub const CEL_WGSL: &str = concat!(
    include_str!("wgsl/cel.wgsl"),
    include_str!("wgsl/color.wgsl"),
    include_str!("wgsl/dither.wgsl"),
    include_str!("wgsl/outline.wgsl"),
);
pub const VS_MAIN: &str = "vs_main";
pub const FS_CEL: &str = "fs_cel";
pub const PARAMS_SIZE: u64 = 112;
