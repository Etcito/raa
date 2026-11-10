//! Generate the shared CPU and shader constants from one definition.
use std::fmt::Write as _;
use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/wgsl/prelude.wgsl");
    println!("cargo:rerun-if-changed=src/wgsl/weather_lighting.wgsl");
    let out = PathBuf::from(env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    let constants = [
        ("FRAME_GPU_SIZE", "usize", 768),
        ("MAX_LOCAL_LIGHTS", "usize", 256),
        ("LIGHT_GPU_SIZE", "usize", 64),
        ("WEATHER_LIGHTING_GPU_SIZE", "usize", 64),
        ("TILE_SIZE", "u32", 16),
        ("TILE_MAX_LIGHTS", "u32", 127),
        ("TILE_WORDS", "u32", 128),
    ];
    let (first_word, word_bits, word_count) = (2, 32, 2);
    let mut rust = String::new();
    let mut wgsl = String::new();
    for (name, ty, value) in constants {
        writeln!(
            rust,
            "/// Shared GPU layout constant.\npub const {name}: {ty} = {value};"
        )
        .unwrap();
        writeln!(wgsl, "const {name}: u32 = {value}u;").unwrap();
    }
    writeln!(
        rust,
        "\
/// Maps a logical feature id to its absolute flags word and bit.\n\
pub const fn feature_address(id: u32) -> Option<(usize, u32)> {{\n\
    if id < {word_bits} * {word_count} {{\n\
        Some(({first_word} + (id / {word_bits}) as usize, id % {word_bits}))\n\
    }} else {{ None }}\n}}"
    )
    .unwrap();
    writeln!(
        wgsl,
        "\
fn raa_feature_enabled(id: u32) -> bool {{\n\
    if (id >= {word_bits}u * {word_count}u) {{ return false; }}\n\
    let word = {first_word}u + id / {word_bits}u;\n\
    let bit = id % {word_bits}u;\n\
    return (raa_frame.flags[word] & (1u << bit)) != 0u;\n}}\n"
    )
    .unwrap();
    let prelude = fs::read_to_string("src/wgsl/prelude.wgsl").expect("read prelude");
    let weather =
        fs::read_to_string("src/wgsl/weather_lighting.wgsl").expect("read weather lighting");
    fs::write(out.join("gpu_constants.rs"), rust).expect("write Rust constants");
    fs::write(
        out.join("prelude.wgsl"),
        format!("{prelude}\n{weather}\n{wgsl}"),
    )
    .expect("write prelude");
}
