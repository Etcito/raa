//! Validate an exported SDTP stream without creating a GPU or compiling shaders.
fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: read_program <scene.sdtp>");
    let bytes = std::fs::read(path).expect("read program");
    let p = raa_sdf_trace::parse(&bytes).expect("valid trace program");
    println!(
        "{} sources, {} instances, {} instructions, {} sampled leaves",
        p.sources.len(),
        p.instances.len(),
        p.code.len(),
        p.code.iter().filter(|w| **w & 255 == 9).count()
    );
}
