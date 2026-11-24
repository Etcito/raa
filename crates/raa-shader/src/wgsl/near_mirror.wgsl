// WGSL perceptual roughness is squared by GGX. Its lobe at 0.08 remains narrow
// enough for one deterministic reflection direction. Broader and dielectric
// lobes stay on the standard reflection routes.
fn raa_near_mirror(metallic: f32, roughness: f32) -> bool {
    return metallic >= 0.95 && roughness <= 0.08;
}
