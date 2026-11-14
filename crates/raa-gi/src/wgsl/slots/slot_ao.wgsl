// The `gi` provider of the `Ao` slot (S11/S39): the screen-space AO of the lit renderer (S10
// group 1 binding 7, loaded as in crates/raa-gtao/src/wgsl/slot_ao.wgsl:4-6) times the occlusion
// stored in the alpha of the GI screen texture (binding 14). This is RAA code.
//
// This file defines exactly `raa_slot_ao` plus the helper `raa_gi_ao_screen`; the textures are
// declared by the composed prelude.

// The GI screen texel at the fragment's pixel, clamped to `textureDimensions - 1`: an inactive
// GI binds a 1x1 dummy of (0, 0, 0, 1), so its alpha is exactly 1 and the slot equals gtao,
// while an out-of-bounds load may return zero.
fn raa_gi_ao_screen(frag: vec4<f32>) -> vec4<f32> {
    let size = vec2<i32>(textureDimensions(raa_gi_screen));
    let pixel = clamp(vec2<i32>(frag.xy), vec2<i32>(0), size - vec2<i32>(1));
    return textureLoad(raa_gi_screen, pixel, 0);
}

fn raa_slot_ao(frag: vec4<f32>) -> f32 {
    return textureLoad(raa_ao_tex, vec2<i32>(frag.xy), 0).r * raa_gi_ao_screen(frag).a;
}
