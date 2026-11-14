// The `gi` provider of the `Ao` slot (S11/S39): the screen-space AO of the lit renderer (S10
// group 1 binding 7, `.r` as in crates/raa-gtao/src/wgsl/slot_ao.wgsl:4-6, with the load
// clamped) and the occlusion stored in the alpha of the GI screen texture (binding 14), one
// screen occlusion: the stronger of the two, never their product. This is RAA code.
//
// This file defines exactly `raa_slot_ao` plus the helpers `raa_gi_ao_screen` and
// `raa_gi_ao_gtao`; the textures are declared by the composed prelude.

// The GI screen texel at the fragment's pixel, clamped to `textureDimensions - 1`: an inactive
// GI binds a 1x1 dummy of (0, 0, 0, 1), so its alpha is exactly 1 and the slot equals gtao,
// while an out-of-bounds load may return zero.
fn raa_gi_ao_screen(frag: vec4<f32>) -> vec4<f32> {
    let size = vec2<i32>(textureDimensions(raa_gi_screen));
    let pixel = clamp(vec2<i32>(frag.xy), vec2<i32>(0), size - vec2<i32>(1));
    return textureLoad(raa_gi_screen, pixel, 0);
}

// The gtao texel at the fragment's pixel, clamped the same way: without gtao (or with its pass
// disabled) binding 7 is the 1x1 dummy of 1, and an unclamped load beyond texel (0, 0) is out
// of bounds, which robust backends answer with zero. That zero went through the combined
// occlusion `min(s.ao, ao)` into the Ambient term and removed the ambient light from every
// pixel but (0, 0).
fn raa_gi_ao_gtao(frag: vec4<f32>) -> f32 {
    let size = vec2<i32>(textureDimensions(raa_ao_tex));
    let pixel = clamp(vec2<i32>(frag.xy), vec2<i32>(0), size - vec2<i32>(1));
    return textureLoad(raa_ao_tex, pixel, 0).r;
}

fn raa_slot_ao(frag: vec4<f32>) -> f32 {
    return min(raa_gi_ao_gtao(frag), raa_gi_ao_screen(frag).a);
}
