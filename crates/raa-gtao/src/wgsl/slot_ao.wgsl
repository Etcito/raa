// The `gtao` provider of the `Ao` slot (S11): the screen-space AO texture of the lit
// renderer (S10 group 1 binding 7), sampled at the fragment's pixel with the AO in `.r`.
// This file defines exactly `raa_slot_ao`; the texture is declared by the composed prelude.
fn raa_slot_ao(frag: vec4<f32>) -> f32 {
    return textureLoad(raa_ao_tex, vec2<i32>(frag.xy), 0).r;
}
