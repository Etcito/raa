// The `gi` provider of the `Indirect` slot (S11/S39): the screen-space GI irradiance of the lit
// renderer (S10 group 1 binding 14) read at the fragment's pixel and clamped into the texture,
// added as indirect diffuse. Only the material ao `s.ao` applies: `g.rgb` is traced irradiance
// with its own visibility and `g.a` belongs to the same traces, so neither GTAO nor `g.a` is
// stacked on it. Transparent fragments and fur shells read the opaque pixel behind
// them, the same approximation as S12 tiling.
//
// Adapted from solstice@7e465b5,
// crates/solstice-render/shaders/lighting/deferred.wgsl:173-175. MIT License
//
// Copyright (c) 2026 SamG-Coder
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// This file defines exactly `raa_slot_indirect` plus helpers prefixed `raa_gi_`. The texture and
// the volume params are declared by the composed prelude (S38 bindings 14 and 18).

const RAA_GI_INDIRECT_PI: f32 = 3.141592653589793;

// `g` of S39: `textureLoad` of the GI screen at the fragment's pixel, clamped to
// `textureDimensions(raa_gi_screen) - 1`. An inactive GI binds a 1x1 dummy (0, 0, 0, 1), which
// the clamp keeps exact where an out-of-bounds load may return zero.
fn raa_gi_indirect_screen(frag: vec4<f32>) -> vec4<f32> {
    let size = vec2<i32>(textureDimensions(raa_gi_screen));
    let pixel = clamp(vec2<i32>(frag.xy), vec2<i32>(0), size - vec2<i32>(1));
    return textureLoad(raa_gi_screen, pixel, 0);
}

fn raa_slot_indirect(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {
    let g = raa_gi_indirect_screen(c.frag);
    return (1.0 - s.metallic) * s.base_color / RAA_GI_INDIRECT_PI
        * g.rgb * raa_gi_volume_params.intensity * s.ao;
}
