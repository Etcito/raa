// Radiance in two words: 16-bit values on a square-root curve (range 512).
//
// Adapted from solstice@7e465b5, crates/solstice-render/shaders/gi/radiance_codec.wgsl:4-14. MIT License
//
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

const RAA_GI_RADIANCE_RANGE: f32 = 512.0;

fn raa_gi_encode_radiance(c: vec3<f32>) -> vec2<u32> {
    let e = sqrt(clamp(c / RAA_GI_RADIANCE_RANGE, vec3<f32>(0.0), vec3<f32>(1.0)));
    return vec2<u32>(pack2x16unorm(e.rg), pack2x16unorm(vec2<f32>(e.b, 0.0)));
}

fn raa_gi_decode_radiance(packed: vec2<u32>) -> vec3<f32> {
    let e = vec3<f32>(unpack2x16unorm(packed.x), unpack2x16unorm(packed.y).x);
    return e * e * RAA_GI_RADIANCE_RANGE;
}
