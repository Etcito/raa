/*
MIT License

Copyright (c) 2026 SamG-Coder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
// Adapted from solstice@7e465b599c13da165948ebaa44df824c36c3ae07
// crates/solstice-render/shaders/vg/depth_stamp.wgsl:1-21.
// A drawing's stamp goes in the top byte of each word, over the top 24 bits of
// `raa_depth_key`: larger is nearer and 0 is far or empty, so an atomic max keeps the
// nearest sample of the latest drawing and the buffer is cleared only when the stamps start
// a new round. Solstice's reversed 24-bit float scale is not ported; the key keeps the
// order. Twin of the CPU `stamped`, `stamp_of` and `stamped_depth` in raster.rs.

const DEPTH_STAMP_SHIFT: u32 = 24u;
const DEPTH_STAMP_RANGE: u32 = 16777215u;

fn stamped(stamp: u32, depth: f32) -> u32 {
    return (stamp << DEPTH_STAMP_SHIFT) | (raa_depth_key(depth) >> 8u);
}

fn stamp_of(word: u32) -> u32 {
    return word >> DEPTH_STAMP_SHIFT;
}

// The depth a word holds, forward again, 0 at the near plane to 1 at the far plane.
fn stamped_depth(word: u32) -> f32 {
    return raa_depth_from_key((word & DEPTH_STAMP_RANGE) << 8u);
}
