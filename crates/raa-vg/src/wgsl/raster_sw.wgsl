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
// crates/solstice-render/shaders/vg/raster_sw.wgsl:1-11.
// The include preprocessor is not ported: `raster_sw_module_wgsl()` concatenates
// `raster_common.wgsl`, `vis_write.wgsl`, this file and `raster_sw_body.wgsl`, so the
// visibility write and the body see the group-2 bindings 0-8 of the hardware pass in one
// compute stage. `vis_write.wgsl` keys the depth through `raa_depth_key`.

// The visibility buffer takes depth as it is.
fn depth_bias(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
    return 0.0;
}
