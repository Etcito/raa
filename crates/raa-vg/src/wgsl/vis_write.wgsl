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
// crates/solstice-render/shaders/vg/vis_write.wgsl:1-27.
// RAA is standard-Z, so the reversed-Z bits of the source become the depth key of the
// `raa_depth_key` helper: larger is nearer and 0 is empty. It orders the same way, so
// the integer compare-exchange loop is kept word for word.

@group(2) @binding(7) var<storage, read_write> vis_depth: array<atomic<u32>>;
@group(2) @binding(8) var<storage, read_write> vis_id: array<atomic<u32>>;

fn write_visibility(index: u32, depth: f32, id: u32) {
    let key = raa_depth_key(depth);
    var current = atomicLoad(&vis_depth[index]);
    loop {
        if key <= current {
            return;
        }
        let result = atomicCompareExchangeWeak(&vis_depth[index], current, key);
        if result.exchanged {
            break;
        }
        current = result.old_value;
    }
    atomicStore(&vis_id[index], id);
}
