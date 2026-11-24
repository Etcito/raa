// Provider ABI. Append a shading kernel with @workgroup_size(64) and dispatch it
// indirectly from byte zero of the work buffer. Write a result only for pixels
// whose exact target HDR the provider can evaluate. The target texture starts clear.
struct RaaPredictionRefreshWork {
    groups_x: u32,
    groups_y: u32,
    groups_z: u32,
    count: u32,
    incomplete: u32,
    pixels: array<u32>,
}

fn raa_prediction_refresh_count(count: u32, capacity: u32) -> u32 {
    return min(count, capacity);
}

fn raa_prediction_refresh_pixel(index: u32, width: u32) -> vec2<i32> {
    return vec2<i32>(i32(index % width), i32(index / width));
}
