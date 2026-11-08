use super::*;
pub(super) fn word(chunk: &[u8]) -> u32 {
    u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
}

pub(super) fn total_words(header: &[u32]) -> Option<u64> {
    let sources = u64::from(header[2]).checked_mul(SOURCE_WORDS as u64)?;
    let instances = u64::from(header[3]).checked_mul(INSTANCE_WORDS as u64)?;
    let materials = u64::from(header[6]).checked_mul(MATERIAL_WORDS as u64)?;
    let bends = u64::from(header[7]).checked_mul(16)?;
    (HEADER_WORDS as u64)
        .checked_add(sources)?
        .checked_add(instances)?
        .checked_add(u64::from(header[4]))?
        .checked_add(u64::from(header[5]))?
        .checked_add(materials)?
        .checked_add(bends)
}

pub(super) fn cut<'a>(
    words: &'a [u32],
    count: usize,
    section: &str,
) -> Result<(&'a [u32], &'a [u32]), TraceError> {
    words
        .split_at_checked(count)
        .ok_or_else(|| TraceError::at(FORMAT, section, "the section does not fit the program"))
}

pub(super) fn affine(w: &[u32], base: usize) -> [[f32; 4]; 3] {
    let mut out = [[0.0; 4]; 3];
    for (index, slot) in out.iter_mut().flatten().enumerate() {
        *slot = f32::from_bits(w[base + index]);
    }
    out
}

pub(super) fn read_source(w: &[u32]) -> Source {
    let mut min = [0.0; 3];
    let mut max = [0.0; 3];
    for (axis, slot) in min.iter_mut().enumerate() {
        *slot = f32::from_bits(w[4 + axis]);
    }
    for (axis, slot) in max.iter_mut().enumerate() {
        *slot = f32::from_bits(w[7 + axis]);
    }
    Source {
        code_start: w[0],
        code_len: w[1],
        data_start: w[2],
        data_len: w[3],
        min,
        max,
        lipschitz: f32::from_bits(w[10]),
        flags: w[11],
    }
}

pub(super) fn read_material(w: &[u32]) -> Material {
    Material {
        color: [
            f32::from_bits(w[0]),
            f32::from_bits(w[1]),
            f32::from_bits(w[2]),
        ],
        roughness: f32::from_bits(w[3]),
        metallic: f32::from_bits(w[4]),
    }
}

pub(super) fn read_instance(w: &[u32]) -> Instance {
    Instance {
        world_from_local: affine(w, 0),
        local_from_world: affine(w, 12),
        dist_scale: f32::from_bits(w[24]),
        source: w[25],
        material: w[26] as i32,
        key: w[27],
        deform_kind: w[28],
        bend_angle: f32::from_bits(w[29]),
        bend_length: f32::from_bits(w[30]),
        axes: w[31],
        bend: None,
    }
}

pub(super) fn read_bend(w: &[u32]) -> [f64; 8] {
    core::array::from_fn(|i| f64::from_bits(u64::from(w[i * 2]) | (u64::from(w[i * 2 + 1]) << 32)))
}
