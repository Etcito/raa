#![allow(unused_imports, dead_code)]
//! Section 67 reader: a valid program, truncation, bad references and malformed instructions.

pub(super) use raa_sdf_trace::{
    DATA, FORMAT, INSTANCE_COUNT, MAGIC, NOT_FINITE, OPCODE, REFERENCE, SOURCE, STACK,
    TraceProgram, TraceScene, VERSION, VERSION_ERROR, VOLUME_AXIS_ALIGN, VOLUME_MAX_AXIS,
    VOLUME_MIPS, VOLUME_ROW_ALIGN, VOLUME_UNIFORM_BYTES, VolumeLayout, opcode, parse, target_size,
};

pub(super) const IDENTITY: [[f32; 4]; 3] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];

pub(super) fn source(code_start: u32, code_len: u32, data_start: u32, data_len: u32) -> Vec<u32> {
    let mut w = vec![code_start, code_len, data_start, data_len];
    w.extend([-1.0f32, -1.0, -1.0].map(f32::to_bits));
    w.extend([1.0f32, 1.0, 1.0].map(f32::to_bits));
    w.push(1.0f32.to_bits()); // lipschitz
    w.push(0); // flags
    w
}

pub(super) fn sphere_data() -> Vec<f32> {
    let mut d: Vec<f32> = IDENTITY.iter().flatten().copied().collect();
    d.push(1.0); // distance scale
    d.push(0.0); // material index
    d.push(0.5); // radius
    d
}

pub(super) fn instance(source: u32, material: u32) -> Vec<u32> {
    let mut w: Vec<u32> = IDENTITY.iter().flatten().map(|v| v.to_bits()).collect();
    w.extend(IDENTITY.iter().flatten().map(|v| v.to_bits()));
    w.push(1.0f32.to_bits()); // distance scale
    w.push(source);
    w.push(material);
    w.push(0x1234_5678); // key
    w.push(0); // deform kind
    w.push(0); // bend angle
    w.push(0); // bend length
    w.push(0); // packed axes
    assert_eq!(w.len(), 32);
    w
}

pub(super) fn material(color: [f32; 3], roughness: f32, metallic: f32) -> Vec<u32> {
    let w: Vec<u32> = color
        .iter()
        .chain([roughness, metallic, 0.0, 0.0, 0.0].iter())
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(w.len(), 8);
    w
}

pub(super) struct Fixture {
    pub(super) sources: Vec<Vec<u32>>,
    pub(super) code: Vec<u32>,
    pub(super) data: Vec<f32>,
    pub(super) materials: Vec<Vec<u32>>,
    pub(super) instances: Vec<Vec<u32>>,
}

impl Fixture {
    pub(super) fn minimal() -> Fixture {
        Fixture {
            sources: vec![source(0, 1, 0, 15)],
            code: vec![opcode::SPHERE],
            data: sphere_data(),
            materials: vec![material([0.8, 0.2, 0.1], 0.5, 0.0)],
            instances: vec![instance(0, u32::MAX)],
        }
    }

    pub(super) fn words(&self) -> Vec<u32> {
        let mut w = vec![
            MAGIC,
            VERSION,
            self.sources.len() as u32,
            self.instances.len() as u32,
            self.code.len() as u32,
            self.data.len() as u32,
            self.materials.len() as u32,
            0,
        ];
        for source in &self.sources {
            w.extend(source);
        }
        w.extend(&self.code);
        w.extend(self.data.iter().map(|v| v.to_bits()));
        for material in &self.materials {
            w.extend(material);
        }
        for instance in &self.instances {
            w.extend(instance);
        }
        w
    }

    pub(super) fn bytes(&self) -> Vec<u8> {
        self.words().iter().flat_map(|w| w.to_le_bytes()).collect()
    }
}

pub(super) fn instance_floats(words: Vec<u32>) -> Vec<f32> {
    words.into_iter().map(f32::from_bits).collect()
}
