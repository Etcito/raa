//! Section 67 reader: a valid program, truncation, bad references and malformed instructions.

use raa_sdf_trace::{
    DATA, FORMAT, INSTANCE_COUNT, MAGIC, NOT_FINITE, OPCODE, REFERENCE, SOURCE, STACK,
    TraceProgram, TraceScene, VERSION, VERSION_ERROR, opcode, parse, target_size,
};

const IDENTITY: [[f32; 4]; 3] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];

fn source(code_start: u32, code_len: u32, data_start: u32, data_len: u32) -> Vec<u32> {
    let mut w = vec![code_start, code_len, data_start, data_len];
    w.extend([-1.0f32, -1.0, -1.0].map(f32::to_bits));
    w.extend([1.0f32, 1.0, 1.0].map(f32::to_bits));
    w.push(1.0f32.to_bits()); // lipschitz
    w.push(0); // flags
    w
}

fn sphere_data() -> Vec<f32> {
    let mut d: Vec<f32> = IDENTITY.iter().flatten().copied().collect();
    d.push(1.0); // distance scale
    d.push(0.0); // material index
    d.push(0.5); // radius
    d
}

fn instance(source: u32, material: u32) -> Vec<u32> {
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

fn material(color: [f32; 3], roughness: f32, metallic: f32) -> Vec<u32> {
    let w: Vec<u32> = color
        .iter()
        .chain([roughness, metallic, 0.0, 0.0, 0.0].iter())
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(w.len(), 8);
    w
}

struct Fixture {
    sources: Vec<Vec<u32>>,
    code: Vec<u32>,
    data: Vec<f32>,
    materials: Vec<Vec<u32>>,
    instances: Vec<Vec<u32>>,
}

impl Fixture {
    fn minimal() -> Fixture {
        Fixture {
            sources: vec![source(0, 1, 0, 15)],
            code: vec![opcode::SPHERE],
            data: sphere_data(),
            materials: vec![material([0.8, 0.2, 0.1], 0.5, 0.0)],
            instances: vec![instance(0, u32::MAX)],
        }
    }

    fn words(&self) -> Vec<u32> {
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

    fn bytes(&self) -> Vec<u8> {
        self.words().iter().flat_map(|w| w.to_le_bytes()).collect()
    }
}

#[test]
fn parses_a_minimal_program() {
    let bytes = Fixture::minimal().bytes();
    let program = parse(&bytes).expect("the fixture is valid");
    assert_eq!(program.sources.len(), 1);
    assert_eq!(program.instances.len(), 1);
    assert_eq!(program.materials.len(), 1);
    assert_eq!(program.sources[0].code(&program), &[opcode::SPHERE]);
    assert_eq!(program.sources[0].data(&program)[14], 0.5);
    assert_eq!(program.instances[0].key, 0x1234_5678);
    assert_eq!(program.instances[0].material, -1);
    assert_eq!(program.instances[0].world_from_local, IDENTITY);
    assert_eq!(program.materials[0].color, [0.8, 0.2, 0.1]);
    assert!(!program.sources[0].placeholder());
    assert!(TraceProgram::parse(&bytes).is_ok());
}

#[test]
fn parses_a_union_stack_and_keeps_the_placeholder_flag() {
    let mut f = Fixture::minimal();
    let mut data = sphere_data();
    data.extend(sphere_data());
    data.push(0.1); // smooth combine factor
    f.sources[0] = source(0, 3, 0, data.len() as u32);
    f.code = vec![
        opcode::SPHERE,
        opcode::SPHERE | (15 << 8),
        opcode::UNION | (30 << 8),
    ];
    f.data = data;
    f.sources[0][11] = 1; // placeholder
    let program = parse(&f.bytes()).expect("two primitives and a union are valid");
    assert!(program.sources[0].placeholder());
    assert_eq!(program.sources[0].code(&program).len(), 3);
}

#[test]
fn parses_source_relative_data_offsets() {
    let mut f = Fixture::minimal();
    let mut data = sphere_data();
    data.extend(sphere_data());
    f.sources = vec![source(0, 1, 0, 15), source(1, 1, 15, 15)];
    f.code = vec![opcode::SPHERE, opcode::SPHERE];
    f.data = data;
    f.instances = vec![instance(0, u32::MAX), instance(1, 0)];
    let program = parse(&f.bytes()).expect("each source reads its own data span");
    assert_eq!(program.sources[1].data(&program)[14], 0.5);
}

#[test]
fn refuses_a_truncated_program() {
    let mut bytes = Fixture::minimal().bytes();
    bytes.truncate(bytes.len() - 4);
    let err = parse(&bytes).expect_err("a missing word is a truncation");
    assert_eq!(err.code(), FORMAT);
    assert!(err.message().contains("truncated"), "{}", err.message());

    let err = parse(&bytes[..16]).expect_err("half a header is a truncation");
    assert_eq!(err.code(), FORMAT);
}

#[test]
fn refuses_extra_words() {
    let mut bytes = Fixture::minimal().bytes();
    bytes.extend(0u32.to_le_bytes());
    let err = parse(&bytes).expect_err("an extra word is not part of the header counts");
    assert_eq!(err.code(), FORMAT);
    assert!(err.message().contains("extra"), "{}", err.message());
}

#[test]
fn refuses_a_foreign_magic_and_a_newer_version() {
    let mut bytes = Fixture::minimal().bytes();
    bytes[..4].copy_from_slice(b"XXXX");
    assert_eq!(parse(&bytes).expect_err("bad magic").code(), FORMAT);

    let mut bytes = Fixture::minimal().bytes();
    bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
    assert_eq!(
        parse(&bytes).expect_err("bad version").code(),
        VERSION_ERROR
    );
}

#[test]
fn refuses_an_unknown_opcode() {
    let mut f = Fixture::minimal();
    f.code[0] = 99;
    let err = parse(&f.bytes()).expect_err("99 is not an opcode");
    assert_eq!(err.code(), OPCODE);
}

#[test]
fn refuses_an_instruction_past_the_source_data() {
    let mut f = Fixture::minimal();
    f.code[0] = opcode::SPHERE | (1 << 8);
    let err = parse(&f.bytes()).expect_err("one word of data is missing");
    assert_eq!(err.code(), DATA);
}

#[test]
fn refuses_a_combination_below_the_stack() {
    let mut f = Fixture::minimal();
    f.code[0] = opcode::UNION;
    assert_eq!(
        parse(&f.bytes())
            .expect_err("a union needs two operands")
            .code(),
        STACK
    );
}

#[test]
fn refuses_an_unbalanced_stack() {
    let mut f = Fixture::minimal();
    let mut data = sphere_data();
    data.extend(sphere_data());
    f.sources[0] = source(0, 2, 0, data.len() as u32);
    f.code = vec![opcode::SPHERE, opcode::SPHERE | (15 << 8)];
    f.data = data;
    let err = parse(&f.bytes()).expect_err("two values are left on the stack");
    assert_eq!(err.code(), STACK);
}

#[test]
fn refuses_a_source_reference_past_the_sources() {
    let mut f = Fixture::minimal();
    f.instances[0] = instance(1, u32::MAX);
    let err = parse(&f.bytes()).expect_err("there is one source");
    assert_eq!(err.code(), REFERENCE);
}

#[test]
fn refuses_a_material_reference_past_the_materials() {
    let mut f = Fixture::minimal();
    f.instances[0] = instance(0, 1);
    assert_eq!(
        parse(&f.bytes()).expect_err("override 1 is missing").code(),
        REFERENCE
    );

    let mut f = Fixture::minimal();
    f.data[13] = 1.0;
    assert_eq!(
        parse(&f.bytes())
            .expect_err("primitive material 1 is missing")
            .code(),
        REFERENCE
    );
}

#[test]
fn refuses_non_finite_geometry() {
    let mut f = Fixture::minimal();
    f.instances[0][0] = f32::NAN.to_bits();
    assert_eq!(
        parse(&f.bytes()).expect_err("a NaN matrix").code(),
        NOT_FINITE
    );

    let mut f = Fixture::minimal();
    f.data[14] = f32::INFINITY;
    assert_eq!(
        parse(&f.bytes()).expect_err("an infinite radius").code(),
        NOT_FINITE
    );
}

#[test]
fn refuses_bad_source_records() {
    let mut f = Fixture::minimal();
    f.sources[0][1] = 2; // code length past the code section
    assert_eq!(parse(&f.bytes()).expect_err("code span").code(), SOURCE);

    let mut f = Fixture::minimal();
    f.sources[0][4] = 2.0f32.to_bits(); // min x above max x
    assert_eq!(
        parse(&f.bytes()).expect_err("inverted bounds").code(),
        SOURCE
    );

    let mut f = Fixture::minimal();
    f.sources[0][10] = 0.0f32.to_bits(); // lipschitz
    assert_eq!(
        parse(&f.bytes()).expect_err("zero lipschitz").code(),
        SOURCE
    );
}

#[test]
fn refuses_an_unknown_deform_kind() {
    let mut f = Fixture::minimal();
    f.instances[0][28] = 2;
    assert_eq!(
        parse(&f.bytes()).expect_err("kind 2 is unknown").code(),
        FORMAT
    );
}

#[test]
fn keeps_bend_bits_and_refuses_an_incomplete_bend() {
    let mut f = Fixture::minimal();
    f.instances[0][28] = 1; // bend
    f.instances[0][29] = 0.5f32.to_bits();
    f.instances[0][30] = 2.0f32.to_bits();
    f.instances[0][31] = 40; // +X along, -Z toward
    assert_eq!(
        parse(&f.bytes()).expect_err("missing bend record").code(),
        FORMAT
    );
    let bytes = with_bend_record(&f);
    let program = parse(&bytes).expect("a complete bend record");
    let bend = &program.instances[0];
    assert_eq!((bend.deform_kind, bend.axes), (1, 40));
    assert_eq!(bend.bend_angle, 0.5);

    for axes in [7, ((6 << 3) | 1), (1 << 20)] {
        let mut f = Fixture::minimal();
        f.instances[0][28] = 1;
        f.instances[0][30] = 2.0f32.to_bits();
        f.instances[0][31] = axes;
        let err = parse(&with_bend_record(&f)).expect_err("an incomplete bend frame");
        assert_eq!(err.code(), FORMAT, "axes {axes}");
    }

    let mut f = Fixture::minimal();
    f.instances[0][28] = 1;
    f.instances[0][30] = 0.0f32.to_bits();
    f.instances[0][31] = 40;
    assert_eq!(
        parse(&with_bend_record(&f))
            .expect_err("zero length")
            .code(),
        FORMAT
    );
}

fn with_bend_record(f: &Fixture) -> Vec<u8> {
    let mut words = f.words();
    words[7] = 1;
    for value in [0.0f64, 30.0, 0.0, -0.5, 0.5, 0.0, 0.0, 0.0] {
        let bits = value.to_bits();
        words.extend([bits as u32, (bits >> 32) as u32]);
    }
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn instance_floats(words: Vec<u32>) -> Vec<f32> {
    words.into_iter().map(f32::from_bits).collect()
}

#[test]
fn target_size_keeps_the_full_render_target() {
    assert_eq!(target_size(2048, 1024), (2048, 1024), "landscape");
    assert_eq!(target_size(1024, 2048), (1024, 2048), "portrait");
    assert_eq!(target_size(800, 600), (800, 600), "already small");
    assert_eq!(
        target_size(3000, 1000),
        (3000, 1000),
        "above the volume cap"
    );
    assert_eq!(target_size(0, 0), (1, 1));

    let mut scene = TraceScene::default();
    scene
        .set_program(&Fixture::minimal().bytes())
        .expect("valid");
    let stats = scene.stats(target_size(2048, 1024));
    assert_eq!(
        (stats.sources, stats.instances, stats.placeholders),
        (1, 1, 0)
    );
    assert_eq!((stats.width, stats.height), (2048, 1024));
}

#[test]
fn keeps_the_working_program_when_a_replacement_is_refused() {
    let mut scene = TraceScene::default();
    scene
        .set_program(&Fixture::minimal().bytes())
        .expect("valid");
    let program = scene.program.clone().expect("a program");
    let bvh = scene.bvh.clone();

    let mut bad = Fixture::minimal().bytes();
    bad[..4].copy_from_slice(b"XXXX");
    assert_eq!(
        scene.set_program(&bad).expect_err("bad magic").code(),
        FORMAT
    );
    let mut bad = Fixture::minimal().bytes();
    bad.truncate(bad.len() - 4);
    assert_eq!(
        scene.set_program(&bad).expect_err("truncated").code(),
        FORMAT
    );
    assert_eq!(scene.program.as_ref(), Some(&program));
    assert_eq!(&scene.bvh, &bvh);
}

#[test]
fn refuses_an_instance_update_with_the_wrong_count() {
    let mut scene = TraceScene::default();
    assert_eq!(
        scene.update_instances(&[]).expect_err("no program").code(),
        INSTANCE_COUNT
    );
    scene
        .set_program(&Fixture::minimal().bytes())
        .expect("valid");
    let program = scene.program.clone().expect("a program");
    let bvh = scene.bvh.clone();

    let one = instance_floats(instance(0, u32::MAX));
    let two = [one.clone(), one.clone()].concat();
    for words in [Vec::new(), two, one[..one.len() - 1].to_vec()] {
        assert_eq!(
            scene
                .update_instances(&words)
                .expect_err("wrong count")
                .code(),
            INSTANCE_COUNT
        );
    }
    assert_eq!(scene.program.as_ref(), Some(&program));
    assert_eq!(&scene.bvh, &bvh);
}

#[test]
fn update_rebuilds_the_bvh_and_clear_is_safe() {
    let mut scene = TraceScene::default();
    scene
        .set_program(&Fixture::minimal().bytes())
        .expect("valid");
    let before = scene.bvh.nodes[0].min[0];

    let mut moved = instance(0, u32::MAX);
    moved[3] = 10.0f32.to_bits();
    scene
        .update_instances(&instance_floats(moved))
        .expect("same count");
    assert_eq!(
        scene.program.as_ref().expect("a program").instances[0].world_from_local[0][3],
        10.0
    );
    assert!(scene.bvh.nodes[0].min[0] > before);

    scene.clear();
    assert!(scene.program.is_none() && scene.bvh.nodes.is_empty());
    scene.clear();
    assert_eq!(
        scene.update_instances(&[]).expect_err("cleared").code(),
        INSTANCE_COUNT
    );
    scene
        .set_program(&Fixture::minimal().bytes())
        .expect("loaded again");
}

#[test]
fn update_rejects_a_bend_without_a_record_atomically() {
    let mut scene = TraceScene::default();
    scene.set_program(&Fixture::minimal().bytes()).unwrap();
    let program = scene.program.clone();
    let bvh = scene.bvh.clone();
    let mut bent = instance(0, u32::MAX);
    bent[28] = 1;
    bent[29] = 0.5f32.to_bits();
    bent[30] = 2.0f32.to_bits();
    bent[31] = 40;
    assert_eq!(
        scene
            .update_instances(&instance_floats(bent))
            .unwrap_err()
            .code(),
        FORMAT
    );
    assert_eq!(scene.program, program);
    assert_eq!(scene.bvh, bvh);
}
