use super::fixture::*;
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
