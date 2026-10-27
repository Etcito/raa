use raa_anim_js::{ApiError, Case, boundary_u32, dispatch, dispatch_cases, hex8};

#[test]
fn brick_contract_and_case_registration() {
    assert_eq!(raa_anim_js::brick::BRICK_NAME, "raa-anim");
    assert_eq!(raa_anim_js::brick::BRICK_VERSION, "0.5.0");
    let cases = raa_anim_js::registered_cases();
    assert_eq!(cases.iter().filter(|c| c.name == "det-selftest").count(), 1);
    assert_eq!(cases.iter().filter(|c| c.name == "clip-sample").count(), 1);
    let mut names: Vec<_> = cases.iter().map(|c| c.name).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), cases.len(), "case names must be unique");
}

#[test]
fn discovers_every_api_file_in_order() {
    let mut stems: Vec<_> = std::fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/src/api"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "rs"))
        .map(|p| p.file_stem().unwrap().to_str().unwrap().to_owned())
        .collect();
    stems.sort();
    assert!(stems.iter().any(|stem| stem == "clips"));
    assert_eq!(
        include_str!(concat!(env!("OUT_DIR"), "/api_files.txt")),
        format!("{}\n", stems.join("\n"))
    );
    let generated = include_str!(concat!(env!("OUT_DIR"), "/api_registry.rs"));
    let mut previous = 0;
    for stem in stems {
        assert!(generated.contains(&format!("pub mod {stem};")));
        let position = generated
            .find(&format!("{stem}::register(&mut cases);"))
            .unwrap();
        assert!(position > previous);
        previous = position;
    }
}

#[test]
fn clip_sample_dispatch_returns_the_brick_record_shape() {
    for (seed, ticks) in raa_det::DET_CASES {
        // Exhaustive destructuring also checks the native record's exact fields.
        let raa_anim_js::SelftestResult {
            case,
            seed: returned_seed,
            ticks: returned_ticks,
            hash,
        } = dispatch("clip-sample", seed.into(), ticks.into()).unwrap();
        assert_eq!(
            (case, returned_seed, returned_ticks),
            ("clip-sample", seed, ticks)
        );
        assert_eq!(hash.len(), 8);
        assert!(
            hash.bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        );
    }
}

#[test]
fn dispatches_canonical_cases_and_formats_hex8() {
    for (seed, ticks) in raa_det::DET_CASES {
        let result = dispatch("det-selftest", seed.into(), ticks.into()).unwrap();
        assert_eq!(
            (result.case, result.seed, result.ticks),
            ("det-selftest", seed, ticks)
        );
        assert_eq!(
            result.hash,
            format!("{:08x}", raa_det::selftest(seed, ticks))
        );
    }
    assert_eq!(hex8(0), "00000000");
    assert_eq!(hex8(0xabc), "00000abc");
    assert_eq!(hex8(u32::MAX), "ffffffff");
}

#[test]
fn rejects_invalid_boundaries_and_unknown_cases() {
    assert_eq!(
        dispatch("missing", 0.0, 0.0),
        Err(ApiError::UnknownCase("missing".into()))
    );
    for value in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -1.0,
        0.5,
        4294967296.0,
    ] {
        for case in ["det-selftest", "clip-sample"] {
            assert_eq!(
                dispatch(case, value, 0.0),
                Err(ApiError::InvalidInput("seed"))
            );
            assert_eq!(
                dispatch(case, 0.0, value),
                Err(ApiError::InvalidInput("ticks"))
            );
        }
    }
    assert_eq!(boundary_u32(u32::MAX.into(), "ticks"), Ok(u32::MAX));
    assert_eq!(boundary_u32(-0.0, "seed"), Ok(0));
}

#[test]
fn additive_registration_needs_no_dispatch_change() {
    let mut cases = raa_anim_js::registered_cases();
    cases.push(Case {
        name: "extra",
        run: |seed, ticks| seed ^ ticks,
    });
    assert_eq!(
        dispatch_cases(&cases, "extra", 1.0, 3.0).unwrap().hash,
        "00000002"
    );
    assert!(dispatch_cases(&cases, "det-selftest", 0.0, 0.0).is_ok());
}
