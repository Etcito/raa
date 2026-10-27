use raa_anim::{MAX_BONES, Rig, rig::ChainSolver};

const BASIC: &str = include_str!("fixtures/rig-basic.json");
const REST: &str = r#""rest":{"t":[0,0,0],"r":[0,0,0,1],"s":[1,1,1]}"#;
const META: &str = concat!(
    r#","roles":{"root":"r","head":"b"},"#,
    r#""ikChains":[{"name":"arm","solver":"twoBone","bones":["r","a","b"],"pole":[0,0,1],"iterations":16}],"#,
    r#""limits":[{"bone":"b","min":[-10,-20,-30],"max":[10,20,30]}],"#,
    r#""attach":{"beak":{"bone":"b","t":[0,0.1,0],"r":[0,0.6,0,0.8]}}"#
);

fn bone(name: &str, parent: &str) -> String {
    format!(r#"{{"name":"{name}","parent":{parent},{REST}}}"#)
}
fn document(bones: &[String]) -> String {
    format!(
        r#"{{"format":"raa.rig","version":1,"name":"test","bones":[{}]}}"#,
        bones.join(",")
    )
}
fn meta_document() -> String {
    format!(
        r#"{{"format":"raa.rig","version":1,"name":"meta","bones":[{},{},{}]{}}}"#,
        bone("r", "null"),
        bone("a", "\"r\""),
        bone("b", "\"a\""),
        META
    )
}
fn refusal(text: &str, code: &str, path: &str) {
    let error = Rig::load(text).unwrap_err();
    assert_eq!((error.code, error.path.as_str()), (code, path), "{error}");
}

#[test]
fn rig_basic() {
    let rig = Rig::load(BASIC).unwrap();
    assert_eq!(rig.name, "rig-basic");
    assert_eq!(rig.bones.len(), 2);
    assert_eq!(rig.bones[0].parent, None);
    assert_eq!(rig.bones[1].parent, Some(0));
    assert_eq!(rig.bones[0].rest.t, [1.0, 2.0, 3.0]);
    assert_eq!(rig.bones[0].rest.r, [0.0, 0.0, 0.6, 0.8]);
    assert_eq!(rig.bones[1].rest.s, [1.0, 1.5, 1.0]);
    assert_eq!(rig.roles["head"], 1);
    assert_eq!(rig.ik_chains[0].solver, ChainSolver::Ccd);
    assert_eq!(rig.ik_chains[0].bones, [0, 1]);
    assert_eq!(rig.ik_chains[0].iterations, 8);
    assert_eq!(rig.limits[0].max, [40.0, 70.0, 30.0]);
    assert_eq!(rig.attach["beak"].bone, 1);
}

#[test]
fn rig_bone_count_boundaries() {
    assert_eq!(MAX_BONES, 256);
    // A full-depth valid chain also exercises the largest permitted hierarchy.
    let bones: Vec<_> = (0..257)
        .map(|i| {
            bone(
                &format!("b{i}"),
                &if i == 0 {
                    "null".into()
                } else {
                    format!("\"b{}\"", i - 1)
                },
            )
        })
        .collect();
    let rig = Rig::load(&document(&bones[..256])).unwrap();
    assert_eq!(rig.bones.len(), 256);
    assert_eq!(rig.bones[255].parent, Some(254));
    refusal(&document(&bones), "rig.cap_bones", "/bones");
}

#[test]
fn rig_names() {
    for name in [
        "",
        "space name",
        "slash/name",
        "é",
        "abcdefghijklmnopqrstuvwxyz1234567",
    ] {
        refusal(
            &document(&[bone(name, "null")]),
            "rig.bone_name",
            "/bones/0/name",
        );
    }
    Rig::load(&document(&[bone(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ_.-012",
        "null",
    )]))
    .unwrap();
    refusal(
        &document(&[bone("root", "null"), bone("root", "\"root\"")]),
        "rig.bone_duplicate",
        "/bones/1/name",
    );
}

#[test]
fn rig_hierarchy() {
    refusal(&document(&[]), "rig.root", "/bones");
    refusal(
        &document(&[bone("a", "null"), bone("b", "null")]),
        "rig.root",
        "/bones",
    );
    for parent in ["\"missing\"", "\"a\"", "\"b\""] {
        refusal(
            &document(&[bone("a", parent), bone("b", "\"a\"")]),
            "rig.parent_order",
            "/bones/0/parent",
        );
    }
    refusal(
        &document(&[bone("a", "false")]),
        "rig.schema",
        "/bones/0/parent",
    );
    // Siblings need not be adjacent to their parent.
    let rig = Rig::load(&document(&[
        bone("a", "null"),
        bone("b", "\"a\""),
        bone("c", "\"b\""),
        bone("d", "\"a\""),
    ]))
    .unwrap();
    assert_eq!(rig.bones[3].parent, Some(0));
}

#[test]
fn rig_transform_validation() {
    let valid = document(&[bone("root", "null")]);
    for (from, to, path) in [
        ("[0,0,0]", "[0,0]", "/bones/0/rest/t"),
        ("[0,0,0]", "[0,true,0]", "/bones/0/rest/t/1"),
        ("[0,0,0,1]", "[0,0,0,0]", "/bones/0/rest/r"),
        ("[0,0,0,1]", "[0,0,0,1.002]", "/bones/0/rest/r"),
        ("[0,0,0,1]", "[1e308,0,0,1]", "/bones/0/rest/r"),
        ("[1,1,1]", "[1,0,1]", "/bones/0/rest/s/1"),
        ("[1,1,1]", "[-1,1,1]", "/bones/0/rest/s/0"),
        ("[1,1,1]", "[1,1,-0]", "/bones/0/rest/s/2"),
    ] {
        refusal(&valid.replace(from, to), "rig.schema", path);
    }
    refusal(
        &valid.replace("[0,0,0]", "[1e400,0,0]"),
        "rig.not_finite",
        "/bones/0/rest/t/0",
    );
    let rig = Rig::load(
        &valid
            .replace("[0,0,0,1]", "[0,0,0,-1.0005]")
            .replace("[0,0,0]", "[0.12345678901234567,0,0]"),
    )
    .unwrap();
    assert_eq!(rig.bones[0].rest.r[3], -1.0005);
    assert_eq!(rig.bones[0].rest.t[0], 0.12345678901234567);
}

#[test]
fn rig_schema() {
    let valid = document(&[bone("root", "null")]);
    for (from, to, code, path) in [
        ("raa.rig", "other", "rig.format", "/format"),
        ("\"version\":1", "\"version\":2", "rig.version", "/version"),
        ("\"name\":\"test\",", "", "rig.schema", "/name"),
        ("\"parent\":null,", "", "rig.schema", "/bones/0/parent"),
        ("\"t\":[0,0,0],", "", "rig.schema", "/bones/0/rest/t"),
        (
            "\"rest\":",
            "\"x/y~\":0,\"rest\":",
            "rig.schema",
            "/bones/0/x~1y~0",
        ),
        (
            "\"t\":",
            "\"extra\":0,\"t\":",
            "rig.schema",
            "/bones/0/rest/extra",
        ),
        (
            "\"format\":",
            "\"ragdoll\":{},\"format\":",
            "rig.schema",
            "/ragdoll",
        ),
        (
            "\"parent\":null",
            "\"parent\":null,\"parent\":null",
            "rig.schema",
            "/bones/0/parent",
        ),
    ] {
        refusal(&valid.replace(from, to), code, path);
    }
}

#[test]
fn rig_metadata() {
    let text = meta_document();
    let rig = Rig::load(&text).unwrap();
    assert_eq!(rig.roles["head"], 2);
    assert_eq!(rig.ik_chains[0].solver, ChainSolver::TwoBone);
    assert_eq!(rig.ik_chains[0].bones, [0, 1, 2]);
    assert_eq!(rig.ik_chains[0].pole, Some([0.0, 0.0, 1.0]));
    assert_eq!(rig.ik_chains[0].iterations, 16);
    assert_eq!(rig.limits[0].min, [-10.0, -20.0, -30.0]);
    assert_eq!(rig.attach["beak"].bone, 2);

    for (from, to, code) in [
        (r#""head":"b""#, r#""snout":"b""#, "rig.role_unknown"),
        (r#""head":"b""#, r#""head":"nope""#, "rig.role_unknown"),
        (r#""twoBone""#, r#""rope""#, "rig.chain"),
        (r#"["r","a","b"]"#, r#"["r","a"]"#, "rig.chain"),
        (r#"["r","a","b"]"#, r#"["a","r","b"]"#, "rig.chain"),
        (r#""pole":[0,0,1]"#, r#""pole":[0,0,2]"#, "rig.chain"),
        (r#""iterations":16"#, r#""iterations":17"#, "rig.chain"),
        (r#"[-10,-20,-30]"#, r#"[-10,30,-30]"#, "rig.limit"),
        (r#""bone":"b","t""#, r#""bone":"nope","t""#, "rig.attach"),
        (r#"[0,0.6,0,0.8]"#, r#"[0,0,0,0]"#, "rig.attach"),
    ] {
        let error = Rig::load(&text.replace(from, to)).unwrap_err();
        assert_eq!(error.code, code, "{error}");
    }
    let mut ragdoll = text.clone();
    ragdoll.insert_str(ragdoll.len() - 1, r#","ragdoll":{}"#);
    assert_eq!(Rig::load(&ragdoll).unwrap_err().code, "rig.schema");

    for (fixture, code) in [
        (
            include_str!("fixtures/invalid-rig-unknown-field.json"),
            "rig.schema",
        ),
        (
            include_str!("fixtures/invalid-rig-duplicate-bone.json"),
            "rig.bone_duplicate",
        ),
        (
            include_str!("fixtures/invalid-rig-parent-order.json"),
            "rig.parent_order",
        ),
        (
            include_str!("fixtures/invalid-rig-role.json"),
            "rig.role_unknown",
        ),
        (include_str!("fixtures/invalid-rig-chain.json"), "rig.chain"),
        (
            include_str!("fixtures/invalid-rig-attach.json"),
            "rig.attach",
        ),
    ] {
        let error = Rig::load(fixture).unwrap_err();
        assert_eq!(error.code, code, "{error}");
    }
}
