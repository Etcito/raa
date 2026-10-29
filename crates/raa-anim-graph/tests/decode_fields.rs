use raa_anim::parse;
use raa_anim_graph::{CondOp, GRAPH_CAPS, ParamType, codes, decode_conditions, decode_document};

const HEADER: &str = r#"{"format":"raa.animgraph","version":1,"name":"penguin","rig":"penguin","params":PARAMS,"blendSpaces":SPACES,"layers":[],"machines":[],"rootMotion":{"bone":"root","mode":"none"}}"#;

fn doc(params: &str, spaces: &str) -> String {
    HEADER
        .replacen("PARAMS", params, 1)
        .replacen("SPACES", spaces, 1)
}

fn refuse(text: &str) -> (String, String) {
    let error = decode_document(text).unwrap_err();
    (error.code.to_owned(), error.path)
}

#[test]
fn decodes_every_parameter_type_and_source_number() {
    let params = r#"[{"name":"speed","type":"float","default":1.5},{"name":"rank","type":"int","default":2},{"name":"jump","type":"bool","default":true},{"name":"shoot","type":"trigger"}]"#;
    let parts = decode_document(&doc(params, "[]")).unwrap();
    assert_eq!(
        (parts.name.as_str(), parts.rig.as_str()),
        ("penguin", "penguin")
    );
    let types: Vec<ParamType> = parts.params.iter().map(|param| param.ty).collect();
    assert_eq!(
        types,
        [
            ParamType::Float,
            ParamType::Int,
            ParamType::Bool,
            ParamType::Trigger
        ]
    );
    assert_eq!(parts.params[0].default.doc, 1.5);
    assert_eq!(parts.params[0].default.rt, 1.5f32);
    assert_eq!(parts.params[1].default.rt, 2.0);
    assert_eq!(parts.params[2].default.rt, 1.0);
    assert_eq!(parts.params[3].default.rt, 0.0);
    let precise = doc(
        r#"[{"name":"x","type":"float","default":1.0000000000000002}]"#,
        "[]",
    );
    assert_eq!(
        decode_document(&precise).unwrap().params[0].default.doc,
        1.0000000000000002
    );
}

#[test]
fn refuses_malformed_parameter_fields() {
    let cases = [
        (
            r#"[{"name":"a","type":"float"}]"#,
            codes::PARAM,
            "/params/0/default",
        ),
        (
            r#"[{"name":"a","type":"int","default":1.5}]"#,
            codes::PARAM,
            "/params/0/default",
        ),
        (
            r#"[{"name":"a","type":"bool","default":1}]"#,
            codes::PARAM,
            "/params/0/default",
        ),
        (
            r#"[{"name":"a","type":"trigger","default":false}]"#,
            codes::PARAM,
            "/params/0/default",
        ),
        (
            r#"[{"name":"a","type":"tone","default":0}]"#,
            codes::PARAM,
            "/params/0/type",
        ),
        (
            r#"[{"name":"","type":"float","default":0}]"#,
            codes::PARAM,
            "/params/0/name",
        ),
        (
            r#"[{"name":"a","type":"float","default":0},{"name":"a","type":"float","default":0}]"#,
            codes::PARAM,
            "/params/1/name",
        ),
        (
            r#"[{"name":"a","type":"float","default":0,"extra":0}]"#,
            codes::SCHEMA,
            "/params/0/extra",
        ),
    ];
    for (params, code, path) in cases {
        assert_eq!(
            refuse(&doc(params, "[]")),
            (code.to_owned(), path.to_owned()),
            "{params}"
        );
    }
}

#[test]
fn refuses_bad_headers_and_unknown_keys() {
    let cases = [
        (
            r#"{"format":"raa.clip","version":1,"name":"n","rig":"r","params":[],"blendSpaces":[]}"#,
            codes::FORMAT,
            "/format",
        ),
        (
            r#"{"format":"raa.animgraph","version":2,"name":"n","rig":"r","params":[],"blendSpaces":[]}"#,
            codes::VERSION,
            "/version",
        ),
        (
            r#"{"format":"raa.animgraph","version":1,"name":1,"rig":"r","params":[],"blendSpaces":[]}"#,
            codes::SCHEMA,
            "/name",
        ),
        (
            r#"{"format":"raa.animgraph","version":1,"name":"n","rig":"r","params":[],"blendSpaces":[],"extra":0}"#,
            codes::SCHEMA,
            "/extra",
        ),
        ("{", codes::SCHEMA, ""),
    ];
    for (text, code, path) in cases {
        assert_eq!(refuse(text), (code.to_owned(), path.to_owned()), "{text}");
    }
    assert!(
        decode_document(r#"{"format":"raa.animgraph","version":1,"name":"n","rig":"r"}"#).is_err()
    );
}

#[test]
fn decodes_one_and_two_dimensional_blend_spaces() {
    let params = r#"[{"name":"speed","type":"float","default":0},{"name":"strafe","type":"int","default":0}]"#;
    let spaces = r#"[{"name":"locomotion","params":["speed"],"points":[{"clip":"idle","at":[0]},{"clip":"run","at":[4]}],"sync":true},
                     {"name":"strafeSpace","params":["speed","strafe"],"points":[{"clip":"a","at":[0,0]},{"clip":"b","at":[1,2]}]}]"#;
    let parts = decode_document(&doc(params, spaces)).unwrap();
    assert_eq!(parts.blend_spaces.len(), 2);
    let locomotion = &parts.blend_spaces[0];
    assert!(locomotion.sync);
    assert_eq!(locomotion.points[1].clip, "run");
    assert_eq!(locomotion.points[1].at[0].rt, 4.0);
    let strafe = &parts.blend_spaces[1];
    assert!(!strafe.sync);
    assert_eq!(strafe.params, ["speed", "strafe"]);
    assert_eq!(strafe.points[1].at[1].doc, 2.0);
}

#[test]
fn refuses_invalid_blend_dimensions_and_fields() {
    let params = r#"[{"name":"speed","type":"float","default":0},{"name":"flag","type":"bool","default":false}]"#;
    let cases = [
        (
            r#"[{"name":"s","params":[],"points":[{"clip":"a","at":[]}]}]"#,
            codes::BLEND_SPACE,
            "/blendSpaces/0/params",
        ),
        (
            r#"[{"name":"s","params":["speed","speed","speed"],"points":[{"clip":"a","at":[0,0,0]}]}]"#,
            codes::BLEND_SPACE,
            "/blendSpaces/0/params",
        ),
        (
            r#"[{"name":"s","params":["flag"],"points":[{"clip":"a","at":[0]}]}]"#,
            codes::BLEND_SPACE,
            "/blendSpaces/0/params/0",
        ),
        (
            r#"[{"name":"s","params":["missing"],"points":[{"clip":"a","at":[0]}]}]"#,
            codes::BLEND_SPACE,
            "/blendSpaces/0/params/0",
        ),
        (
            r#"[{"name":"s","params":["speed"],"points":[{"clip":"a","at":[0,1]}]}]"#,
            codes::BLEND_SPACE,
            "/blendSpaces/0/points/0/at",
        ),
        (
            r#"[{"name":"s","params":["speed"],"points":[]}]"#,
            codes::BLEND_SPACE,
            "/blendSpaces/0/points",
        ),
        (
            r#"[{"name":"s","params":["speed"],"points":[{"clip":"a","at":[0]}],"extra":0}]"#,
            codes::SCHEMA,
            "/blendSpaces/0/extra",
        ),
        (
            r#"[{"name":"s","params":["speed"],"points":[{"clip":"a","at":[0],"x":0}]}]"#,
            codes::SCHEMA,
            "/blendSpaces/0/points/0/x",
        ),
        (
            r#"[{"name":"s","params":["speed"],"points":[{"clip":"a","at":[0]}]},{"name":"s","params":["speed"],"points":[{"clip":"b","at":[1]}]}]"#,
            codes::BLEND_SPACE,
            "/blendSpaces/1/name",
        ),
    ];
    for (spaces, code, path) in cases {
        assert_eq!(
            refuse(&doc(params, spaces)),
            (code.to_owned(), path.to_owned()),
            "{spaces}"
        );
    }
}

#[test]
fn decodes_and_refuses_typed_conditions() {
    let params = r#"[{"name":"speed","type":"float","default":0},{"name":"jump","type":"trigger"},{"name":"ready","type":"bool","default":false},{"name":"rank","type":"int","default":0}]"#;
    let parts = decode_document(&doc(params, "[]")).unwrap();
    let conditions = |text: &str| decode_conditions(&parse(text).unwrap(), "/when", &parts.params);
    let decoded = conditions(r#"[{"param":"jump","op":"trigger"},{"param":"speed","op":">=","value":2.5},{"param":"ready","op":"==","value":true},{"param":"rank","op":"<","value":3}]"#).unwrap();
    assert_eq!(decoded[0].op, CondOp::Trigger);
    assert_eq!(decoded[1].op, CondOp::Ge);
    assert_eq!(decoded[1].value.rt, 2.5);
    assert_eq!(decoded[2].value.rt, 1.0);
    assert_eq!(decoded[3].value.doc, 3.0);
    let errors = [
        (r#"[{"param":"jump","op":"==","value":1}]"#, "/when/0/op"),
        (r#"[{"param":"speed","op":"trigger"}]"#, "/when/0/op"),
        (r#"[{"param":"ready","op":">","value":true}]"#, "/when/0/op"),
        (
            r#"[{"param":"speed","op":">","value":true}]"#,
            "/when/0/value",
        ),
        (
            r#"[{"param":"rank","op":"<","value":1.5}]"#,
            "/when/0/value",
        ),
        (r#"[{"param":"speed","op":">"}]"#, "/when/0/value"),
        (
            r#"[{"param":"jump","op":"trigger","value":1}]"#,
            "/when/0/value",
        ),
        (r#"[{"param":"nope","op":">","value":1}]"#, "/when/0/param"),
        (r#"[{"param":"speed","op":"~","value":1}]"#, "/when/0/op"),
    ];
    for (text, path) in errors {
        let error = conditions(text).unwrap_err();
        assert_eq!(
            (error.code, error.path.as_str()),
            (codes::PARAM, path),
            "{text}"
        );
    }
    let error = conditions(r#"[{"param":"speed","op":"==","value":1,"x":0}]"#).unwrap_err();
    assert_eq!(
        (error.code, error.path.as_str()),
        (codes::SCHEMA, "/when/0/x")
    );
}

#[test]
fn enforces_graph_caps_inclusively() {
    let params = |n: usize| {
        let items: Vec<String> = (0..n)
            .map(|i| format!(r#"{{"name":"p{i}","type":"float","default":0}}"#))
            .collect();
        format!("[{}]", items.join(","))
    };
    let decoded = decode_document(&doc(&params(GRAPH_CAPS.max_params), "[]")).unwrap();
    assert_eq!(decoded.params.len(), GRAPH_CAPS.max_params);
    let above = params(GRAPH_CAPS.max_params + 1);
    assert_eq!(
        refuse(&doc(&above, "[]")),
        (codes::CAP.to_owned(), "/params".to_owned())
    );

    let space = |name: &str, points: String| {
        format!(r#"{{"name":"{name}","params":["p0"],"points":{points}}}"#)
    };
    let points = |n: usize| {
        let items: Vec<String> = (0..n)
            .map(|i| format!(r#"{{"clip":"a","at":[{i}]}}"#))
            .collect();
        format!("[{}]", items.join(","))
    };
    let full = space("s0", points(GRAPH_CAPS.max_points_per_space));
    let decoded = decode_document(&doc(&params(1), &format!("[{full}]"))).unwrap();
    assert_eq!(
        decoded.blend_spaces[0].points.len(),
        GRAPH_CAPS.max_points_per_space
    );
    let over = space("s0", points(GRAPH_CAPS.max_points_per_space + 1));
    assert_eq!(
        refuse(&doc(&params(1), &format!("[{over}]"))),
        (codes::CAP.to_owned(), "/blendSpaces/0/points".to_owned())
    );

    let spaces = |n: usize| {
        let items: Vec<String> = (0..n).map(|i| space(&format!("s{i}"), points(1))).collect();
        format!("[{}]", items.join(","))
    };
    let decoded = decode_document(&doc(&params(1), &spaces(GRAPH_CAPS.max_blend_spaces))).unwrap();
    assert_eq!(decoded.blend_spaces.len(), GRAPH_CAPS.max_blend_spaces);
    assert_eq!(
        refuse(&doc(&params(1), &spaces(GRAPH_CAPS.max_blend_spaces + 1))),
        (codes::CAP.to_owned(), "/blendSpaces".to_owned())
    );

    let parts = decode_document(&doc(&params(1), "[]")).unwrap();
    let conditions = |n: usize| {
        let items: Vec<String> = (0..n)
            .map(|_| r#"{"param":"p0","op":">","value":1}"#.to_owned())
            .collect();
        format!("[{}]", items.join(","))
    };
    let at_limit = conditions(GRAPH_CAPS.max_conditions_per_transition);
    let decoded = decode_conditions(&parse(&at_limit).unwrap(), "/when", &parts.params).unwrap();
    assert_eq!(decoded.len(), GRAPH_CAPS.max_conditions_per_transition);
    let over = conditions(GRAPH_CAPS.max_conditions_per_transition + 1);
    let error = decode_conditions(&parse(&over).unwrap(), "/when", &parts.params).unwrap_err();
    assert_eq!((error.code, error.path.as_str()), (codes::CAP, "/when"));
}
