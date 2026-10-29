use raa_anim::{Rig, RootMotion};
use raa_anim_graph::{Graph, codes, decode_graph};

const DOC: &str = r#"{"format":"raa.animgraph","version":1,"name":"g","rig":"basic","params":[{"name":"jump","type":"trigger"},{"name":"speed","type":"float","default":0}],"blendSpaces":[],"layers":[{"name":"base","machine":"m","mask":null,"weight":1,"additive":false},{"name":"top","machine":"m","mask":["spine","head"],"weight":0.5,"additive":true}],"machines":[{"name":"m","entry":"idle","states":[{"name":"idle","motion":"idle","loop":true,"speedParam":"speed"},{"name":"jump","motion":"jump","loop":false,"speedParam":null}],"transitions":[{"from":"*","to":"jump","when":[{"param":"jump","op":"trigger"}],"blendTicks":6,"exitTick":null,"priority":0},{"from":"jump","to":"idle","when":[],"blendTicks":4,"exitTick":20,"priority":-1}]}],"rootMotion":{"bone":"root","mode":"xz"}}"#;

#[test]
fn decodes_machines_layers_and_root_motion_in_file_order() {
    let g = decode_graph(DOC).unwrap();
    let (m, t, l) = (&g.machines[0], &g.machines[0].transitions, &g.layers);
    assert_eq!((m.entry.as_str(), t[0].from.as_str()), ("idle", "*"));
    assert_eq!(m.states[0].speed_param.as_deref(), Some("speed"));
    assert_eq!((t[0].exit_tick, t[0].when.len()), (None, 1));
    assert_eq!((t[1].exit_tick, t[1].priority), (Some(20), -1));
    assert_eq!((l[0].mask.as_deref(), l[0].additive), (None, false));
    assert_eq!(l[1].mask, Some(vec!["spine".into(), "head".into()]));
    assert_eq!((l[1].additive, l[1].weight.doc), (true, 0.5));
    assert_eq!(
        (g.root_motion.bone.as_str(), g.root_motion.mode),
        ("root", RootMotion::Xz)
    );
}

#[test]
fn refuses_malformed_declarations() {
    let cases = [
        r#""entry":"idle"|"entry":"walk"|graph.state_unknown|/machines/0/entry"#,
        r#""from":"*"|"from":"run"|graph.state_unknown|/machines/0/transitions/0/from"#,
        r#""to":"jump"|"to":"*"|graph.state_unknown|/machines/0/transitions/0/to"#,
        r#""exitTick":20|"exitTick":-1|graph.schema|/machines/0/transitions/1/exitTick"#,
        r#""exitTick":20|"exitTick":1.5|graph.schema|/machines/0/transitions/1/exitTick"#,
        r#""name":"jump","motion"|"name":"*","motion"|graph.schema|/machines/0/states/1/name"#,
        r#""name":"jump","motion"|"name":"idle","motion"|graph.schema|/machines/0/states/1/name"#,
        r#""speedParam":"speed"|"speedParam":"jump"|graph.param|/machines/0/states/0/speedParam"#,
        r#""machine":"m","mask":null|"machine":"x","mask":null|graph.state_unknown|/layers/0/machine"#,
        r#"["spine","head"]|["spine","spine"]|graph.mask|/layers/1/mask/1"#,
        r#"["spine","head"]|"spine"|graph.schema|/layers/1/mask"#,
        r#""additive":true|"additive":1|graph.schema|/layers/1/additive"#,
        r#""weight":0.5|"weight":2|graph.schema|/layers/1/weight"#,
        r#""mode":"xz"|"mode":"xy"|graph.schema|/rootMotion/mode"#,
        r#""bone":"root"|"bone":""|graph.schema|/rootMotion/bone"#,
        r#""bone":"root"|"bone":"root","extra":1|graph.schema|/rootMotion/extra"#,
    ];
    for case in cases {
        let parts: Vec<&str> = case.split('|').collect();
        let (from, to, code, path) = (parts[0], parts[1], parts[2], parts[3]);
        let error = decode_graph(&DOC.replacen(from, to, 1)).unwrap_err();
        assert_eq!((error.code, error.path.as_str()), (code, path), "{case}");
    }
}

#[test]
fn load_refuses_documents_and_unresolved_references() {
    let rig = Rig::load(include_str!("../../raa-anim/tests/fixtures/rig-basic.json")).unwrap();
    assert_eq!(Graph::load("{}", &rig, &[]).unwrap_err().path, "/format");
    let error = Graph::load(DOC, &rig, &[]).unwrap_err();
    assert_eq!((error.code, error.path.as_str()), (codes::SCHEMA, ""));
}
