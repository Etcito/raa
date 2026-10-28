use raa_anim_graph::{decode_graph, to_canonical};

const DOC: &str = r#"{"format":"raa.animgraph","version":1,"name":"g","rig":"basic","params":[{"name":"jump","type":"trigger"},{"name":"speed","type":"float","default":1e-7},{"name":"on","type":"bool","default":true},{"name":"n","type":"int","default":-0}],"blendSpaces":[{"name":"loco","params":["speed"],"points":[{"clip":"idle","at":[0]},{"clip":"run","at":[4.50]}],"sync":true}],"layers":[{"name":"base","machine":"m","mask":null,"weight":1,"additive":false},{"name":"top","machine":"m","mask":["spine","head"],"weight":0.5,"additive":true}],"machines":[{"name":"m","entry":"idle","states":[{"name":"idle","motion":"loco","loop":true,"speedParam":"speed"},{"name":"jump","motion":"jump","loop":false,"speedParam":null}],"transitions":[{"from":"jump","to":"idle","when":[{"param":"on","op":"==","value":false},{"param":"speed","op":">=","value":1.5}],"blendTicks":4,"exitTick":20,"priority":-1},{"from":"*","to":"jump","when":[{"param":"jump","op":"trigger"}],"blendTicks":6,"exitTick":null,"priority":0}]}],"rootMotion":{"bone":"root","mode":"xz"}}"#;

#[test]
fn writes_s19_order_and_s4_numbers_idempotently() {
    let text = to_canonical(&decode_graph(DOC).unwrap());
    assert!(text.ends_with("}\n") && !text.ends_with("\n\n"));
    assert!(text.starts_with("{\n  \"format\": \"raa.animgraph\",\n  \"version\": 1,\n"));
    for needle in ["1e-7", "\"default\": true", "\"default\": 0", "4.5", "\"value\": false"] {
        assert!(text.contains(needle), "{needle}\n{text}");
    }
    assert!(!text.contains("\"type\": \"trigger\",\n      \"default\""));
    let keys = ["\"params\"", "\"blendSpaces\"", "\"layers\"", "\"machines\"", "\"rootMotion\""];
    let at: Vec<usize> = keys.iter().map(|k| text.find(k).unwrap()).collect();
    assert!(at.windows(2).all(|w| w[0] < w[1]));
    let back = decode_graph(&text).unwrap();
    assert_eq!(to_canonical(&back), text);
    assert_eq!(back.canonical(), text);
}

#[test]
fn preserves_transition_and_layer_file_order() {
    let g = decode_graph(DOC).unwrap();
    let text = g.canonical();
    assert!(text.find("\"from\": \"jump\"").unwrap() < text.find("\"from\": \"*\"").unwrap());
    assert!(text.find("\"name\": \"base\"").unwrap() < text.find("\"name\": \"top\"").unwrap());
    assert_eq!(decode_graph(&text).unwrap(), g);
}
