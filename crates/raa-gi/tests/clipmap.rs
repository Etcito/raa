//! The hand-derived clipmap cases of S37 against `crates/raa-gi/src/clipmap.rs`. The fixture
//! `tests/fixtures/clipmap.json` carries each expected value together with the arithmetic that
//! derives it in `why`; the Rust-only checks at the end need no fixture.

use raa_core::MeshId;
use raa_gi::GiQuality;
use raa_gi::clipmap::{
    RebuildRounds, RelightCursor, choose_rebuild, flood_pass_count, flood_steps, forward_lod,
    level_view, light_schedule, snapped_origin, volume_bytes, voxel_size,
};
use raa_lit::LodLevel;
use raa_loader::json::{JsonValue, parse};
use raa_math::Vec3;
use raa_vg::InstanceFilter;

fn fixture() -> JsonValue {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/clipmap.json");
    let text = std::fs::read_to_string(path).expect("read the clipmap fixture");
    parse(&text).expect("parse the clipmap fixture")
}

fn field<'a>(value: &'a JsonValue, name: &str) -> &'a JsonValue {
    let JsonValue::Object(fields) = value else {
        panic!("{name}: expected an object");
    };
    let found = fields.iter().find(|(key, _)| key == name);
    &found.unwrap_or_else(|| panic!("missing field `{name}`")).1
}

fn array(value: &JsonValue) -> &[JsonValue] {
    let JsonValue::Array(items) = value else {
        panic!("expected an array");
    };
    items
}

fn number(value: &JsonValue) -> f64 {
    let JsonValue::Number(number) = value else {
        panic!("expected a number");
    };
    *number
}

fn integer(value: &JsonValue) -> i64 {
    let number = number(value);
    assert_eq!(number.fract(), 0.0, "expected an integer, got {number}");
    number as i64
}

fn boolean(value: &JsonValue) -> bool {
    let JsonValue::Bool(boolean) = value else {
        panic!("expected a boolean");
    };
    *boolean
}

fn string(value: &JsonValue) -> &str {
    let JsonValue::String(text) = value else {
        panic!("expected a string");
    };
    text
}

fn vec3(value: &JsonValue) -> Vec3 {
    let items = array(value);
    Vec3::from_array(std::array::from_fn(|i| number(&items[i]) as f32))
}

fn i32_3(value: &JsonValue) -> [i32; 3] {
    let items = array(value);
    std::array::from_fn(|i| integer(&items[i]) as i32)
}

fn u32_of(value: &JsonValue) -> u32 {
    integer(value) as u32
}

fn case_why(case: &JsonValue) -> &str {
    string(field(case, "why"))
}

#[test]
fn snapped_origin_matches_the_fixture() {
    let fixture = fixture();
    for case in array(field(&fixture, "snapped_origin")) {
        let got = snapped_origin(
            vec3(field(case, "position")),
            number(field(case, "voxel")) as f32,
            u32_of(field(case, "resolution")),
        );
        assert_eq!(got, i32_3(field(case, "expected")), "{}", case_why(case));
    }
}

fn optional_origin(value: &JsonValue) -> Option<[i32; 3]> {
    match value {
        JsonValue::Null => None,
        _ => Some(i32_3(value)),
    }
}

fn rebuild_expectation(value: &JsonValue) -> Option<(usize, [i32; 3])> {
    match value {
        JsonValue::Null => None,
        _ => Some((
            u32_of(field(value, "level")) as usize,
            i32_3(field(value, "origin")),
        )),
    }
}

#[test]
fn choose_rebuild_matches_the_fixture() {
    let fixture = fixture();
    for case in array(field(&fixture, "choose_rebuild")) {
        let origins: Vec<Option<[i32; 3]>> = array(field(case, "origins"))
            .iter()
            .map(optional_origin)
            .collect();
        let got = choose_rebuild(
            &origins,
            vec3(field(case, "camera")),
            u32_of(field(case, "resolution")),
        );
        assert_eq!(
            got,
            rebuild_expectation(field(case, "expected")),
            "{}",
            case_why(case)
        );
    }
}

#[test]
fn rebuild_rounds_match_the_fixture() {
    let fixture = fixture();
    for case in array(field(&fixture, "rounds")) {
        let mut rounds = RebuildRounds::new(u32_of(field(case, "levels")));
        for step in array(field(case, "steps")) {
            match string(field(step, "op")) {
                "mark_stale" => rounds.mark_stale(),
                "start_waiting" => rounds.start_waiting(),
                "complete" => rounds.complete(u32_of(field(step, "level")) as usize),
                "choose_outdated" => {
                    let expected = match field(step, "expect") {
                        JsonValue::Null => None,
                        value => Some(u32_of(value) as usize),
                    };
                    assert_eq!(rounds.choose_outdated(), expected, "{}", case_why(case));
                }
                "pending" => assert_eq!(
                    rounds.pending(),
                    boolean(field(step, "expect")),
                    "{}",
                    case_why(case)
                ),
                other => panic!("unknown rounds step `{other}`"),
            }
        }
    }
}

#[test]
fn relight_cursor_matches_the_fixture() {
    let fixture = fixture();
    for case in array(field(&fixture, "relight")) {
        let levels = u32_of(field(case, "levels"));
        let mut cursor = RelightCursor::default();
        let expected = array(field(case, "expected"));
        for (i, call) in array(field(case, "calls")).iter().enumerate() {
            let being_rebuilt = match call {
                JsonValue::Null => None,
                value => Some(u32_of(value) as usize),
            };
            let want = u32_of(&expected[i]) as usize;
            assert_eq!(
                cursor.next(levels, being_rebuilt),
                want,
                "{}",
                case_why(case)
            );
        }
    }
}

#[test]
fn light_schedule_matches_the_fixture() {
    let fixture = fixture();
    for case in array(field(&fixture, "light_schedule")) {
        let expected: Vec<u32> = array(field(case, "expected")).iter().map(u32_of).collect();
        let got = light_schedule(u32_of(field(case, "levels")));
        assert_eq!(got, expected.as_slice(), "{}", case_why(case));
    }
}

#[test]
fn flood_steps_and_pass_count_match_the_fixture() {
    let fixture = fixture();
    for case in array(field(&fixture, "flood_steps")) {
        let expected: Vec<i32> = array(field(case, "expected"))
            .iter()
            .map(|value| integer(value) as i32)
            .collect();
        let got = flood_steps(u32_of(field(case, "resolution")));
        assert_eq!(got.to_vec(), expected, "{}", case_why(case));
    }
    let count = field(&fixture, "flood_pass_count");
    assert_eq!(
        flood_pass_count() as i64,
        integer(field(count, "expected")),
        "{}",
        case_why(count)
    );
}

#[test]
fn volume_bytes_matches_the_fixture() {
    let fixture = fixture();
    for case in array(field(&fixture, "volume_bytes")) {
        let quality = GiQuality {
            resolution: u32_of(field(case, "resolution")),
            levels: u32_of(field(case, "levels")),
            probe_rays: u32_of(field(case, "probe_rays")),
        };
        assert_eq!(
            volume_bytes(quality) as i64,
            integer(field(case, "expected")),
            "{}",
            case_why(case)
        );
    }
}

#[test]
fn forward_lod_matches_the_fixture() {
    let fixture = fixture();
    for case in array(field(&fixture, "forward_lod")) {
        let chain: Vec<LodLevel> = array(field(case, "errors"))
            .iter()
            .enumerate()
            .map(|(i, error)| LodLevel {
                mesh: MeshId(i as u32),
                error: number(error) as f32,
            })
            .collect();
        let got = forward_lod(
            &chain,
            number(field(case, "world_scale")) as f32,
            number(field(case, "voxel")) as f32,
        );
        assert_eq!(
            got as i64,
            integer(field(case, "expected")),
            "{}",
            case_why(case)
        );
    }
}

#[test]
fn level_view_is_the_gi_orthographic_view() {
    let view = level_view([-48, -48, -48], 0.2, 96);
    assert_eq!(view.allow_software, 2);
    assert_eq!(view.instance_filter, InstanceFilter::Gi as u32);
    assert_eq!(view.lod_scale, 1.0 / 0.2);
}

#[test]
fn voxel_size_doubles_per_level() {
    assert_eq!(voxel_size(0), 0.2);
    assert_eq!(voxel_size(4), 3.2);
}
