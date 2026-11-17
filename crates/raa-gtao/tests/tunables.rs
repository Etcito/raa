//! S56 tunables of raa-gtao: keys in table order, defaults and ranges.

use raa_gtao::{AoQuality, tunables};

#[test]
fn keys_defaults_and_ranges_follow_the_table() {
    let t = tunables();
    let keys: Vec<&str> = t.iter().map(|s| s.key).collect();
    assert_eq!(keys, ["slices", "steps"]);
    assert_eq!(t[0].default, 4.0);
    assert_eq!(t[1].default, 8.0);
    assert_eq!((t[0].min, t[0].max), (Some(1.0), Some(8.0)));
    assert_eq!((t[1].min, t[1].max), (Some(1.0), Some(16.0)));
    assert!(t.iter().all(|s| s.integer && s.rebuild == "none"));
    let q = AoQuality::default();
    assert_eq!((q.slices, q.steps), (4, 8));
}

#[test]
fn validate_tunable_refuses_out_of_range_and_non_integer() {
    for (k, v) in [("slices", 9.0), ("steps", 17.0), ("steps", 1.5), ("x", 1.0)] {
        assert!(raa_pipeline::validate_tunable(tunables(), k, v).is_err());
    }
}
