//! Q6: every invalid case, run as parse then build, is refused with exactly its expected
//! code and a non-empty message.

use raa_geom::{GeometryRegistry, Quality};
use raa_loader::prims::samples::invalid_cases;
use raa_loader::prims::{build, parse};
use raa_loader::{LoadContext, NoBooleans};

#[test]
fn every_invalid_case_is_refused_with_its_code() {
    let mut g = GeometryRegistry::new();
    raa_prims::register_all(&mut g).expect("register_all");
    let ctx = LoadContext {
        geometry: &g,
        booleans: &NoBooleans,
        quality: Quality::LOD0,
        strict_kinds: true,
    };
    let cases = invalid_cases();
    let mut failures = Vec::new();
    let mut refused = 0usize;
    for (label, text, expected) in &cases {
        match parse(text).and_then(|a| build(&a, &ctx)) {
            Ok(_) => failures.push(format!("{label}: accepted, expected {expected}")),
            Err(e) if e.code != *expected => {
                failures.push(format!("{label}: got {e}, expected {expected}"))
            }
            Err(e) if e.message.is_empty() => failures.push(format!("{label}: empty message")),
            Err(e) => {
                refused += 1;
                eprintln!("{label}: {e}");
            }
        }
    }
    eprintln!("refused {refused}/{}", cases.len());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!(refused, cases.len());
}
