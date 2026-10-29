use raa_det::{DET_CASES, selftest};
use std::io::{self, Write};

// Shared with the integration test so it checks the actual example serializer.
pub fn write_json(mut out: impl Write) -> io::Result<()> {
    write!(out, "[")?;
    for (index, (seed, ticks)) in DET_CASES.into_iter().enumerate() {
        if index != 0 {
            write!(out, ",")?;
        }
        write!(
            out,
            "{{\"case\":\"det-selftest\",\"seed\":{seed},\"ticks\":{ticks},\"hash\":\"{:08x}\"}}",
            selftest(seed, ticks)
        )?;
    }
    writeln!(out, "]")
}

fn main() -> io::Result<()> {
    write_json(io::stdout().lock())
}
