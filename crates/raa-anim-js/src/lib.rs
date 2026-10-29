//! Host-testable case dispatch; wasm conversion lives in the discovered API modules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApiError {
    UnknownCase(String),
    InvalidInput(&'static str),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelftestResult {
    pub case: &'static str,
    pub seed: u32,
    pub ticks: u32,
    pub hash: String,
}

pub struct Case {
    pub name: &'static str,
    pub run: fn(u32, u32) -> u32,
}

include!(concat!(env!("OUT_DIR"), "/api_registry.rs"));

/// Validate before casting: JS-to-u32 coercion would silently wrap or truncate.
pub fn boundary_u32(value: f64, field: &'static str) -> Result<u32, ApiError> {
    if !value.is_finite() || value < 0.0 || value > u32::MAX as f64 || value.fract() != 0.0 {
        return Err(ApiError::InvalidInput(field));
    }
    Ok(value as u32)
}

pub fn hex8(hash: u32) -> String {
    format!("{hash:08x}")
}

pub fn dispatch(case: &str, seed: f64, ticks: f64) -> Result<SelftestResult, ApiError> {
    dispatch_cases(&registered_cases(), case, seed, ticks)
}

pub fn dispatch_cases(
    cases: &[Case],
    case: &str,
    seed: f64,
    ticks: f64,
) -> Result<SelftestResult, ApiError> {
    let entry = cases
        .iter()
        .find(|c| c.name == case)
        .ok_or_else(|| ApiError::UnknownCase(case.to_owned()))?;
    let seed = boundary_u32(seed, "seed")?;
    let ticks = boundary_u32(ticks, "ticks")?;
    Ok(SelftestResult {
        case: entry.name,
        seed,
        ticks,
        hash: hex8((entry.run)(seed, ticks)),
    })
}
